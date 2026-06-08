"""Python client for the Node.js sim_bridge.

Spawns ``sim_bridge/bridge.js`` once and talks to it via a JSON-line stdin/
stdout protocol. The bridge wraps Pokemon Showdown's simulator (@pkmn/sim)
and lets us run real Gen-9 turns from Python: build a battle from two
teams, then apply (p1_choice, p2_choice) pairs and read the resulting
state.

Typical usage
-------------

    bridge = PkmnBridge()
    snap = bridge.init(p1_team=our_sets, p2_team=opp_sets)
    # snap['requests']['p1'] tells us what to choose
    snap = bridge.step(snap['state'], p1_choice='team 123456', p2_choice='team 123456')
    snap = bridge.step(snap['state'], p1_choice='move 1', p2_choice='move 1')
    print(snap['turn'], snap['ended'])
    bridge.close()

The bridge keeps no per-call state: every ``step`` is given the serialized
state of the previous turn. That lets us fork the simulation cheaply for
search.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE_DIR = PROJECT_ROOT / "sim_bridge"
DEFAULT_BRIDGE_SCRIPT = DEFAULT_BRIDGE_DIR / "bridge.js"


class PkmnBridgeError(RuntimeError):
    """Raised when the Node bridge returns an error or dies unexpectedly."""


class PkmnBridge:
    """One-shot Node.js subprocess wrapping @pkmn/sim.

    Thread-safe: a single internal lock ensures stdin/stdout pairs are not
    interleaved across threads.
    """

    def __init__(
        self,
        bridge_script: Optional[Path] = None,
        *,
        node_executable: str = "node",
        startup_timeout_s: float = 5.0,
        call_timeout_s: float = 10.0,
        cwd: Optional[Path] = None,
    ) -> None:
        self.script_path = Path(bridge_script or DEFAULT_BRIDGE_SCRIPT)
        if not self.script_path.exists():
            raise PkmnBridgeError(f"Bridge script not found: {self.script_path}")
        self.cwd = Path(cwd or self.script_path.parent)
        self.node_executable = node_executable
        self.call_timeout_s = call_timeout_s

        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._start(startup_timeout_s=startup_timeout_s)

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def _start(self, *, startup_timeout_s: float) -> None:
        # On Windows, ensure no stderr blocking by letting it inherit.
        creationflags = 0
        if os.name == "nt":
            # CREATE_NO_WINDOW so spawning doesn't flash a console window.
            creationflags = 0x08000000

        self._proc = subprocess.Popen(
            [self.node_executable, str(self.script_path)],
            cwd=str(self.cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            encoding="utf-8",
            creationflags=creationflags,
        )
        # Ping with a short deadline to confirm the bridge is up.
        deadline = time.monotonic() + startup_timeout_s
        while time.monotonic() < deadline:
            try:
                resp = self._raw_call({"cmd": "ping"})
                if resp.get("ok") and resp.get("pong"):
                    return
            except PkmnBridgeError:
                time.sleep(0.05)
                continue
        raise PkmnBridgeError("Bridge did not respond to ping within timeout")

    def close(self) -> None:
        if not self._proc:
            return
        try:
            try:
                self._raw_call({"cmd": "quit"}, timeout=1.0)
            except PkmnBridgeError:
                pass
        finally:
            try:
                self._proc.stdin and self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None

    def __enter__(self) -> "PkmnBridge":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Raw JSON-line communication
    # ------------------------------------------------------------------

    def _raw_call(self, payload: Dict[str, Any], *, timeout: Optional[float] = None) -> Dict[str, Any]:
        if not self._proc or self._proc.poll() is not None:
            raise PkmnBridgeError("Bridge subprocess is not running")
        timeout = timeout if timeout is not None else self.call_timeout_s
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise PkmnBridgeError(f"Failed to write to bridge: {exc}") from exc
            assert self._proc.stdout is not None
            # readline does not honor a timeout natively; use a thread.
            response_line = _readline_with_timeout(self._proc, timeout)
        if response_line is None:
            raise PkmnBridgeError("Bridge timed out")
        if not response_line.strip():
            raise PkmnBridgeError("Bridge returned empty line")
        try:
            return json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise PkmnBridgeError(f"Bridge sent invalid JSON: {response_line[:120]!r}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        resp = self._raw_call({"cmd": "ping"})
        return bool(resp.get("ok") and resp.get("pong"))

    def init(
        self,
        *,
        p1_team: Sequence[Dict[str, Any]],
        p2_team: Sequence[Dict[str, Any]],
        format_id: str = "gen9customgame",
        p1_name: str = "P1",
        p2_name: str = "P2",
        seed: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "cmd": "init",
            "format": format_id,
            "p1": {"name": p1_name, "team": list(p1_team)},
            "p2": {"name": p2_name, "team": list(p2_team)},
        }
        if seed is not None:
            payload["seed"] = list(seed)
        resp = self._raw_call(payload)
        _check_ok(resp)
        return resp

    def step(
        self,
        state: Dict[str, Any],
        *,
        p1_choice: Optional[str] = None,
        p2_choice: Optional[str] = None,
        seed: Optional[Sequence[int]] = None,
        include_log: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "cmd": "step",
            "state": state,
            "p1_choice": p1_choice,
            "p2_choice": p2_choice,
            "include_log": include_log,
        }
        if seed is not None:
            payload["seed"] = list(seed)
        resp = self._raw_call(payload)
        _check_ok(resp)
        return resp

    def requests(self, state: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._raw_call({"cmd": "requests", "state": state})
        _check_ok(resp)
        return resp


def _check_ok(resp: Dict[str, Any]) -> None:
    if not resp.get("ok", False):
        raise PkmnBridgeError(resp.get("error") or "Bridge reported failure")


# ---------------------------------------------------------------------------
# Timeout-aware readline (the Node process is line-buffered)
# ---------------------------------------------------------------------------


def _readline_with_timeout(proc: subprocess.Popen, timeout: float) -> Optional[str]:
    """Read one line from proc.stdout with a wall-clock timeout."""
    assert proc.stdout is not None
    container: List[Optional[str]] = [None]

    def reader() -> None:
        try:
            container[0] = proc.stdout.readline()
        except Exception:
            container[0] = None

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        return None
    return container[0]


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def battle_ended(snap: Dict[str, Any]) -> bool:
    return bool(snap.get("ended"))


def winner_of(snap: Dict[str, Any]) -> Optional[str]:
    winner = snap.get("winner")
    return winner if isinstance(winner, str) and winner else None


def teampreview_choice(side_size: int = 6) -> str:
    """Default team-preview ordering: keep slots in their declared order."""
    return "team " + "".join(str(i + 1) for i in range(side_size))
