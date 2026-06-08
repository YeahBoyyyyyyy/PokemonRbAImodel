"""Patch poke-env login for local Showdown (guest assertion via play.pokemonshowdown.com)."""

from __future__ import annotations

import json
from typing import List

import requests

from poke_env.ps_client import ps_client as ps_client_module


def _parse_assertion_response(text: str) -> str:
    body = text[1:] if text.startswith("]") else text
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return ""
    if isinstance(payload, dict):
        return str(payload.get("assertion", "") or "")
    return ""


def apply_local_login_patch() -> None:
    """Allow password-less bots to log in on a local server."""
    if getattr(apply_local_login_patch, "_applied", False):
        return

    original_log_in = ps_client_module.PSClient.log_in

    async def patched_log_in(self, split_message: List[str]):
        if self.account_configuration.password:
            return await original_log_in(self, split_message)

        challstr = f"{split_message[2]}%7C{split_message[3]}"
        raw_name = self.account_configuration.username
        userid = "".join(ch for ch in raw_name.lower() if ch.isalnum())
        ws_url = getattr(self.server_configuration, "websocket_url", "") or ""
        is_local = "localhost" in ws_url or "127.0.0.1" in ws_url

        auth_urls = []
        if is_local:
            auth_urls.append("http://localhost:8000/action.php?")
        auth_urls.append(self.server_configuration.authentication_url)

        assertion = ""
        for auth_url in auth_urls:
            try:
                response = requests.post(
                    auth_url,
                    data={
                        "act": "getassertion",
                        "userid": userid,
                        "challstr": challstr,
                    },
                    timeout=15,
                )
                assertion = _parse_assertion_response(response.text)
            except Exception:
                assertion = ""
            if assertion:
                break

        if not assertion:
            if is_local:
                self.logger.warning(
                    "getassertion failed for %s — token vide (OK si serveur lance avec --no-security)",
                    self.username,
                )
            else:
                self.logger.warning(
                    "getassertion failed for %s — fallback empty token",
                    self.username,
                )
            assertion = ""

        await self.send_message(f"/trn {self.username},0,{assertion}")
        await self.change_avatar(self._avatar)

    ps_client_module.PSClient.log_in = patched_log_in  # type: ignore[method-assign]
    apply_local_login_patch._applied = True  # type: ignore[attr-defined]
