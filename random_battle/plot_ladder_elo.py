"""Plot ladder Elo evolution from session logs.

Reads ``*_elo.jsonl`` (``--track_elo``) or ``cumulative.json`` and draws an
Elo-vs-battle curve (SVG by default; PNG if matplotlib is installed).

Examples::

    python random_battle/plot_ladder_elo.py
    python random_battle/plot_ladder_elo.py --username MBTIPE
    python random_battle/plot_ladder_elo.py \\
        --input random_battle/artifacts/sessions/MBTIPE_elo.jsonl \\
        --output random_battle/artifacts/ladder_MBTIPE.svg
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from random_battle.config import FORMAT_ID, SESSION_DIR  # noqa: E402


@dataclass
class EloPoint:
    battle_index: int
    elo: int
    won: Optional[bool] = None
    opponent: Optional[str] = None
    battle_tag: Optional[str] = None
    ts: Optional[str] = None
    delta: Optional[int] = None


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def load_elo_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return rows


def load_cumulative_elo(
    path: Path,
    *,
    username: Optional[str] = None,
    battle_format: Optional[str] = None,
) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    elo_block = data.get("elo") or {}
    if not elo_block:
        return []
    if username and battle_format:
        key = f"{username}:{battle_format}"
        block = elo_block.get(key)
        if isinstance(block, dict):
            return list(block.get("history") or [])
    # Single user/format or merge all histories chronologically.
    merged: List[Dict[str, Any]] = []
    for block in elo_block.values():
        if isinstance(block, dict):
            merged.extend(block.get("history") or [])
    merged.sort(key=lambda r: str(r.get("ts") or ""))
    return merged


def find_elo_jsonl(session_dir: Path, username: Optional[str]) -> Optional[Path]:
    pattern = "*_elo.jsonl" if not username else f"{username}_elo.jsonl"
    matches = sorted(session_dir.glob(pattern))
    if not matches and username:
        matches = sorted(session_dir.glob("*_elo.jsonl"))
    return matches[-1] if matches else None


def infer_post_battle_elo(
    entries: Sequence[Dict[str, Any]], index: int
) -> Optional[int]:
    row = entries[index]
    after = _to_int(row.get("rating_after"))
    if after is not None:
        return after
    if index + 1 < len(entries):
        nxt = _to_int(entries[index + 1].get("rating_before"))
        if nxt is not None:
            return nxt
    before = _to_int(row.get("rating_before"))
    delta = _to_int(row.get("rating_delta"))
    if before is not None and delta is not None:
        return before + delta
    return _to_int(row.get("session_elo")) or before


def build_elo_series(entries: Sequence[Dict[str, Any]]) -> List[EloPoint]:
    if not entries:
        return []

    series: List[EloPoint] = []
    first_before = _to_int(entries[0].get("rating_before"))
    if first_before is not None:
        series.append(
            EloPoint(
                battle_index=0,
                elo=first_before,
                won=None,
                opponent=None,
                battle_tag=None,
                ts=entries[0].get("ts"),
            )
        )

    for i, row in enumerate(entries):
        post = infer_post_battle_elo(entries, i)
        if post is None:
            continue
        before = _to_int(row.get("rating_before"))
        delta = (post - before) if before is not None else _to_int(row.get("rating_delta"))
        series.append(
            EloPoint(
                battle_index=i + 1,
                elo=post,
                won=row.get("won"),
                opponent=row.get("opponent"),
                battle_tag=row.get("battle_tag"),
                ts=row.get("ts"),
                delta=delta,
            )
        )
    return series


def summarize_series(series: Sequence[EloPoint]) -> Dict[str, Any]:
    if not series:
        return {}
    battles = [p for p in series if p.battle_index > 0]
    start = series[0].elo
    end = series[-1].elo
    peak = max(p.elo for p in series)
    trough = min(p.elo for p in series)
    wins = sum(1 for p in battles if p.won is True)
    losses = sum(1 for p in battles if p.won is False)
    return {
        "battles": len(battles),
        "start_elo": start,
        "end_elo": end,
        "delta": end - start,
        "peak": peak,
        "trough": trough,
        "wins": wins,
        "losses": losses,
    }


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_svg(
    series: Sequence[EloPoint],
    *,
    title: str,
    width: int = 960,
    height: int = 480,
) -> str:
    if len(series) < 2:
        raise ValueError("Need at least 2 Elo points to plot.")

    margin_l, margin_r, margin_t, margin_b = 64, 24, 48, 56
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    xs = [p.battle_index for p in series]
    ys = [p.elo for p in series]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        y_min -= 25
        y_max += 25
    y_pad = max(12, int((y_max - y_min) * 0.08))
    y_min -= y_pad
    y_max += y_pad

    def x_px(x: float) -> float:
        if x_max == x_min:
            return margin_l + plot_w / 2
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def y_px(y: float) -> float:
        return margin_t + (y_max - y) / (y_max - y_min) * plot_h

    # Grid lines (Elo).
    y_ticks = 5
    lines: List[str] = []
    for i in range(y_ticks + 1):
        elo = y_min + (y_max - y_min) * i / y_ticks
        y = y_px(elo)
        lines.append(
            f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" '
            f'y2="{y:.1f}" stroke="#e8e8e8" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#555">{int(round(elo))}</text>'
        )

    # X ticks.
    x_step = max(1, (x_max - x_min) // 10)
    x_tick = x_min
    while x_tick <= x_max:
        x = x_px(x_tick)
        lines.append(
            f'<line x1="{x:.1f}" y1="{margin_t}" x2="{x:.1f}" '
            f'y2="{height - margin_b}" stroke="#f0f0f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{height - margin_b + 20}" text-anchor="middle" '
            f'font-size="11" fill="#555">{x_tick}</text>'
        )
        x_tick += x_step

    # Main polyline.
    path_pts = " ".join(f"{x_px(p.battle_index):.1f},{y_px(p.elo):.1f}" for p in series)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="16" '
        f'font-weight="600" fill="#222">{_svg_escape(title)}</text>',
        *lines,
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{path_pts}"/>',
    ]

    # Points: green win, red loss, gray start / unknown.
    for p in series:
        cx, cy = x_px(p.battle_index), y_px(p.elo)
        if p.battle_index == 0:
            color = "#6b7280"
        elif p.won is True:
            color = "#16a34a"
        elif p.won is False:
            color = "#dc2626"
        else:
            color = "#2563eb"
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{color}" '
            f'stroke="#fff" stroke-width="1.2">'
            f'<title>#{p.battle_index} Elo {p.elo}'
            f'{f" ({p.delta:+d})" if p.delta is not None else ""}'
            f'{f" vs {p.opponent}" if p.opponent else ""}</title></circle>'
        )

    parts.append(
        f'<text x="{width / 2:.1f}" y="{height - 12}" text-anchor="middle" '
        f'font-size="12" fill="#444">Combat #</text>'
    )
    parts.append(
        f'<text x="16" y="{height / 2:.1f}" text-anchor="middle" font-size="12" '
        f'fill="#444" transform="rotate(-90 16 {height / 2:.1f})">Elo</text>'
    )
    # Legend.
    legend = [
        ("Départ", "#6b7280"),
        ("Victoire", "#16a34a"),
        ("Défaite", "#dc2626"),
    ]
    lx = width - margin_r - 120
    ly = margin_t + 8
    for label, color in legend:
        parts.append(
            f'<circle cx="{lx}" cy="{ly}" r="4" fill="{color}"/>'
            f'<text x="{lx + 10}" y="{ly + 4}" font-size="11" fill="#444">'
            f'{_svg_escape(label)}</text>'
        )
        ly += 18

    parts.append("</svg>")
    return "\n".join(parts)


def render_matplotlib(
    series: Sequence[EloPoint],
    *,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    xs = [p.battle_index for p in series]
    ys = [p.elo for p in series]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(xs, ys, color="#2563eb", linewidth=2, zorder=1)
    for p in series:
        if p.battle_index == 0:
            color = "#6b7280"
        elif p.won is True:
            color = "#16a34a"
        elif p.won is False:
            color = "#dc2626"
        else:
            color = "#2563eb"
        ax.scatter(
            [p.battle_index],
            [p.elo],
            c=color,
            s=36,
            zorder=2,
            edgecolors="white",
            linewidths=0.8,
        )
    ax.set_title(title)
    ax.set_xlabel("Combat #")
    ax.set_ylabel("Elo")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def print_summary(stats: Dict[str, Any], *, source: Path) -> None:
    if not stats:
        print(f"Aucune donnée Elo dans {source}")
        return
    delta = stats["delta"]
    sign = "+" if delta >= 0 else ""
    print(f"Source : {source}")
    print(
        f"Combats : {stats['battles']} | "
        f"W/L : {stats['wins']}/{stats['losses']} | "
        f"Elo : {stats['start_elo']} -> {stats['end_elo']} ({sign}{delta}) | "
        f"Pic : {stats['peak']} | Creux : {stats['trough']}"
    )


def resolve_input(
    *,
    input_path: Optional[Path],
    cumulative_path: Path,
    session_dir: Path,
    username: Optional[str],
    battle_format: str,
    use_cumulative: bool,
) -> Tuple[Path, List[Dict[str, Any]]]:
    if input_path is not None:
        path = input_path
        if path.suffix == ".json" and "cumulative" in path.name:
            return path, load_cumulative_elo(
                path, username=username, battle_format=battle_format
            )
        return path, load_elo_jsonl(path)

    if use_cumulative and cumulative_path.is_file():
        rows = load_cumulative_elo(
            cumulative_path, username=username, battle_format=battle_format
        )
        if rows:
            return cumulative_path, rows

    found = find_elo_jsonl(session_dir, username)
    if found is not None:
        return found, load_elo_jsonl(found)

    raise FileNotFoundError(
        "Aucun historique Elo trouvé. Lance le ladder avec --track_elo "
        f"(fichiers attendus dans {session_dir})."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Courbe Elo ladder du bot.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Fichier *_elo.jsonl ou cumulative.json.",
    )
    parser.add_argument("--username", default=None, help="Filtrer un compte Showdown.")
    parser.add_argument("--format", default=FORMAT_ID, dest="battle_format")
    parser.add_argument(
        "--session_dir",
        type=Path,
        default=SESSION_DIR,
        help="Dossier des sessions (défaut: artifacts/sessions).",
    )
    parser.add_argument(
        "--cumulative",
        type=Path,
        default=SESSION_DIR / "cumulative.json",
    )
    parser.add_argument(
        "--prefer_cumulative",
        action="store_true",
        help="Lire cumulative.json avant le JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Fichier de sortie (.svg ou .png). Défaut: artifacts/ladder_<user>.svg",
    )
    parser.add_argument(
        "--matplotlib",
        action="store_true",
        help="Utiliser matplotlib (PNG) si installé.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Ouvrir le graphique dans le navigateur (SVG/HTML).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Titre du graphique.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        source, rows = resolve_input(
            input_path=args.input,
            cumulative_path=args.cumulative,
            session_dir=args.session_dir,
            username=args.username,
            battle_format=args.battle_format,
            use_cumulative=args.prefer_cumulative,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    series = build_elo_series(rows)
    if len(series) < 2:
        print(
            f"Pas assez de points Elo dans {source} "
            f"({len(series)} point(s)). Lance plus de combats ladder avec --track_elo.",
            file=sys.stderr,
        )
        return 1

    stats = summarize_series(series)
    print_summary(stats, source=source)

    user = args.username
    if not user:
        for row in rows:
            if row.get("username"):
                user = str(row["username"])
                break
    safe_user = user or "bot"
    out = args.output
    if out is None:
        out = SESSION_DIR.parent / f"ladder_{safe_user}.svg"

    title = args.title or f"Ladder {safe_user} — {args.battle_format}"

    use_mpl = args.matplotlib and out.suffix.lower() == ".png"
    if use_mpl:
        try:
            render_matplotlib(series, title=title, output_path=out)
        except ImportError:
            print("matplotlib absent — export SVG à la place.", file=sys.stderr)
            out = out.with_suffix(".svg")
            use_mpl = False

    if not use_mpl:
        if out.suffix.lower() not in (".svg", ".html"):
            out = out.with_suffix(".svg")
        svg = render_svg(series, title=title)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".html":
            html = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<title>{_svg_escape(title)}</title></head>"
                f"<body style='margin:0;background:#f5f5f5;display:flex;"
                "justify-content:center;padding:24px'>"
                f"{svg}</body></html>"
            )
            out.write_text(html, encoding="utf-8")
        else:
            out.write_text(svg, encoding="utf-8")

    print(f"Graphique : {out.resolve()}")

    if args.open:
        webbrowser.open(out.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
