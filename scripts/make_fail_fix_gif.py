"""Render a fail-then-fix agent loop as a GIF."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from agent.loop import run_episode
from agent.policies import recovery_policy
from katas.bank import by_id

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "figures" / "fail_fix.gif"

W, H = 720, 400
BG, PANEL = (15, 23, 42), (30, 41, 59)
FG, DIM = (226, 232, 240), (148, 163, 184)
GREEN, RED, BLUE, AMBER = (52, 211, 153), (248, 113, 113), (96, 165, 250), (251, 191, 36)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("consola.ttf", "cascadiamono.ttf", "cascadia.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    words = text.replace("\n", " ").split()
    lines, cur = [], ""
    for word in words:
        nxt = (cur + " " + word).strip()
        if draw.textlength(nxt, font=font) <= width:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def _scene(title: str, lines: list[tuple[str, tuple[int, int, int]]]) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    title_font, body_font = _font(22), _font(18)
    draw.rectangle((0, 0, W, 48), fill=PANEL)
    draw.text((20, 12), title, fill=BLUE, font=title_font)
    y = 68
    for text, color in lines:
        for part in _wrap(draw, text, body_font, W - 40):
            draw.text((20, y), part, fill=color, font=body_font)
            y += 28
            if y > H - 24:
                return img
    return img


def main() -> None:
    kata = by_id("hand_002")
    st = run_episode(kata, recovery_policy)
    frames = [
        _scene(
            "KataAgent",
            [
                ("Task: write add(a, b)", FG),
                ("Return the sum of two numbers.", DIM),
                ("write  ->  test  ->  fix", AMBER),
            ],
        )
    ]
    for turn in st.transcript:
        tool = turn.get("tool") or "invalid"
        obs = str(turn.get("obs") or "")
        code = ""
        if tool == "write_solution":
            code = (turn.get("args") or {}).get("code", "").strip().replace("\n", "  ")
        bad = "broken" in code or "fail" in obs.lower() or "Error" in obs or "Assertion" in obs
        good = ("pass" in obs.lower() and "fail" not in obs.lower()) or obs == "ok"
        color = RED if bad else GREEN if good else FG
        rows = [(f"> {tool}", BLUE)]
        if code:
            rows.append((code[:72], RED if "broken" in code else FG))
        rows.append((obs.replace("\n", "  ")[:120], color))
        frames.append(_scene("fail then fix", rows))
    ok = st.hidden_ok()
    frames.append(
        _scene(
            "done",
            [
                ("hidden tests PASS" if ok else "hidden tests FAIL", GREEN if ok else RED),
                ("First write fails the tests.", DIM),
                ("Second write is the fix.", DIM),
            ],
        )
    )
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=1200,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"wrote {OUT} frames={len(frames)} size={OUT.stat().st_size} hidden_ok={ok}")


if __name__ == "__main__":
    main()
