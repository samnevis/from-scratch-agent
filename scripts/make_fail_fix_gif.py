"""Render a fail-then-fix agent loop as a GIF (stdlib only)."""

from __future__ import annotations

import struct
from pathlib import Path

from agent.loop import run_episode
from agent.policies import recovery_policy
from katas.bank import by_id

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "figures" / "fail_fix.gif"

W, H = 640, 360
BG, FG, DIM = (15, 23, 42), (226, 232, 240), (148, 163, 184)
GREEN, RED, BLUE, AMBER = (52, 211, 153), (248, 113, 113), (96, 165, 250), (251, 191, 36)
PANEL = (30, 41, 59)

# 5x7 bitmaps, one string row per line, '#' = on.
_FONT = {
    "A": [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "B": ["#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "],
    "C": [" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "],
    "D": ["#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "],
    "E": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"],
    "F": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "],
    "G": [" ### ", "#   #", "#    ", "# ###", "#   #", "#   #", " ### "],
    "H": ["#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "I": [" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    "J": ["  ###", "   # ", "   # ", "   # ", "   # ", "#  # ", " ##  "],
    "K": ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"],
    "L": ["#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"],
    "M": ["#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #"],
    "N": ["#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"],
    "O": [" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "P": ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "],
    "Q": [" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"],
    "R": ["#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"],
    "S": [" ### ", "#   #", "#    ", " ### ", "    #", "#   #", " ### "],
    "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
    "U": ["#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "],
    "V": ["#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "],
    "W": ["#   #", "#   #", "#   #", "# # #", "# # #", "## ##", "#   #"],
    "X": ["#   #", " # # ", "  #  ", "  #  ", "  #  ", " # # ", "#   #"],
    "Y": ["#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
    "Z": ["#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"],
    "a": ["     ", "     ", " ### ", "    #", " ####", "#   #", " ####"],
    "b": ["#    ", "#    ", "#### ", "#   #", "#   #", "#   #", "#### "],
    "c": ["     ", "     ", " ### ", "#    ", "#    ", "#   #", " ### "],
    "d": ["    #", "    #", " ####", "#   #", "#   #", "#   #", " ####"],
    "e": ["     ", "     ", " ### ", "#   #", "#####", "#    ", " ### "],
    "f": ["  ## ", " #  #", " #   ", "###  ", " #   ", " #   ", " #   "],
    "g": ["     ", "     ", " ####", "#   #", " ####", "    #", " ### "],
    "h": ["#    ", "#    ", "#### ", "#   #", "#   #", "#   #", "#   #"],
    "i": ["  #  ", "     ", " ##  ", "  #  ", "  #  ", "  #  ", " ### "],
    "j": ["   # ", "     ", "  ## ", "   # ", "   # ", "#  # ", " ##  "],
    "k": ["#    ", "#    ", "#  # ", "# #  ", "##   ", "# #  ", "#  # "],
    "l": [" ##  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    "m": ["     ", "     ", "## # ", "# # #", "# # #", "#   #", "#   #"],
    "n": ["     ", "     ", "#### ", "#   #", "#   #", "#   #", "#   #"],
    "o": ["     ", "     ", " ### ", "#   #", "#   #", "#   #", " ### "],
    "p": ["     ", "     ", "#### ", "#   #", "#### ", "#    ", "#    "],
    "q": ["     ", "     ", " ####", "#   #", " ####", "    #", "    #"],
    "r": ["     ", "     ", "# ## ", "##   ", "#    ", "#    ", "#    "],
    "s": ["     ", "     ", " ####", "#    ", " ### ", "    #", "#### "],
    "t": [" #   ", " #   ", "#### ", " #   ", " #   ", " #  #", "  ## "],
    "u": ["     ", "     ", "#   #", "#   #", "#   #", "#  ##", " ## #"],
    "v": ["     ", "     ", "#   #", "#   #", "#   #", " # # ", "  #  "],
    "w": ["     ", "     ", "#   #", "#   #", "# # #", "# # #", " # # "],
    "x": ["     ", "     ", "#   #", " # # ", "  #  ", " # # ", "#   #"],
    "y": ["     ", "     ", "#   #", "#   #", " ####", "    #", " ### "],
    "z": ["     ", "     ", "#####", "   # ", "  #  ", " #   ", "#####"],
    "0": [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
    "1": ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    "2": [" ### ", "#   #", "    #", "  ## ", " #   ", "#    ", "#####"],
    "3": [" ### ", "#   #", "    #", "  ## ", "    #", "#   #", " ### "],
    "4": ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
    "5": ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
    "6": [" ### ", "#    ", "#    ", "#### ", "#   #", "#   #", " ### "],
    "7": ["#####", "    #", "   # ", "  #  ", "  #  ", "  #  ", "  #  "],
    "8": [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
    "9": [" ### ", "#   #", "#   #", " ####", "    #", "    #", " ### "],
    " ": ["     ", "     ", "     ", "     ", "     ", "     ", "     "],
    ".": ["     ", "     ", "     ", "     ", "     ", "  ## ", "  ## "],
    ",": ["     ", "     ", "     ", "     ", "     ", "  ## ", "  #  "],
    ":": ["     ", "  ## ", "  ## ", "     ", "  ## ", "  ## ", "     "],
    "-": ["     ", "     ", "     ", " ### ", "     ", "     ", "     "],
    "_": ["     ", "     ", "     ", "     ", "     ", "     ", "#####"],
    "+": ["     ", "  #  ", "  #  ", "#####", "  #  ", "  #  ", "     "],
    "=": ["     ", "     ", "#####", "     ", "#####", "     ", "     "],
    "/": ["    #", "   # ", "   # ", "  #  ", " #   ", " #   ", "#    "],
    "(": ["  ## ", " #   ", "#    ", "#    ", "#    ", " #   ", "  ## "],
    ")": [" ##  ", "   # ", "    #", "    #", "    #", "   # ", " ##  "],
    "[": [" ### ", " #   ", " #   ", " #   ", " #   ", " #   ", " ### "],
    "]": [" ### ", "   # ", "   # ", "   # ", "   # ", "   # ", " ### "],
    "{": ["  ## ", " #   ", " #   ", "##   ", " #   ", " #   ", "  ## "],
    "}": [" ##  ", "   # ", "   # ", "   ##", "   # ", "   # ", " ##  "],
    "'": ["  #  ", "  #  ", " #   ", "     ", "     ", "     ", "     "],
    '"': [" # # ", " # # ", "     ", "     ", "     ", "     ", "     "],
    "!": ["  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "     ", "  #  "],
    "?": [" ### ", "#   #", "    #", "  ## ", "  #  ", "     ", "  #  "],
    ">": ["#    ", " #   ", "  #  ", "   # ", "  #  ", " #   ", "#    "],
    "<": ["    #", "   # ", "  #  ", " #   ", "  #  ", "   # ", "    #"],
    "|": ["  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "],
    "*": ["     ", " # # ", "  #  ", "#####", "  #  ", " # # ", "     "],
    "#": [" # # ", "#####", " # # ", " # # ", "#####", " # # ", "     "],
    "%": ["##  #", "## # ", "  #  ", " #   ", " #  ##", "#   ##", "     "],
    "&": [" ##  ", "#  # ", "# #  ", " ##  ", "# # #", "#  # ", " ## #"],
    "@": [" ### ", "#   #", "# ###", "# # #", "# ###", "#    ", " ### "],
    "^": ["  #  ", " # # ", "#   #", "     ", "     ", "     ", "     "],
    "~": ["     ", "     ", " ## #", "#  # ", "     ", "     ", "     "],
    "`": [" #   ", "  #  ", "     ", "     ", "     ", "     ", "     "],
}


def _px(frame, x, y, color) -> None:
    if 0 <= x < W and 0 <= y < H:
        frame[y][x] = color


def _rect(frame, x, y, w, h, color) -> None:
    for yy in range(max(0, y), min(H, y + h)):
        for xx in range(max(0, x), min(W, x + w)):
            frame[yy][xx] = color


def _text(frame, x, y, s, color=FG, scale=2) -> None:
    cx = x
    for ch in s:
        glyph = _FONT.get(ch) or _FONT.get(ch.upper()) or _FONT["?"]
        for row, line in enumerate(glyph):
            for col, bit in enumerate(line):
                if bit == "#":
                    for dy in range(scale):
                        for dx in range(scale):
                            _px(frame, cx + col * scale + dx, y + row * scale + dy, color)
        cx += 6 * scale


def _wrap(text: str, width: int) -> list[str]:
    words = text.replace("\n", " | ").split()
    lines, cur = [], ""
    for w in words:
        nxt = (cur + " " + w).strip()
        if len(nxt) > width:
            if cur:
                lines.append(cur)
            cur = w
        else:
            cur = nxt
    if cur:
        lines.append(cur)
    return lines or [""]


def _lzw_encode(indexes: list[int], min_code: int = 8) -> bytes:
    clear, eoi = 1 << min_code, (1 << min_code) + 1
    next_code, code_size = eoi + 1, min_code + 1
    table = {bytes([i]): i for i in range(clear)}
    buf = nbits = 0
    out = bytearray()

    def emit(code: int) -> None:
        nonlocal buf, nbits, code_size
        buf |= code << nbits
        nbits += code_size
        while nbits >= 8:
            out.append(buf & 0xFF)
            buf >>= 8
            nbits -= 8

    emit(clear)
    w = bytes([indexes[0]])
    for idx in indexes[1:]:
        k = bytes([idx])
        wk = w + k
        if wk in table:
            w = wk
            continue
        emit(table[w])
        if next_code < 4096:
            table[wk] = next_code
            next_code += 1
            if next_code == (1 << code_size) and code_size < 12:
                code_size += 1
        else:
            emit(clear)
            table = {bytes([i]): i for i in range(clear)}
            next_code, code_size = eoi + 1, min_code + 1
        w = k
    emit(table[w])
    emit(eoi)
    if nbits:
        out.append(buf & 0xFF)
    return bytes(out)


def write_gif(frames, path: Path, delay_cs: int = 110) -> None:
    palette = [BG, FG, DIM, GREEN, RED, BLUE, AMBER, PANEL, (51, 65, 85)]
    while len(palette) < 256:
        palette.append((0, 0, 0))
    pal = b"".join(struct.pack("BBB", *c) for c in palette[:256])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"GIF89a")
        f.write(struct.pack("<HH", W, H))
        f.write(bytes([0xF7, 0, 0]))
        f.write(pal)
        f.write(b"!\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00")
        for frame in frames:
            f.write(b"!\xF9\x04\x09" + struct.pack("<H", delay_cs) + b"\x00\x00,")
            f.write(struct.pack("<HHHH", 0, 0, W, H) + b"\x00\x08")
            cache = {c: i for i, c in enumerate(palette[:9])}
            indexes = [cache.get(px, 0) for row in frame for px in row]
            data = _lzw_encode(indexes, 8)
            i = 0
            while i < len(data):
                chunk = data[i : i + 255]
                f.write(bytes([len(chunk)]))
                f.write(chunk)
                i += 255
            f.write(b"\x00")
        f.write(b";")


def _scene(title: str, lines: list[tuple[str, tuple[int, int, int]]]):
    frame = [[BG for _ in range(W)] for _ in range(H)]
    _rect(frame, 0, 0, W, 40, PANEL)
    _text(frame, 18, 12, title, BLUE, 2)
    y = 58
    for text, color in lines:
        for part in _wrap(text, 48):
            _text(frame, 18, y, part, color, 2)
            y += 24
            if y > H - 20:
                return frame
    return frame


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
        lines = [(f"> {tool}", BLUE)]
        if code:
            lines.append((code[:70], RED if "broken" in code else FG))
        for ln in _wrap(obs, 48)[:5]:
            lines.append((ln, color))
        frames.append(_scene("fail then fix", lines))
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
    write_gif(frames, OUT)
    print(f"wrote {OUT} frames={len(frames)} hidden_ok={ok}")


if __name__ == "__main__":
    main()
