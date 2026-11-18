#!/usr/bin/env python3
"""Measure how often a PCSX2 game uses each of its textures, by sampling frames with RenderDoc while you play.

Pick a game, then play it in the PCSX2 window that opens. Every --interval seconds one frame is captured
(a short freeze), the textures drawn in it are counted, and the capture is deleted. Counts are matched to the
names in <textures>/<GAME>/dumps and replacements/, and accumulate in <textures>/<GAME>/usage.json across sessions.
Close PCSX2 to end the session and print the report.

Usage: texture_usage.py [textures_dir] [--pcsx2 PATH] [--renderdoc DIR] [--interval S] [--top N] [--report] [--reset]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from pcsx2_upscale import DEFAULT_TEXTURES, IMAGE_EXTS, choose, load_game_titles, load_skip_list

ROOT = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache" / "pcsx2-upscaler"
USAGE_FILE = "usage.json"

# Bytes per 4x4 block of the compressed formats a .dds replacement can use
DXGI_BLOCK = {71: 8, 72: 8, 74: 16, 75: 16, 77: 16, 78: 16, 80: 8, 81: 8, 83: 16, 84: 16, 95: 16, 96: 16, 98: 16, 99: 16}
FOURCC_BLOCK = {b"DXT1": 8, b"DXT3": 16, b"DXT5": 16, b"ATI1": 8, b"BC4U": 8, b"ATI2": 16, b"BC5U": 16}


def dds_level0(path: Path) -> tuple[int, int, bytes] | None:
    """Width, height and the raw top mip level of a block-compressed .dds: what PCSX2 uploads to the GPU."""
    with open(path, "rb") as f:
        head = f.read(148)
        if len(head) < 128 or head[:4] != b"DDS ":
            return None
        h, w = struct.unpack_from("<II", head, 12)
        fourcc = head[84:88]
        block = DXGI_BLOCK.get(struct.unpack_from("<I", head, 128)[0]) if fourcc == b"DX10" else FOURCC_BLOCK.get(fourcc)
        if block is None:
            return None
        f.seek(148 if fourcc == b"DX10" else 128)
        return w, h, f.read(max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * block)


def texture_hash(path: Path) -> tuple[str, int, int] | None:
    """Content hash of a texture as it sits on the GPU: .dds by its compressed data, images by their RGBA pixels."""
    try:
        if path.suffix.lower() == ".dds":
            r = dds_level0(path)
            if r is None:
                return None
            w, h, blob = r
        else:
            with Image.open(path) as im:
                blob = np.asarray(im.convert("RGBA")).tobytes()
                w, h = im.size
        return hashlib.sha1(blob).hexdigest(), w, h
    except Exception:
        return None


def build_index(game: Path) -> Path:
    """Hashes every dump and replacement once; later runs only hash new or changed files."""
    cache_dir = CACHE / "usage" / game.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "hashes.json"
    try:
        cache = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        cache = {}
    files = [p for d in ("dumps", "replacements") if (game / d).is_dir()
             for p in (game / d).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    fresh, done = {}, 0
    for p in files:
        key, st = str(p.relative_to(game)), p.stat()
        stamp = [st.st_mtime_ns, st.st_size]
        entry = cache.get(key)
        if not entry or entry["stamp"] != stamp:
            r = texture_hash(p)
            entry = {"stamp": stamp, "hash": r and list(r)}
            done += 1
            if done % 200 == 0:
                print(f"\rIndexing textures... {done}", end="", flush=True)
        fresh[key] = entry
    if done:
        print(f"\rIndexed {done} new or changed textures.")
    cache_file.write_text(json.dumps(fresh))

    # Names with identical content (same texture, different palette hash) can't be told apart on the GPU:
    # they're counted together under the first name.
    groups, sizes = {}, set()
    for key, entry in fresh.items():
        if entry["hash"]:
            h, w, hh = entry["hash"]
            groups.setdefault(h, set()).add(Path(key).stem)
            sizes.add(f"{w}x{hh}")
    index = cache_dir / "index.json"
    index.write_text(json.dumps({"hashes": {h: sorted(n) for h, n in groups.items()}, "sizes": sorted(sizes)}))
    return index


def find_renderdoc(arg: Path | None) -> Path:
    cands = [arg] if arg else sorted(Path.home().glob("Applications/renderdoc_*"), key=lambda p: [
        int(x) for x in re.findall(r"\d+", p.name)], reverse=True)
    if not arg and shutil.which("qrenderdoc"):
        cands.append(Path(shutil.which("qrenderdoc")).resolve().parent.parent)
    for c in cands:
        if c and (c / "bin" / "qrenderdoc").exists():
            return c
    sys.exit("RenderDoc not found. Download the Linux tarball from https://renderdoc.org/builds, extract it to "
             "~/Applications/ or pass --renderdoc DIR (see README).")


def find_pcsx2(arg: Path | None) -> Path:
    """The real pcsx2-qt binary. An AppImage is extracted once: RenderDoc can't inject into its static launcher."""
    if arg is None:
        found = shutil.which("pcsx2-qt") or next(iter(sorted(Path.home().glob("Applications/*[Pp][Cc][Ss][Xx]2*.AppImage"))), None)
        if not found:
            sys.exit("PCSX2 not found; pass --pcsx2 PATH (the AppImage or the pcsx2-qt binary).")
        arg = Path(found)
    arg = arg.expanduser().resolve()
    if arg.suffix.lower() != ".appimage":
        return arg
    dest = CACHE / "pcsx2" / f"{arg.stem}-{int(arg.stat().st_mtime)}"
    exe = dest / "squashfs-root" / "usr" / "bin" / "pcsx2-qt"
    if not exe.exists():
        print(f"Extracting {arg.name} for RenderDoc...")
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(arg), "--appimage-extract"], cwd=dest, stdout=subprocess.DEVNULL, check=True)
    return exe


def check_vulkan_layer(renderdoc: Path):
    out = subprocess.run([str(renderdoc / "bin" / "renderdoccmd"), "vulkanlayer", "--explain"],
                         capture_output=True, text=True).stdout
    if "not correctly registered" in out:
        sys.exit("RenderDoc's Vulkan layer isn't registered. Run once:\n"
                 f"  {renderdoc / 'bin' / 'renderdoccmd'} vulkanlayer --register --user")


def run_session(game: Path, index: Path, renderdoc: Path, pcsx2: Path, interval: float):
    work = CACHE / "usage" / game.name
    captures = work / "captures"
    shutil.rmtree(captures, ignore_errors=True)  # leftovers of an interrupted session
    captures.mkdir(parents=True)
    log = work / "session.log"
    log.write_text("")
    cfg = work / "agent.json"
    cfg.write_text(json.dumps({"index": str(index), "output": str(game / USAGE_FILE), "log": str(log),
                               "pcsx2": str(pcsx2), "interval": interval,
                               "capture_template": str(captures / "frame")}))
    env = dict(os.environ, USAGE_AGENT_CONFIG=str(cfg), QT_QPA_PLATFORM="xcb")
    agent = subprocess.Popen([str(renderdoc / "bin" / "qrenderdoc"), "--python", str(ROOT / "usage_agent.py")],
                             env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Starting PCSX2 under RenderDoc. Open the game and play; one frame is captured every {interval:g}s.\n"
          "Close PCSX2 to finish (Ctrl+C here stops sampling but leaves PCSX2 open).\n")
    pos = 0
    try:
        while True:
            done = agent.poll() is not None
            with open(log) as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            if chunk:
                print(chunk, end="", flush=True)
            if done:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        agent.terminate()
        agent.wait()
        print("\nStopped sampling.")
    shutil.rmtree(captures, ignore_errors=True)


def report(game: Path, title: str, top: int):
    try:
        res = json.loads((game / USAGE_FILE).read_text())
    except (OSError, ValueError):
        print("No usage recorded yet.")
        return
    n = res["captures"]
    if not n:
        print("No frames captured yet.")
        return
    repl = game / "replacements"
    replaced = {p.stem for p in repl.rglob("*") if p.is_file()} if repl.is_dir() else set()
    skip = load_skip_list(game)
    rows = sorted(res["textures"].items(), key=lambda kv: (-kv[1]["frames"], -kv[1]["draws"]))
    try:  # how many other names share each texture's content
        index = json.loads((CACHE / "usage" / game.name / "index.json").read_text())
        same = {names[0]: len(names) - 1 for names in index["hashes"].values()}
    except (OSError, ValueError):
        same = {}
    print(f"\n{title}: {n} frames sampled, {len(rows)} distinct textures seen.\n")
    print(f"{'in frames':>9}  {'draws/frame':>11}  {'status':<8}  texture")
    for name, t in rows[:top]:
        status = "skipped" if name in skip else "replaced" if name in replaced else "original"
        extra = f"  (+{same[name]} identical)" if same.get(name) else ""
        print(f"{t['frames'] / n:>9.0%}  {t['draws'] / t['frames']:>11.1f}  {status:<8}  {name}{extra}")
    if len(rows) > top:
        print(f"... {len(rows) - top} more in {game / USAGE_FILE}")


def main():
    ap = argparse.ArgumentParser(description="Measure how often a PCSX2 game uses each texture (RenderDoc sampling).")
    ap.add_argument("textures_dir", nargs="?", type=Path, default=DEFAULT_TEXTURES)
    ap.add_argument("--pcsx2", type=Path, help="PCSX2 AppImage or pcsx2-qt binary")
    ap.add_argument("--renderdoc", type=Path, help="extracted RenderDoc folder (default: newest ~/Applications/renderdoc_*)")
    ap.add_argument("--interval", type=float, default=20, help="seconds between captured frames (default 20)")
    ap.add_argument("--top", type=int, default=30, help="rows in the report (default 30)")
    ap.add_argument("--report", action="store_true", help="only print the report of earlier sessions")
    ap.add_argument("--reset", action="store_true", help="discard earlier sessions' counts first")
    args = ap.parse_args()
    textures = args.textures_dir.expanduser()

    games = sorted(p for p in textures.iterdir() if p.is_dir()) if textures.is_dir() else []
    if not games:
        sys.exit(f"No game folders found in {textures}")
    titles = load_game_titles(textures)
    print(f"Games in {textures}:")
    game = games[choose("Choose a game", [f"{titles[g.name]}  [{g.name}]" if g.name in titles else g.name
                                          for g in games])]
    title = titles.get(game.name, game.name)

    if not args.report:
        renderdoc = find_renderdoc(args.renderdoc)
        check_vulkan_layer(renderdoc)
        pcsx2 = find_pcsx2(args.pcsx2)
        if args.reset:
            (game / USAGE_FILE).unlink(missing_ok=True)
        index = build_index(game)
        run_session(game, index, renderdoc, pcsx2, args.interval)
    report(game, title, args.top)


if __name__ == "__main__":
    main()
