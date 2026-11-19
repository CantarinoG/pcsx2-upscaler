#!/usr/bin/env python3
"""Upscale a PCSX2 game's dumped textures 4x with ONNX models and write them as BC7 .dds replacements.

Pick a game from PCSX2's textures folder; every dump in <textures>/<GAME>/dumps/ that has no replacement
yet is upscaled and saved to <textures>/<GAME>/replacements/ under the same name, as .dds.
A dump counts as replaced if any file with the same name (any extension) exists anywhere under
replacements/, including texture-pack subfolders, so downloaded packs are never overwritten.

Textures with readable text (signs, logos, UI) go through TEXT_MODEL; everything else through OTHER_MODEL.

Textures listed in <textures>/<GAME>/skip.txt are never upscaled, and any replacement for them under
replacements/ (including texture-pack subfolders) is deleted, so the game shows the original.

If texture_usage.py has measured the game (<textures>/<GAME>/usage.json), textures seen strongly magnified get a
second 4x pass (16x the dump in total), kept only if it visibly adds detail; 16x replacements of textures that no
longer qualify go back to 4x. See BOOST_* below.

Usage: pcsx2_upscale.py [textures_dir]      (default: ~/.config/PCSX2/textures)
"""
import argparse
import gc
import json
import multiprocessing
import os
import re
import struct
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

import dds
from textdet import TextDetector

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
TEXT_MODEL = MODELS_DIR / "text.onnx"   # keeps lettering legible
OTHER_MODEL = MODELS_DIR / "other.onnx"  # materials, images, foliage, ...
DEFAULT_TEXTURES = Path.home() / ".config" / "PCSX2" / "textures"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".dds")
SKIP_FILE = "skip.txt"  # per game: textures to leave unreplaced, one name per line
USAGE_FILE = "usage.json"  # per game: how often each texture is drawn, written by texture_usage.py
USAGE_CACHE = Path.home() / ".cache" / "pcsx2-upscaler" / "usage"
# Second 4x pass (16x in total). Thresholds from a blind A/B test of 50 textures at their closest in-game view:
# below 1.5x magnification nobody could tell 4x from 16x, between 1.5x and 3.3x it was a coin toss (and 16x text
# sometimes looked worse), from 3.3x on 16x won 18 of 21 times and never lost.
BOOST_MIN_MAG = 3.3     # closest sampled view: native screen pixels per original texel
BOOST_MIN_GAIN = 3.0    # mean luminance change (0-255) 16x makes over 4x at that view; less = nothing to see
BOOST_MAX_SIDE = 512    # only textures whose 4x replacement is at most this big
BOOST_REJECTED = "boost_rejected.json"  # per game, in USAGE_CACHE: textures whose 16x added too little
BOOST_RETRY_MAG = 1.25  # a rejected texture is tried again once seen this much more magnified than when rejected
DDS_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))  # BC7 encoding is CPU-bound and single-threaded

MIN_SIDE = 64          # some models fail on tiny inputs; smaller images are wrap-padded up to this
EDGE_PAD = 16          # context added around each texture before upscaling, cropped off afterwards
TILE_MIN = 64          # smallest tile tried after running out of GPU memory
OVERLAP = 16           # context shared between neighbouring tiles, hides tile seams
ALPHA_UNIFORM_SPAN = 2 # alpha this close to constant (e.g. PS2's 127..128) is resized plainly, not by the model


def edge_modes(arr: np.ndarray) -> tuple[str, str]:
    """np.pad mode for the (vertical, horizontal) edges of an HxWxC texture.

    A texture that tiles is padded with its opposite edge, so repeated surfaces upscale without seams.
    One that doesn't (gradients, shadows, UI bars) is mirrored instead: wrapping would bleed the far edge in,
    e.g. a shadow that fades to transparent at the bottom would get an opaque last row.
    It tiles along an axis when its two opposite edges differ about as much as neighbouring rows/columns do."""
    a = arr.astype(np.float32)
    modes = []
    for axis in (0, 1):
        n = a.shape[axis]
        if n < 3:
            modes.append("wrap")
            continue
        first, last = np.take(a, 0, axis), np.take(a, n - 1, axis)
        step = np.abs(np.diff(a, axis=axis)).mean(axis=tuple(i for i in range(a.ndim) if i != axis))
        seam = np.abs(first - last).mean()
        modes.append("wrap" if seam <= 2 * float(np.median(step)) + 2 else "symmetric")
    return modes[0], modes[1]


def init_onnxruntime():
    """Called before the first GPU session, not at import: the .dds worker processes import this module too."""
    try:
        ort.preload_dlls()  # CUDA/cuDNN from the pip packages
    except Exception:
        pass
    ort.set_default_logger_severity(4)  # out-of-memory errors are handled in Upscaler; don't print them


class Upscaler:
    """Runs an image-to-image ONNX model with wrap-around padding and an out-of-memory tile fallback."""

    def __init__(self, model_path: Path):
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(model_path), providers=providers)
        self.provider = self.sess.get_providers()[0]
        self.oom_side = 1 << 30  # smallest tile side that has run out of GPU memory

    def _run(self, rgb: np.ndarray) -> np.ndarray:
        """HxWx3 float32 [0,1] -> upscaled HxWx3 float32 [0,1]."""
        h, w = rgb.shape[:2]
        ph, pw = max(0, MIN_SIDE - h), max(0, MIN_SIDE - w)
        if ph or pw:
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="wrap")
        inp = self.sess.get_inputs()[0]
        dtype = np.float16 if "float16" in inp.type else np.float32
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(dtype)
        y = self.sess.run(None, {inp.name: x})[0][0].astype(np.float32).transpose(1, 2, 0)
        s = y.shape[0] // rgb.shape[0]
        return np.clip(y[: h * s, : w * s], 0.0, 1.0)

    def _run_tiled(self, rgb: np.ndarray, tile: int) -> np.ndarray:
        h, w = rgb.shape[:2]
        if h <= tile and w <= tile:
            return self._run(rgb)
        step, out = tile - 2 * OVERLAP, None
        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                # tile = core [y0, y0+step) plus OVERLAP context on each side, clamped to the image
                ty0, tx0 = max(0, y0 - OVERLAP), max(0, x0 - OVERLAP)
                ty1, tx1 = min(h, y0 + step + OVERLAP), min(w, x0 + step + OVERLAP)
                up = self._run(rgb[ty0:ty1, tx0:tx1])
                s = up.shape[0] // (ty1 - ty0)
                if out is None:
                    out = np.empty((h * s, w * s, 3), np.float32)
                cy1, cx1 = min(h, y0 + step), min(w, x0 + step)
                out[y0 * s:cy1 * s, x0 * s:cx1 * s] = up[(y0 - ty0) * s:(cy1 - ty0) * s, (x0 - tx0) * s:(cx1 - tx0) * s]
        return out

    def upscale_rgb(self, rgb: np.ndarray, modes: tuple[str, str]) -> np.ndarray:
        """Whole image first; if the GPU runs out of memory, progressively smaller tiles.
        modes: np.pad mode for the vertical and horizontal edges (see edge_modes)."""
        h, w = rgb.shape[:2]
        padded = np.pad(rgb, ((EDGE_PAD, EDGE_PAD), (0, 0), (0, 0)), mode=modes[0])
        padded = np.pad(padded, ((0, 0), (EDGE_PAD, EDGE_PAD), (0, 0)), mode=modes[1])
        tile = max(padded.shape[:2])
        if tile >= self.oom_side:  # this size already ran out of memory: go straight to tiles
            tile = self.oom_side // 2
        while True:
            try:
                up = self._run_tiled(padded, tile)
                break
            except Exception as e:
                msg = str(e).lower()
                if not ("allocate" in msg or "out of memory" in msg) or tile <= TILE_MIN:
                    raise
                self.oom_side = min(self.oom_side, tile)
                tile = max(TILE_MIN, tile // 2)
        s = up.shape[0] // padded.shape[0]
        return up[EDGE_PAD * s:(EDGE_PAD + h) * s, EDGE_PAD * s:(EDGE_PAD + w) * s]

    def upscale_alpha(self, alpha: np.ndarray, out_hw: tuple[int, int], modes: tuple[str, str]) -> np.ndarray:
        """HxW uint8 -> out_hw uint8, never outside the source's [min, max].

        PS2 alpha runs 0..128 (128 = opaque; higher blends brighter), so it must not drift. The model adds
        noise and overshoot on alpha, so it's only used for real cut-outs, on alpha stretched to 0..1."""
        amin, amax = int(alpha.min()), int(alpha.max())
        oh, ow = out_hw
        if amax - amin <= ALPHA_UNIFORM_SPAN:
            a = np.asarray(Image.fromarray(alpha.astype(np.float32), "F").resize((ow, oh), Image.Resampling.BICUBIC))
        else:
            norm = (alpha.astype(np.float32) - amin) / (amax - amin)
            a = amin + self.upscale_rgb(np.repeat(norm[..., None], 3, axis=2), modes).mean(axis=2) * (amax - amin)
            if a.shape != (oh, ow):
                a = np.asarray(Image.fromarray(a, "F").resize((ow, oh), Image.Resampling.BICUBIC))
        return np.clip(np.rint(a), amin, amax).astype(np.uint8)

    def upscale_image(self, img: Image.Image) -> Image.Image:
        has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
        arr = np.asarray(img.convert("RGBA" if has_alpha else "RGB"))
        modes = edge_modes(arr)  # decided on colour and alpha together: a shadow is all black, only alpha fades
        rgb = (self.upscale_rgb(arr[..., :3].astype(np.float32) / 255.0, modes) * 255.0 + 0.5).astype(np.uint8)
        if not has_alpha:
            return Image.fromarray(rgb, "RGB")
        return Image.fromarray(np.dstack([rgb, self.upscale_alpha(arr[..., 3], rgb.shape[:2], modes)]), "RGBA")


def load_game_titles(textures: Path) -> dict[str, str]:
    """Serial -> title, from PCSX2's game list cache, falling back to its GameIndex.yaml."""
    titles: dict[str, str] = {}
    cfg = textures.parent
    try:  # gamelist.cache entries are length-prefixed (u32 LE) strings: path, serial, title, ...
        data = (cfg / "cache" / "gamelist.cache").read_bytes()
        for m in re.finditer(rb"[A-Z]{4}-\d{5}", data):
            if m.start() < 4 or int.from_bytes(data[m.start() - 4:m.start()], "little") != len(m[0]):
                continue
            n = int.from_bytes(data[m.end():m.end() + 4], "little")
            if 0 < n < 512:
                titles.setdefault(m[0].decode(), data[m.end() + 4:m.end() + 4 + n].decode("utf-8", "replace"))
    except OSError:
        pass
    for idx in [cfg / "resources" / "GameIndex.yaml", Path("/usr/share/PCSX2/resources/GameIndex.yaml"),
                *Path("/tmp").glob(".mount_PCSX2*/usr/bin/resources/GameIndex.yaml")]:
        try:
            serial = None
            with open(idx, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if m := re.match(r"^([A-Z]{4}-\d{5}):", line):
                        serial = m[1]
                    elif serial and (m := re.match(r'^\s+name:\s*"?(.*?)"?\s*$', line)):
                        titles.setdefault(serial, m[1])
                        serial = None
            break
        except OSError:
            continue
    return titles


def load_skip_list(game: Path) -> set[str]:
    """Texture names (hash, file name or path; any extension) listed in <game>/skip.txt. '#' starts a comment."""
    path = game / SKIP_FILE
    if not path.is_file():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name = Path(line).name
            names.add(Path(name).stem if Path(name).suffix.lower() in IMAGE_EXTS else name)
    return names


def remove_skipped(repl_dir: Path, skip: set[str]) -> int:
    """Deletes replacements of skipped textures, wherever they are under replacements/."""
    if not skip or not repl_dir.is_dir():
        return 0
    removed = 0
    for p in sorted(repl_dir.rglob("*")):
        if p.is_file() and p.stem in skip:
            p.unlink()
            removed += 1
            print(f"  removed {p.relative_to(repl_dir)}")
    return removed


def dds_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        h, w = struct.unpack_from("<II", f.read(20), 12)
    return w, h


def upscale_multiplier(textures: Path) -> float:
    """PCSX2's internal resolution (upscale_multiplier in PCSX2.ini); 3 if it can't be read."""
    try:
        m = re.search(r"^upscale_multiplier\s*=\s*([\d.]+)", (textures.parent / "inis" / "PCSX2.ini").read_text(), re.M)
        return float(m[1]) if m else 3.0
    except OSError:
        return 3.0


def second_pass_gain(x4: Image.Image, x16: Image.Image, scale: float) -> float:
    """How much the 16x changes what's on screen compared to the 4x: mean absolute luminance difference (0-255),
    weighted by alpha, with both resampled to `scale` screen pixels per original texel (capped at 16) the way the
    GPU would show them (bilinear when magnified, roughly mipmapped when minified)."""
    scale = min(scale, 16.0)
    w, h = x4.size[0] // 4, x4.size[1] // 4
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    views = []
    for img, factor in ((x4, 4), (x16, 16)):
        filt = Image.Resampling.BILINEAR if scale >= factor else Image.Resampling.LANCZOS
        a = np.asarray(img.convert("RGBA").resize(size, filt)).astype(np.float32)
        alpha = np.clip(a[..., 3:] * 2, 0, 255) / 255  # PS2 alpha: 128 = opaque; shown over mid grey
        rgb = a[..., :3] * alpha + 96 * (1 - alpha)
        views.append((rgb @ np.array([0.299, 0.587, 0.114], np.float32), alpha[..., 0]))
    (l4, alpha), (l16, _) = views
    weight = alpha if alpha.sum() > 0 else np.ones_like(alpha)
    return float((np.abs(l16 - l4) * weight).sum() / weight.sum())


def plan_second_pass(game: Path, dumps: list[Path], repl_dir: Path, skip: set[str], new: set[Path],
                     rejected: dict) -> tuple[dict[Path, float], list[Path]]:
    """Which textures get 16x, and which of this script's own replacements (replacements/<name>.dds, never pack
    files) go back from 16x to 4x.

    Returns ({dump: magnification} to try at 16x, [dumps whose 16x goes back to 4x]). Textures in `new` (no
    replacement yet) that qualify go straight to 16x. Needs usage.json from texture_usage.py; without it nothing
    changes. A texture with no measurement keeps what it has."""
    try:
        usage = json.loads((game / USAGE_FILE).read_text())
    except (OSError, ValueError):
        return {}, []
    try:  # names with identical content are measured under the first one
        index = json.loads((USAGE_CACHE / game.name / "index.json").read_text())
        aliases = {names[0]: names for names in index["hashes"].values()}
    except (OSError, ValueError, KeyError):
        aliases = {}
    mag = {}
    for name, t in usage.get("textures", {}).items():
        if "max_mag" in t:
            for alias in aliases.get(name, [name]):
                mag[alias] = max(mag.get(alias, 0.0), t["max_mag"])
    boost, downgrade = {}, []
    for src in dumps:
        m, repl = mag.get(src.stem), repl_dir / f"{src.stem}.dds"
        if m is None or src.stem in skip or not (src in new or repl.is_file()):
            continue
        try:
            with Image.open(src) as im:
                sw, sh = im.size
            w, h = (sw * 4, sh * 4) if src in new else dds_size(repl)
        except Exception:
            continue
        if (w, h) == (sw * 4, sh * 4):
            if m >= BOOST_MIN_MAG and max(w, h) <= BOOST_MAX_SIDE and retry_rejected(rejected, src.stem, m):
                boost[src] = m
        elif (w, h) == (sw * 16, sh * 16) and m < BOOST_MIN_MAG:
            downgrade.append(src)
    return boost, downgrade


def retry_rejected(rejected: dict, name: str, mag: float) -> bool:
    """True if `name` was never rejected, or has since been seen at least BOOST_RETRY_MAG times closer (a closer
    view can make the 16x worth it). Entries without a magnification (older format) are retried once."""
    entry = rejected.get(name)
    if entry is None:
        return True
    return not isinstance(entry, dict) or mag >= entry.get("mag", 0.0) * BOOST_RETRY_MAG


def save_rejected(path: Path, rejected: dict):
    if rejected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rejected, indent=1, sort_keys=True))


def choose(prompt: str, options: list[str]) -> int:
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        try:
            ans = input(f"{prompt} [1-{len(options)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nAborted.")
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return int(ans) - 1
        print("Invalid choice.")


def main():
    ap = argparse.ArgumentParser(description="Upscale a PCSX2 game's dumped textures into BC7 .dds replacements.")
    ap.add_argument("textures_dir", nargs="?", type=Path, default=DEFAULT_TEXTURES)
    textures = ap.parse_args().textures_dir.expanduser()

    games = sorted(p for p in textures.iterdir() if p.is_dir()) if textures.is_dir() else []
    if not games:
        sys.exit(f"No game folders found in {textures}")
    titles = load_game_titles(textures)

    def describe(g: Path) -> str:
        n = sum(1 for _ in (g / "dumps").glob("*")) if (g / "dumps").is_dir() else 0
        return f"{titles[g.name]}  [{g.name}]  ({n} dumps)" if g.name in titles else f"{g.name}  ({n} dumps)"

    print(f"Games in {textures}:")
    game = games[choose("Choose a game", [describe(g) for g in games])]
    dumps_dir, repl_dir = game / "dumps", game / "replacements"
    if not dumps_dir.is_dir():
        sys.exit(f"{game.name} has no dumps folder.")

    skip = load_skip_list(game)
    if skip:
        print(f"\n{SKIP_FILE}: {len(skip)} textures to leave unreplaced.")
        if n := remove_skipped(repl_dir, skip):
            print(f"Removed {n} replacements of skipped textures.")

    dumps = sorted(p for p in dumps_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    replaced = {p.stem for p in repl_dir.rglob("*") if p.is_file()} if repl_dir.is_dir() else set()
    skipped = [p for p in dumps if p.stem in skip]
    new = [p for p in dumps if p.stem not in replaced and p.stem not in skip]
    rejected_file = USAGE_CACHE / game.name / BOOST_REJECTED
    try:
        rejected = json.loads(rejected_file.read_text())
    except (OSError, ValueError):
        rejected = {}
    new_set = set(new)
    boost, downgrade = plan_second_pass(game, dumps, repl_dir, skip, new_set, rejected)
    print(f"\n{titles.get(game.name, game.name)}: {len(dumps)} dumps, {len(dumps) - len(new) - len(skipped)} already "
          f"replaced, {len(skipped)} skipped, {len(new)} to upscale.")
    if boost:
        print(f"{USAGE_FILE}: {len(boost)} close-up textures get a second 4x pass (16x), kept if it adds detail.")
    if downgrade:
        print(f"{USAGE_FILE}: {len(downgrade)} textures at 16x no longer seen close enough go back to 4x.")
    todo = new + [p for p in boost if p not in new_set] + downgrade
    downgraded = set(downgrade)
    internal = upscale_multiplier(textures)
    if not todo:
        return

    for m in (TEXT_MODEL, OTHER_MODEL):
        if not m.exists():
            sys.exit(f"Model not found: {m}  (see README)")
    init_onnxruntime()
    cuda = "CUDAExecutionProvider" in ort.get_available_providers()

    # Pass 1: sort the textures by whether they contain text.
    t_all = time.perf_counter()
    detector = TextDetector(use_cuda=cuda)
    text, other = [], []
    try:
        for i, src in enumerate(todo, 1):
            try:
                with Image.open(src) as im:
                    (text if detector.has_text(im) else other).append(src)
            except Exception:
                other.append(src)  # unreadable: fails (and is reported) in pass 2
            print(f"\rLooking for text... {i}/{len(todo)}", end="", flush=True)
    except KeyboardInterrupt:
        sys.exit("\nAborted.")
    del detector
    print(f"\rLooking for text: {len(text)} with text, {len(other)} without "
          f"({time.perf_counter() - t_all:.0f}s).\n")
    repl_dir.mkdir(exist_ok=True)

    # BC7 encoding runs in worker processes so it overlaps with GPU upscaling ("spawn": no inherited CUDA state).
    pool = ProcessPoolExecutor(DDS_WORKERS, mp_context=multiprocessing.get_context("spawn"))
    pending: dict[Future, Path] = {}
    failed = converted = 0

    def drain(block: bool):
        nonlocal failed, converted
        for fut in [f for f in pending if block or f.done()]:
            src = pending.pop(fut)
            if fut.exception() is None:
                converted += 1
            else:
                failed += 1
                print(f"  {src.name}  .dds FAILED: {fut.exception()!r}")

    n = 0
    try:
        for model, files in ((OTHER_MODEL, other), (TEXT_MODEL, text)):
            if not files:
                continue
            upscaler = Upscaler(model)
            print(f"{model.name}  [{upscaler.provider}]: {len(files)} textures")
            for src in files:
                n += 1
                t0 = time.perf_counter()
                note = "  (back to 4x)" if src in downgraded else ""
                try:
                    with Image.open(src) as im:
                        im.load()
                        up = upscaler.upscale_image(im)
                        if src in boost:
                            up16 = upscaler.upscale_image(up)
                            gain = second_pass_gain(up, up16, boost[src] * internal)
                            if gain >= BOOST_MIN_GAIN:
                                up, note = up16, f"  (16x, gain {gain:.1f})"
                            else:
                                rejected[src.stem] = {"gain": round(gain, 2), "mag": boost[src]}
                                note = f"  (16x adds too little: gain {gain:.1f}, stays 4x)"
                                if src not in new_set:  # its 4x replacement is already there
                                    print(f"  [{n}/{len(todo)}] {src.name}{note}  "
                                          f"{(time.perf_counter() - t0) * 1000:.0f} ms")
                                    continue
                except Exception as e:
                    failed += 1
                    print(f"  [{n}/{len(todo)}] {src.name}  FAILED: {type(e).__name__}: {e}")
                    continue
                pending[pool.submit(dds.save_dds, up, repl_dir / f"{src.stem}.dds")] = src
                drain(block=False)
                print(f"  [{n}/{len(todo)}] {src.name}  {im.width}x{im.height} -> {up.width}x{up.height}  "
                      f"{(time.perf_counter() - t0) * 1000:.0f} ms" + note)
            del upscaler
            gc.collect()  # free its GPU memory before the next model loads
        if pending:
            print(f"\nFinishing .dds conversion...")
            drain(block=True)
    except KeyboardInterrupt:
        print("\nInterrupted. Finished textures are kept; run again to continue.")
        pool.shutdown(wait=True, cancel_futures=True)
        save_rejected(rejected_file, rejected)
        sys.exit(1)
    pool.shutdown()
    save_rejected(rejected_file, rejected)
    print(f"\nDone in {time.perf_counter() - t_all:.0f}s: {converted} written, {failed} failed.")
    print(f"Replacements: {repl_dir}")


if __name__ == "__main__":
    main()
