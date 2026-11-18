#!/usr/bin/env python3
"""Upscale a PCSX2 game's dumped textures 4x with ONNX models and write them as BC7 .dds replacements.

Pick a game from PCSX2's textures folder; every dump in <textures>/<GAME>/dumps/ that has no replacement
yet is upscaled and saved to <textures>/<GAME>/replacements/ under the same name, as .dds.
A dump counts as replaced if any file with the same name (any extension) exists anywhere under
replacements/, including texture-pack subfolders, so downloaded packs are never overwritten.

Textures with readable text (signs, logos, UI) go through TEXT_MODEL; everything else through OTHER_MODEL.

Usage: pcsx2_upscale.py [textures_dir]      (default: ~/.config/PCSX2/textures)
"""
import argparse
import gc
import multiprocessing
import os
import re
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
DDS_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))  # BC7 encoding is CPU-bound and single-threaded

MIN_SIDE = 64          # some models fail on tiny inputs; smaller images are wrap-padded up to this
WRAP_PAD = 16          # textures repeat: pad each edge with the opposite edge so upscales tile without seams
TILE_MIN = 64          # smallest tile tried after running out of GPU memory
OVERLAP = 16           # context shared between neighbouring tiles, hides tile seams
ALPHA_UNIFORM_SPAN = 2 # alpha this close to constant (e.g. PS2's 127..128) is resized plainly, not by the model


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

    def upscale_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Whole image first; if the GPU runs out of memory, progressively smaller tiles."""
        h, w = rgb.shape[:2]
        padded = np.pad(rgb, ((WRAP_PAD, WRAP_PAD), (WRAP_PAD, WRAP_PAD), (0, 0)), mode="wrap")
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
        return up[WRAP_PAD * s:(WRAP_PAD + h) * s, WRAP_PAD * s:(WRAP_PAD + w) * s]

    def upscale_alpha(self, alpha: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        """HxW uint8 -> out_hw uint8, never outside the source's [min, max].

        PS2 alpha runs 0..128 (128 = opaque; higher blends brighter), so it must not drift. The model adds
        noise and overshoot on alpha, so it's only used for real cut-outs, on alpha stretched to 0..1."""
        amin, amax = int(alpha.min()), int(alpha.max())
        oh, ow = out_hw
        if amax - amin <= ALPHA_UNIFORM_SPAN:
            a = np.asarray(Image.fromarray(alpha.astype(np.float32), "F").resize((ow, oh), Image.Resampling.BICUBIC))
        else:
            norm = (alpha.astype(np.float32) - amin) / (amax - amin)
            a = amin + self.upscale_rgb(np.repeat(norm[..., None], 3, axis=2)).mean(axis=2) * (amax - amin)
            if a.shape != (oh, ow):
                a = np.asarray(Image.fromarray(a, "F").resize((ow, oh), Image.Resampling.BICUBIC))
        return np.clip(np.rint(a), amin, amax).astype(np.uint8)

    def upscale_image(self, img: Image.Image) -> Image.Image:
        has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
        arr = np.asarray(img.convert("RGBA" if has_alpha else "RGB"))
        rgb = (self.upscale_rgb(arr[..., :3].astype(np.float32) / 255.0) * 255.0 + 0.5).astype(np.uint8)
        if not has_alpha:
            return Image.fromarray(rgb, "RGB")
        return Image.fromarray(np.dstack([rgb, self.upscale_alpha(arr[..., 3], rgb.shape[:2])]), "RGBA")


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

    dumps = sorted(p for p in dumps_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    replaced = {p.stem for p in repl_dir.rglob("*") if p.is_file()} if repl_dir.is_dir() else set()
    todo = [p for p in dumps if p.stem not in replaced]
    print(f"\n{titles.get(game.name, game.name)}: {len(dumps)} dumps, {len(dumps) - len(todo)} already replaced, "
          f"{len(todo)} to upscale.")
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
                try:
                    with Image.open(src) as im:
                        im.load()
                        up = upscaler.upscale_image(im)
                except Exception as e:
                    failed += 1
                    print(f"  [{n}/{len(todo)}] {src.name}  FAILED: {type(e).__name__}: {e}")
                    continue
                pending[pool.submit(dds.save_dds, up, repl_dir / f"{src.stem}.dds")] = src
                drain(block=False)
                print(f"  [{n}/{len(todo)}] {src.name}  {im.width}x{im.height} -> {up.width}x{up.height}  "
                      f"{(time.perf_counter() - t0) * 1000:.0f} ms")
            del upscaler
            gc.collect()  # free its GPU memory before the next model loads
        if pending:
            print(f"\nFinishing .dds conversion...")
            drain(block=True)
    except KeyboardInterrupt:
        print("\nInterrupted. Finished textures are kept; run again to continue.")
        pool.shutdown(wait=True, cancel_futures=True)
        sys.exit(1)
    pool.shutdown()
    print(f"\nDone in {time.perf_counter() - t_all:.0f}s: {converted} written, {failed} failed.")
    print(f"Replacements: {repl_dir}")


if __name__ == "__main__":
    main()
