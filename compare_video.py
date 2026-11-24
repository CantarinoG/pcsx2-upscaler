#!/usr/bin/env python3
"""Turn two PCSX2 video captures of the same replay (textures original / upscaled) into one comparison video.

Record an input recording once (Tools -> Input Recording), replay it twice with Video Capture on, once with
Load Texture Replacements off and once on. This script then:
  - picks the two newest captures in PCSX2's videos folder (or the two files given),
  - tells which one is upscaled (the sharper one),
  - lines them up: roughly by their audio (the captures can start a few frames apart), then frame by frame by
    the picture, since a stutter while capturing drops or repeats a frame in one of them,
  - renders a title card, then the whole replay with a divider sweeping across (original on the left, upscaled
    on the right), with the game's audio.

Usage: compare_video.py [original.mp4 upscaled.mp4] [--out FILE] [--title TEXT]
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

VIDEOS = Path.home() / ".config" / "PCSX2" / "videos"
OUT_W, OUT_H = 1920, 1080
VIEW_W, VIEW_H = 1440, 1080          # the 4:3 game picture, centred
TITLE_SECONDS = 4.0
FADE_SECONDS = 0.5                   # fades in and out of the title card
WIPE_PERIOD = 8.0                    # seconds per full left-right-left sweep
AUDIO_RATE = 8000
THUMB = (64, 48)                     # frame size the captures are lined up at
SEARCH = 15                          # frames either side of the audio's offset that the picture may be off by
SWITCH_COST = 30.0                   # how sure the pictures must be before the offset moves (stutters)
FONTS = ("/usr/share/fonts/opentype/inter/Inter-Bold.otf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONTS_REGULAR = ("/usr/share/fonts/opentype/inter/Inter-Regular.otf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def font(size: int, bold: bool = True):
    for path in FONTS if bold else FONTS_REGULAR:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return {"w": s["width"], "h": s["height"], "fps": num / den, "rate": s["r_frame_rate"],
            "frames": int(s["nb_read_packets"])}


def audio(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(AUDIO_RATE),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def xcorr(x: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Lag k (x[n] ~ y[n - k]) maximising the cross-correlation, and the normalised peak."""
    n = 1 << int(math.ceil(math.log2(len(x) + len(y))))
    c = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(y, n)), n)
    c = np.concatenate([c[-(len(y) - 1):], c[:len(x)]])
    k = int(np.argmax(c))
    return k - (len(y) - 1), float(c[k] / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))


def audio_lag(a: Path, b: Path, fps: float) -> int:
    """Rough lag in frames (b's frame n + lag shows a's frame n), from the audio. Exits if they don't match."""
    sa, sb = audio(a), audio(b)
    lag, _ = xcorr(sa, sb)
    # Judge the match on short windows: a stutter shifts one capture by a frame or two, which spoils the score
    # over the whole length but not within a few seconds.
    scores, win = [], 5 * AUDIO_RATE
    for t in range(0, len(sa) - win, 10 * AUDIO_RATE):
        lo, hi = max(0, t - lag - AUDIO_RATE), min(len(sb), t - lag + win + AUDIO_RATE)
        if hi - lo >= win:
            scores.append(xcorr(sb[lo:hi], sa[t:t + win])[1])
    score = float(np.median(scores)) if scores else xcorr(sa, sb)[1]
    if score < 0.6:
        sys.exit(f"The two captures don't sound alike (match {score:.2f}): are they replays of the same recording?")
    return round(-lag / AUDIO_RATE * fps)  # a[n] ~ b[n - lag]


def thumbs(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vf",
                          f"scale={THUMB[0]}:{THUMB[1]},format=gray", "-fps_mode", "passthrough", "-f", "rawvideo",
                          "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, THUMB[1], THUMB[0]).astype(np.float32)


def align(ta: np.ndarray, tb: np.ndarray, lag: int) -> np.ndarray:
    """For each frame of a, the frame of b showing the same moment (-1 past b's end).

    A stutter while capturing drops or repeats a frame in one capture, so the offset between them can change along
    the way. Every offset within SEARCH frames of `lag` is scored per frame by how different the two pictures are,
    and the cheapest path through them (Viterbi) is taken, paying SWITCH_COST per frame of change, so the offset
    only moves where the pictures clearly say so."""
    ks = np.arange(lag - SEARCH, lag + SEARCH + 1)
    cost = np.full((len(ta), len(ks)), np.inf, np.float32)
    for c, k in enumerate(ks):
        lo, hi = max(0, -k), min(len(ta), len(tb) - k)
        if hi > lo:
            cost[lo:hi, c] = np.abs(ta[lo:hi] - tb[lo + k:hi + k]).mean((1, 2))
    inside = np.isfinite(cost).all(1)
    cost[~inside] = 0.0   # near either end, where some offsets run past a capture: no say, the offset carries on
    n = len(ta)
    step = SWITCH_COST * np.abs(ks[:, None] - ks[None, :])   # [from, to]
    total, back = cost[0].copy(), np.zeros((n, len(ks)), np.int32)
    for i in range(1, n):
        t = total[:, None] + step
        back[i] = t.argmin(0)
        total = t[back[i], np.arange(len(ks))] + cost[i]
    path = np.empty(n, np.int32)
    path[-1] = int(total.argmin())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    if inside.any() and (path[inside].min() == 0 or path[inside].max() == len(ks) - 1):
        sys.exit("The replays drift apart too far to line up: the input recording didn't replay identically.")
    out = np.arange(n) + ks[path]
    out[(out < 0) | (out >= len(tb))] = -1
    return out


def sync(a: Path, b: Path, fps: float) -> tuple[int, np.ndarray]:
    """Frames of a to skip at the start, and for each following frame of a, the matching frame of b."""
    lag = audio_lag(a, b, fps)
    match = align(thumbs(a), thumbs(b), lag)
    valid = match >= 0
    skip = int(np.argmax(valid))                        # a starts before b: drop a's frames b has no match for
    match = match[skip:]
    end = len(match) if valid[skip:].all() else int(np.argmin(match >= 0))
    match = match[:end]
    jumps = [i for i in range(1, len(match)) if match[i] - match[i - 1] != 1]
    where = ", ".join(f"{(skip + i) / fps // 60:.0f}:{(skip + i) / fps % 60:04.1f}" for i in jumps[:8])
    print(f"Synced: offset {match[0] - skip:+d} frames" +
          (f", re-aligned {len(jumps)} time(s) after stutters ({where}{', ...' if len(jumps) > 8 else ''})."
           if jumps else ", steady."))
    return skip, match


def grab(path: Path, t: float, size=(VIEW_W, VIEW_H)) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                          "-vf", f"scale={size[0]}:{size[1]}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(size[1], size[0]).astype(np.float32)


def sharpness(img: np.ndarray) -> float:
    return float(np.abs(np.diff(img, axis=1)).mean() + np.abs(np.diff(img, axis=0)).mean())


class Reader:
    """Decodes a video from a given frame on as RGB numpy frames at its own resolution."""

    def __init__(self, path: Path, skip: int, w: int, h: int):
        self.w, self.h = w, h
        self.pos, self.last = -1, None   # index and picture of the last frame read
        # Every frame, in order (ffmpeg's trim filter drops frames on these captures); the first `skip` are discarded.
        self.proc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "rawvideo",
                                      "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-"], stdout=subprocess.PIPE)
        for _ in range(skip):
            self.read()

    def frame(self, n: int) -> Image.Image | None:
        """Frame n, for n that never goes back: frames in between are skipped, the same n again repeats it."""
        while self.pos < n:
            self.last = self.read()
        return self.last

    def read(self) -> Image.Image | None:
        self.pos += 1
        buf = self.proc.stdout.read(self.w * self.h * 3)
        if len(buf) < self.w * self.h * 3:
            return None
        return Image.frombuffer("RGB", (self.w, self.h), buf)

    def close(self):
        self.proc.stdout.close()
        self.proc.wait()


def label(draw: ImageDraw.ImageDraw, xy, text: str, anchor: str, size: int = 34):
    draw.text(xy, text, font=font(size), fill=(255, 255, 255), anchor=anchor, stroke_width=4,
              stroke_fill=(0, 0, 0))


def frame_wipe(orig: Image.Image, up: Image.Image, t: float) -> Image.Image:
    o, u = orig.resize((VIEW_W, VIEW_H), Image.Resampling.BICUBIC), up.resize((VIEW_W, VIEW_H), Image.Resampling.BICUBIC)
    x = round(VIEW_W * (0.5 + 0.5 * math.cos(2 * math.pi * t / WIPE_PERIOD)))
    view = u.copy()
    if x:
        view.paste(o.crop((0, 0, x, VIEW_H)), (0, 0))
    canvas = Image.new("RGB", (OUT_W, OUT_H))
    ox = (OUT_W - VIEW_W) // 2
    canvas.paste(view, (ox, 0))
    d = ImageDraw.Draw(canvas)
    lx = ox + min(max(x, 2), VIEW_W - 2)
    d.rectangle((lx - 2, 0, lx + 1, OUT_H), fill=(255, 255, 255))
    hx = ox + min(max(x, 22), VIEW_W - 22)
    d.ellipse((hx - 20, OUT_H // 2 - 20, hx + 20, OUT_H // 2 + 20), fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    for s in (-1, 1):
        d.polygon([(hx + 5 * s, OUT_H // 2 - 9), (hx + 14 * s, OUT_H // 2), (hx + 5 * s, OUT_H // 2 + 9)], fill=(0, 0, 0))
    label(d, (ox + 24, 20), "Original", "la")
    label(d, (ox + VIEW_W - 24, 20), "Upscaled", "ra")
    return canvas


def title_card(still: Image.Image, title: str, subtitle: str) -> Image.Image:
    bg = still.resize((OUT_W, OUT_W * still.height // still.width), Image.Resampling.BICUBIC)
    bg = bg.crop((0, (bg.height - OUT_H) // 2, OUT_W, (bg.height - OUT_H) // 2 + OUT_H))
    bg = bg.filter(ImageFilter.GaussianBlur(18))
    bg = Image.blend(bg, Image.new("RGB", bg.size), 0.6)
    d = ImageDraw.Draw(bg)
    d.text((OUT_W // 2, OUT_H // 2 - 50), title, font=font(84), fill=(255, 255, 255), anchor="mm")
    d.text((OUT_W // 2, OUT_H // 2 + 50), subtitle, font=font(40, bold=False), fill=(225, 225, 225), anchor="mm")
    return bg


def game_title(path: Path) -> str:
    m = re.match(r"(.+?)_[A-Z]{4}-\d{5}_\d+", path.stem)
    return (m[1] if m else path.stem).replace(" - ", ": ")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="*", type=Path, help="original.mp4 upscaled.mp4 (default: the two newest "
                    "captures in PCSX2's videos folder, sorted automatically)")
    ap.add_argument("--out", type=Path, help="output file (default: <videos folder>/<game>_comparison.mp4)")
    ap.add_argument("--title", help="title card text (default: the game's name)")
    args = ap.parse_args()

    if len(args.videos) not in (0, 2):
        sys.exit("Give two videos (original, upscaled) or none.")
    auto = not args.videos
    if auto:
        caps = sorted((p for p in VIDEOS.glob("*.mp4") if not p.stem.endswith("_comparison")),
                      key=lambda p: p.stat().st_mtime)
        if len(caps) < 2:
            sys.exit(f"Need two captures in {VIDEOS}.")
        a, b = caps[-2], caps[-1]
    else:
        a, b = args.videos
    pa, pb = probe(a), probe(b)
    if (pa["w"], pa["h"]) != (pb["w"], pb["h"]):
        sys.exit(f"The captures have different sizes ({pa['w']}x{pa['h']} vs {pb['w']}x{pb['h']}).")
    fps = pa["fps"]
    if auto:  # the upscaled replay is the sharper one
        ts = [0.2, 0.4, 0.6, 0.8]
        sa = sum(sharpness(grab(a, f * pa["frames"] / fps)) for f in ts)
        sb = sum(sharpness(grab(b, f * pb["frames"] / fps)) for f in ts)
        if sa > sb:
            a, b, pa, pb = b, a, pb, pa
        print(f"Original: {a.name}\nUpscaled: {b.name}")
    skip, match = sync(a, b, fps)
    frames = len(match)
    seconds = frames / fps
    title = args.title or game_title(b)
    out = args.out or b.parent / f"{re.sub(r'[^A-Za-z0-9]+', '_', title).strip('_')}_comparison.mp4"

    title_frames = round(TITLE_SECONDS * fps)
    fade = round(FADE_SECONDS * fps)
    total = title_frames + frames
    ra, rb = Reader(a, skip, pa["w"], pa["h"]), Reader(b, 0, pb["w"], pb["h"])
    first_o, first_u = ra.read(), rb.frame(int(match[0]))
    card = title_card(first_u, title, "Original vs. upscaled textures")

    enc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}", "-r", pb["rate"], "-i", "-",
         "-ss", f"{skip / fps:.4f}", "-i", str(a),   # the original's sound: its frames are all shown in order
         "-filter_complex",
         f"[1:a]atrim=duration={seconds:.3f},asetpts=PTS-STARTPTS,"
         f"afade=t=in:d=0.5,afade=t=out:st={max(0, seconds - 1.5):.3f}:d=1.5,"
         f"adelay={int(TITLE_SECONDS * 1000)}:all=1[aud]",
         "-map", "0:v", "-map", "[aud]", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)
    black = Image.new("RGB", (OUT_W, OUT_H))
    try:
        orig, up = first_o, first_u
        for n in range(total):
            if n < title_frames:  # title: fade in from black, crossfade into the wipe
                img = card
                if n < fade:
                    img = Image.blend(black, card, n / fade)
                elif n >= title_frames - fade:
                    img = Image.blend(card, frame_wipe(orig, up, 0.0), (n - (title_frames - fade)) / fade)
            else:
                i = n - title_frames
                if i > 0:
                    orig, up = ra.read(), rb.frame(int(match[i]))
                    if orig is None or up is None:
                        break
                img = frame_wipe(orig, up, i / fps)
                if i >= frames - fade * 2:  # fade out to black
                    img = Image.blend(img, black, (i - (frames - fade * 2)) / (fade * 2))
            enc.stdin.write(img.tobytes())
            if n % 60 == 0:
                print(f"\rRendering... {n * 100 // total}%", end="", flush=True)
    finally:
        ra.close()
        rb.close()
        enc.stdin.close()
        enc.wait()
    print(f"\rRendering... done.\n{out}")


if __name__ == "__main__":
    main()
