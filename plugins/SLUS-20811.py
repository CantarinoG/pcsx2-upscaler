"""Need for Speed: Underground (NTSC-U)."""
import numpy as np
from PIL import Image

# 512x512 textures with this format are video memory the game uses as scratch space: mostly black with a few
# scrambled strips, and a new hash every time, so a replacement is never loaded again.
SCRATCH = "-8000264a"

# Car body skins: the game builds one 512x512 texture per car, paint colour, vinyl and decals combination, so new
# ones keep appearing (every garage preview and rival car) and each upscaled one costs ~5.6 MB. They share this
# format with the city map, track outlines, magazine covers, models and garage backgrounds, so they're told apart
# by content: a skin is mostly one flat, opaque paint colour. On 478 hand-labelled dumps the skins' most common
# colour covered at least 44% of the texture; the other textures' most common colour was either transparent or
# covered at most 10% (the maps).
SKIN_FORMAT = "-00002653"
SKIN_MIN_PAINT = 0.30   # share of the texture covered by its most common colour
SKIN_MIN_ALPHA = 64     # that colour's alpha (PS2 range, 128 = opaque)


def is_car_skin(path) -> bool:
    a = np.asarray(Image.open(path).convert("RGBA"))
    q = (a // 4).astype(np.uint32)  # ignore tiny shading differences
    key = (q[..., 0] << 18) | (q[..., 1] << 12) | (q[..., 2] << 6) | q[..., 3]
    counts = np.bincount(key.ravel())
    top = int(counts.argmax())
    return counts[top] >= SKIN_MIN_PAINT * key.size and (top & 63) * 4 >= SKIN_MIN_ALPHA


def skip(path):
    name = path.stem
    if name.endswith(SCRATCH):
        return "scratch memory"
    if name.endswith(SKIN_FORMAT) and is_car_skin(path):
        return "car skins"
    return None
