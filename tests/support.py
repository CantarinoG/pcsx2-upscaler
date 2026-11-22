"""Shared by the tests: the repo on sys.path, and generated textures."""
import io
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def surface(w=64, h=64, channels=3, seed=1, alpha=128) -> Image.Image:
    """A noisy mid-grey surface."""
    a = np.random.default_rng(seed).integers(90, 160, (h, w, channels), np.uint8)
    if channels == 4:
        a[..., 3] = alpha
    return Image.fromarray(a)


def dds_levels(data: bytes) -> list[np.ndarray]:
    """The levels of a .dds written by dds.save_dds that are at least 4x4."""
    h, w = struct.unpack_from("<II", data, 12)
    count = struct.unpack_from("<I", data, 28)[0]
    offset, levels = 148, []
    for _ in range(count):
        size = max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 16
        if min(w, h) >= 4:
            header = bytearray(data[:148])
            struct.pack_into("<5I", header, 12, h, w, size, 0, 1)
            with Image.open(io.BytesIO(bytes(header) + data[offset:offset + size])) as im:
                levels.append(np.asarray(im.convert("RGBA"), np.float32))
        offset += size
        w, h = max(1, w // 2), max(1, h // 2)
    assert offset == len(data), "file length doesn't match its mip chain"
    return levels
