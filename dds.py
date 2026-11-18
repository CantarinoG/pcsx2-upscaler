"""BC7 .dds writer with a full mipmap chain, in the layout PCSX2's texture-replacement loader expects.

Imports nothing GPU-related, so it can run in worker processes next to the upscaler.
"""
import struct
from pathlib import Path

import etcpak
import numpy as np
from PIL import Image

DXGI_FORMAT_BC7_UNORM = 98
DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PIXELFORMAT = 0x1, 0x2, 0x4, 0x1000
DDSD_MIPMAPCOUNT, DDSD_LINEARSIZE = 0x20000, 0x80000
DDSCAPS_COMPLEX, DDSCAPS_TEXTURE, DDSCAPS_MIPMAP = 0x8, 0x1000, 0x400000
DDPF_FOURCC = 0x4
D3D10_RESOURCE_DIMENSION_TEXTURE2D = 3


def _mip_chain(img: Image.Image) -> list[Image.Image]:
    """Base level plus half-size levels down to 1x1. PCSX2 can't generate mipmaps for compressed
    textures itself (it disables mipmapping for them instead), so they're baked in here."""
    levels = [img]
    w, h = img.size
    while w > 1 or h > 1:
        w, h = max(1, w // 2), max(1, h // 2)
        levels.append(levels[-1].resize((w, h), Image.Resampling.BOX))
    return levels


def _bc7(level: Image.Image) -> bytes:
    """BC7 works on 4x4 blocks; levels smaller than that are padded by repeating edge pixels."""
    arr = np.asarray(level)
    ph, pw = -arr.shape[0] % 4, -arr.shape[1] % 4
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="edge")
    params = etcpak.BC7CompressBlockParams()
    params.init_linear_weights()  # the default perceptual weights favour colour; linear keeps alpha more accurate
    return etcpak.compress_bc7(np.ascontiguousarray(arr).tobytes(), arr.shape[1], arr.shape[0], params)


def _header(width: int, height: int, mip_count: int, base_size: int) -> bytes:
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_MIPMAPCOUNT | DDSD_LINEARSIZE
    caps = DDSCAPS_TEXTURE | (DDSCAPS_COMPLEX | DDSCAPS_MIPMAP if mip_count > 1 else 0)
    pixel_format = struct.pack("<II4s5I", 32, DDPF_FOURCC, b"DX10", 0, 0, 0, 0, 0)
    header = struct.pack("<7I44x", 124, flags, height, width, base_size, 0, mip_count)
    header += pixel_format + struct.pack("<5I", caps, 0, 0, 0, 0)
    dx10 = struct.pack("<5I", DXGI_FORMAT_BC7_UNORM, D3D10_RESOURCE_DIMENSION_TEXTURE2D, 0, 1, 0)
    return b"DDS " + header + dx10


def save_dds(img: Image.Image, dst: str | Path) -> None:
    """Save as BC7 .dds with mipmaps, via a temp file so an interrupted run never leaves a broken file."""
    dst = Path(dst)
    img = img.convert("RGBA")
    w, h = img.size
    if w % 4 or h % 4:  # PCSX2 rejects block-compressed textures whose base size isn't a multiple of 4
        img = img.resize((max(4, round(w / 4) * 4), max(4, round(h / 4) * 4)), Image.Resampling.LANCZOS)
    blocks = [_bc7(level) for level in _mip_chain(img)]
    tmp = dst.with_name(dst.name + ".part")
    try:
        with open(tmp, "wb") as f:
            f.write(_header(img.width, img.height, len(blocks), len(blocks[0])))
            for b in blocks:
                f.write(b)
        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
