# pcsx2-upscaler

Upscales a PCSX2 game's dumped textures 4x with ONNX models and writes them as BC7 `.dds` replacements that PCSX2 loads directly. Textures with text use one model and everything else uses another.

```
                                     ┌─ has text ─► text model ──┐
<textures>/<GAME>/dumps/<hash>.png ──┤                           ├─► BC7 .dds + mipmaps ─► <textures>/<GAME>/replacements/<hash>.dds
                                     └─ no text ──► other model ─┘
```

## How it works

- **Only what's missing:** a dump counts as replaced if a file with the same name (any extension) exists anywhere under `replacements/`, including texture-pack subfolders. Downloaded packs are never overwritten, and an interrupted run continues where it stopped.
- **Text detection:** OCR (RapidOCR with PP-OCRv4, Latin and Chinese) runs on each texture first. A texture counts as text when OCR reads at least 2 characters with confidence ≥ 0.5. On 229 hand-labelled dumps this gave 89% precision and 98% recall; the old "detected text area" approach peaked at about 63% / 81%. The rule favours recall because the non-text model garbles lettering, while the text model handles any texture well. It runs on the GPU at about 75 ms per texture.
- **Seamless tiling:** each texture is padded with its own opposite edges before upscaling, so repeating ground and wall textures don't show seams.
- **PS2 alpha:** transparency runs 0–128 on the PS2 (128 = opaque). It stays within each dump's own range:
  - near-uniform alpha is resized normally;
  - real cut-outs go through the model, then are limited to the dump's range.
- **Low VRAM:** each texture is upscaled whole first. If the GPU runs out of memory, it retries in progressively smaller overlapping tiles.
- **`.dds` output:** BC7 with a full mipmap chain. PCSX2 can't generate mipmaps for compressed textures itself. Encoding runs in background processes while the GPU upscales.

## Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Put two 4x ONNX upscalers of your choice in `models/` (the file names are set at the top of `pcsx2_upscale.py`):

| File | Used for |
|---|---|
| `other.onnx` | everything without text |
| `text.onnx` | textures with text |

The OCR models are downloaded by `rapidocr` on first run.

In PCSX2, under **Settings → Graphics → Texture Replacement**:
- **Dump Replaceable Textures:** turn on and play, to collect dumps.
- **Load Texture Replacements:** turn on to use the results.

## Usage

```bash
.venv/bin/python pcsx2_upscale.py [textures_dir]    # default: ~/.config/PCSX2/textures
```

Pick a game from the list (names come from PCSX2's game list). The script first sorts every dump without a replacement into text and non-text, then upscales each group with its model.

## Files

| File | Purpose |
|---|---|
| `pcsx2_upscale.py` | The whole flow: game selection, upscaling, writing replacements |
| `textdet.py` | Text detection (OCR), decides which model a texture goes to |
| `dds.py` | BC7 `.dds` writer with mipmaps, in the layout PCSX2's loader expects |
