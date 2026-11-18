# pcsx2-upscaler

Upscales a PCSX2 game's dumped textures 4x with ONNX models and writes them as BC7 `.dds` replacements that PCSX2 loads directly. Textures with text use one model and everything else uses another.

```
                                     ┌─ has text ─► text model ──┐
<textures>/<GAME>/dumps/<hash>.png ──┤                           ├─► BC7 .dds + mipmaps ─► <textures>/<GAME>/replacements/<hash>.dds
                                     └─ no text ──► other model ─┘
```

![Before and after: a graffiti-covered wall, a logo and a brick facade from Need for Speed: Underground](docs/comparison.jpg)

## How it works

- **Only what's missing:** a dump counts as replaced if a file with the same name (any extension) exists anywhere under `replacements/`, including texture-pack subfolders. Downloaded packs are never overwritten, and an interrupted run continues where it stopped.
- **Skip list:** textures named in `<textures>/<GAME>/skip.txt` are never upscaled, and any existing replacement for them under `replacements/` (pack subfolders included) is deleted on each run, so the game shows the original texture.
- **Text detection:** OCR (RapidOCR with PP-OCRv4, Latin and Chinese) runs on each texture first. A texture counts as text when OCR reads at least 2 characters with confidence ≥ 0.5. On 229 hand-labelled dumps this gave 89% precision and 98% recall; the old "detected text area" approach peaked at about 63% / 81%. The rule favours recall because the non-text model garbles lettering, while the text model handles any texture well. It runs on the GPU at about 75 ms per texture.
- **Edges:** each texture is padded before upscaling. If its opposite edges match (repeating ground and wall textures), it's padded with them, so the upscale tiles without seams. Otherwise (gradients, shadows, UI bars) it's mirrored, so one edge doesn't bleed into the other.
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

### Leaving textures unreplaced

Create `<textures>/<GAME>/skip.txt` with one texture per line. Each line can be the hash, the file name (any extension) or a full path; `#` starts a comment:

```
# blurry HUD font, the original looks better
1a5f3316f752e298-2838fc9f5ff51c7c-000059d3
replacements/pack/1a8dd21b75996f4a-8b1fe45826ac4020-00005194.dds
```

On the next run those textures are skipped, and their existing replacements are deleted, including ones from downloaded packs.

## Files

| File | Purpose |
|---|---|
| `pcsx2_upscale.py` | The whole flow: game selection, upscaling, writing replacements |
| `textdet.py` | Text detection (OCR), decides which model a texture goes to |
| `dds.py` | BC7 `.dds` writer with mipmaps, in the layout PCSX2's loader expects |
