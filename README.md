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
- **Second pass for frequent textures:** if the game has been measured with `texture_usage.py` (see below), textures drawn in at least 50% of the sampled frames whose replacement is still 512 px or smaller are upscaled 4x again, to 16x the dump. Only this script's own replacements are redone, never pack files, and a texture already at 16x is left alone on later runs. The thresholds are `BOOST_MIN_FRAMES` and `BOOST_MAX_SIDE` at the top of `pcsx2_upscale.py`.
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

## Texture usage (optional)

`texture_usage.py` measures how often the game draws each texture, so you can see which replacements matter most (or which texture is behind a glitch). It samples frames with [RenderDoc](https://renderdoc.org) while you play: every 20 seconds one frame is captured (a freeze of about half a second), the textures drawn in it are counted, and the capture is deleted.

Setup, once:
1. Download the Linux tarball from [renderdoc.org/builds](https://renderdoc.org/builds) and extract it to `~/Applications/` (or pass `--renderdoc DIR`).
2. Register its Vulkan layer for your user: `~/Applications/renderdoc_<version>/bin/renderdoccmd vulkanlayer --register --user`. It only activates in programs launched through RenderDoc.
3. In PCSX2, set **Settings → Graphics → Renderer** to Vulkan (OpenGL also works).

```bash
.venv/bin/python texture_usage.py [textures_dir] [--pcsx2 PATH] [--interval 20] [--top 30]
```

Pick the game, then play it in the PCSX2 window that opens. Close PCSX2 to finish. Each frame shows up in the terminal as it's counted, and the report at the end lists the most-used textures:

```
in frames  draws/frame  status    texture
     100%        110.7  replaced  c4010ca101394499-41db6a081dc88d63-00001dd3
      67%          3.0  skipped   8fb42b62ecb75632-59e16594d3d6e113-00001dd3  (+2 identical)
```

- **in frames:** share of sampled frames that used the texture. **draws/frame:** how many times it was drawn in those frames.
- Textures are recognised by content: a replacement by its `.dds` data, a texture without one by its dump. Variants with identical content (same texture, different palette hash) are counted together.
- The upscaler reads these counts to give frequently drawn textures a second pass (see "How it works").
- Counts add up across sessions in `<textures>/<GAME>/usage.json`. `--reset` starts over, `--report` only prints the report.
- PCSX2 runs under X11 (XWayland) for this, since RenderDoc can't capture Wayland windows. An AppImage is extracted once to `~/.cache/pcsx2-upscaler/pcsx2/`, because RenderDoc can't inject into the AppImage launcher.

## Files

| File | Purpose |
|---|---|
| `pcsx2_upscale.py` | The whole flow: game selection, upscaling, writing replacements |
| `textdet.py` | Text detection (OCR), decides which model a texture goes to |
| `dds.py` | BC7 `.dds` writer with mipmaps, in the layout PCSX2's loader expects |
| `texture_usage.py` | Texture usage sampling: indexes the textures, runs the session, prints the report |
| `usage_agent.py` | Runs inside RenderDoc: launches PCSX2, captures frames, counts textures per frame |
