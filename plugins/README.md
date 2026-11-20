# Game plugins

A plugin holds rules for one game. It's a Python file named after the game's serial, the same name as its folder under PCSX2's `textures/` (e.g. `SLUS-20811.py`). `pcsx2_upscale.py` and `texture_usage.py` load it automatically when that game is chosen; games without one are handled normally.

Plugins run as ordinary Python inside the scripts, so only use ones you trust.

## Functions

A plugin defines only the functions it needs.

### `skip(path) -> str | None`

Called once per dump, with the dump file's `Path`. Return a short reason (e.g. `"car skins"`) to leave that texture unreplaced, or `None` to treat it normally. Skipped textures work exactly like `skip.txt` entries: they're never upscaled, and any existing replacement for them is deleted.

Answers are cached per dump in `~/.cache/pcsx2-upscaler/usage/<GAME>/plugin_skip.json`, so each dump is checked once; editing the plugin file invalidates the cache. Check the name first and open the image only when needed.

## Plugins

| File | Game | Rules |
|---|---|---|
| `SLUS-20811.py` | Need for Speed: Underground | Skips scratch video memory dumped as textures, and car body skins (one per car/paint/vinyl combination, endless and about 5.6 MB each upscaled) |
