"""Runs inside RenderDoc's own Python (qrenderdoc --python), started by texture_usage.py; not meant to be run directly.

Launches PCSX2 with RenderDoc injected, captures one frame every few seconds, and for each capture counts which
textures the game drew with. GPU textures are identified by content: a replacement is uploaded as the exact BC7
data of its .dds, and a texture without one as the decoded RGBA pixels of its dump. Both were hashed beforehand by
texture_usage.py. Each capture is deleted once counted; totals are saved to the output JSON after every capture.

RenderDoc's Python is 3.8 with no third-party packages, so this file uses only the standard library.
"""
import hashlib
import json
import os
import sys
import time
import traceback

import renderdoc as rd

cfg = json.load(open(os.environ["USAGE_AGENT_CONFIG"]))
log_file = open(cfg["log"], "a")


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, file=log_file, flush=True)


def load_results():
    try:
        with open(cfg["output"]) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"captures": 0, "textures": {}, "unknown": {}}


def save_results(res):
    tmp = cfg["output"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    os.replace(tmp, cfg["output"])


def analyse(path, index, res):
    """Counts pixel-shader reads per known texture in one capture. Returns (known textures, draws) seen."""
    cap = rd.OpenCaptureFile()
    if cap.OpenFile(path, "", None) != rd.ResultCode.Succeeded:
        raise RuntimeError("can't open " + path)
    result, ctrl = cap.OpenCapture(rd.ReplayOptions(), None)
    if result != rd.ResultCode.Succeeded:
        cap.Shutdown()
        raise RuntimeError("can't replay %s: %s" % (path, result))
    try:
        seen = {}
        for tex in ctrl.GetTextures():
            draws = sum(1 for u in ctrl.GetUsage(tex.resourceId) if u.usage == rd.ResourceUsage.PS_Resource)
            if not draws:
                continue
            key = "%dx%d" % (tex.width, tex.height)
            if key not in index["sizes"]:  # render targets, palettes, ...: nothing from the dumps has this size
                res["unknown"][key] = res["unknown"].get(key, 0) + draws
                continue
            blob = bytes(ctrl.GetTextureData(tex.resourceId, rd.Subresource(0, 0, 0)))
            names = index["hashes"].get(hashlib.sha1(blob).hexdigest())
            name = names[0] if names else None  # identical-content names are counted under the first
            if name is None:
                res["unknown"][key] = res["unknown"].get(key, 0) + draws
            else:
                seen[name] = seen.get(name, 0) + draws
        for name, draws in seen.items():
            t = res["textures"].setdefault(name, {"frames": 0, "draws": 0})
            t["frames"] += 1
            t["draws"] += draws
        res["captures"] += 1
        return len(seen), sum(seen.values())
    finally:
        ctrl.Shutdown()
        cap.Shutdown()


def main():
    index = json.load(open(cfg["index"]))
    index["sizes"] = set(index["sizes"])
    res = load_results()

    xcb = rd.EnvironmentModification()  # RenderDoc can't capture Wayland windows: run PCSX2 under XWayland
    xcb.mod, xcb.sep, xcb.name, xcb.value = rd.EnvMod.Set, rd.EnvSep.NoSep, "QT_QPA_PLATFORM", "xcb"
    env = [xcb]
    opts = rd.CaptureOptions()
    exe = cfg["pcsx2"]
    launched = rd.ExecuteAndInject(exe, os.path.dirname(exe), "", env, cfg["capture_template"], opts, False)
    if launched.result != rd.ResultCode.Succeeded:
        log("FAILED to launch PCSX2 under RenderDoc: %s" % launched.result)
        return 2
    target = rd.CreateTargetControl("localhost", launched.ident, "pcsx2-upscaler", True)
    if target is None:
        log("FAILED to connect to PCSX2")
        return 2
    log("PCSX2 running; capturing every %ss" % cfg["interval"])

    next_capture = time.time() + cfg["interval"]
    while target.Connected():
        msg = target.ReceiveMessage(None)
        if msg.type == rd.TargetControlMessageType.Disconnected:
            break
        if msg.type == rd.TargetControlMessageType.NewCapture:
            path = msg.newCapture.path
            t0 = time.time()
            try:
                n, draws = analyse(path, index, res)
                save_results(res)
                log("capture %d: %d known textures, %d draws (%.1fs)" % (res["captures"], n, draws, time.time() - t0))
            except Exception as e:
                log("capture skipped: %s" % e)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
            next_capture = time.time() + cfg["interval"]
        if time.time() >= next_capture:
            target.TriggerCapture(1)
            next_capture = time.time() + 60  # re-armed when the capture arrives; retried if it never does
        time.sleep(0.1)
    target.Shutdown()
    log("PCSX2 closed")
    return 0


try:
    code = main()
except Exception:
    log(traceback.format_exc())
    code = 1
log_file.close()
os._exit(code)  # skip opening the RenderDoc UI
