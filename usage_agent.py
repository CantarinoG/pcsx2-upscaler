"""Runs inside RenderDoc's own Python (qrenderdoc --python), started by texture_usage.py; not meant to be run directly.

Launches PCSX2 with RenderDoc injected, captures one frame every few seconds, and for each capture counts which
textures the game drew with. GPU textures are identified by content: a replacement is uploaded as the exact BC7
data of its .dds, and a texture without one as the decoded RGBA pixels of its dump. Both were hashed beforehand by
texture_usage.py. Each capture is deleted once counted; totals are saved to the output JSON after every capture.

Frames are only captured while the chosen game is actually running: PCSX2's PINE interface is asked first, so time
spent paused, in the BIOS or in another game doesn't count.

RenderDoc's Python is 3.8 with no third-party packages, so this file uses only the standard library.
"""
import hashlib
import json
import os
import socket
import struct
import sys
import time
import traceback

import renderdoc as rd

cfg = json.load(open(os.environ["USAGE_AGENT_CONFIG"]))
log_file = open(cfg["log"], "a")


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, file=log_file, flush=True)


class Pine:
    """Minimal client for PCSX2's PINE IPC: request = [u32 size][u8 opcode], reply = [u32 size][u8 result][data]."""
    MSG_ID, MSG_STATUS = 0x0C, 0x0F
    STATUS = {0: "running", 1: "paused", 2: "shutdown"}

    def __init__(self, path):
        self.path, self.sock = path, None

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise OSError("PINE connection closed")
            buf += chunk
        return buf

    def _call(self, opcode):
        """Reply data, or None if PCSX2 can't be reached or refused the request."""
        try:
            if self.sock is None:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(2)
                s.connect(self.path)
                self.sock = s
            self.sock.sendall(struct.pack("<IB", 5, opcode))
            size = struct.unpack("<I", self._recv(4))[0]
            body = self._recv(size - 4)
        except OSError:
            if self.sock is not None:
                self.sock.close()
            self.sock = None
            return None
        return body[1:] if body[:1] == b"\x00" else None

    def state(self):
        """'running', 'paused', 'shutdown', or None when PCSX2 isn't reachable."""
        data = self._call(self.MSG_STATUS)
        return self.STATUS.get(struct.unpack_from("<I", data)[0]) if data and len(data) >= 4 else None

    def serial(self):
        data = self._call(self.MSG_ID)
        if not data or len(data) < 4:
            return None
        n = struct.unpack_from("<I", data)[0]
        return data[4:4 + n].split(b"\x00")[0].decode(errors="replace")


def why_not_capture(pine):
    """None when the chosen game is running, otherwise the reason to wait."""
    if pine is None:  # no PINE configured: capture regardless, as before
        return None
    state = pine.state()
    if state is None:
        return "waiting for PCSX2 (PINE not reachable yet)"
    if state != "running":
        return "emulation %s, not capturing" % state
    serial = pine.serial()
    if serial != cfg["serial"]:
        return "%s is running, not %s: not capturing" % (serial or "no game", cfg["serial"])
    return None


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


MAG_PERCENTILE = 0.95  # per texture and frame, the magnification of its 95th-percentile triangle is kept, so a
                       # single odd triangle can't set a texture's all-time maximum
MIN_AREA = 1.0         # triangles under 1 screen pixel or 1 texel are too small to measure


def triangle_mags(verts, tw, th, out):
    """Appends the magnification of each triangle: sqrt(screen area / texel area), in native pixels per texel.

    verts: (x, y, u, v) per vertex, three per triangle; x, y in native GS pixels, u, v normalised (1 = texture size)."""
    for i in range(0, len(verts) - 2, 3):
        (x0, y0, u0, v0), (x1, y1, u1, v1), (x2, y2, u2, v2) = verts[i:i + 3]
        screen = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2
        texels = abs((u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)) / 2 * tw * th
        if screen >= MIN_AREA and texels >= MIN_AREA:
            out.append((screen / texels) ** 0.5)


def measure_magnification(ctrl, uses, dims):
    """How magnified each texture was in this frame: {name: magnification}.

    uses: {eventId: [texture names sampled by that draw]}. Geometry is read straight from PCSX2's vertex and index
    buffers, which hold raw GS vertices (32 bytes: ST, RGBA, Q, XY in 1/16 pixels, Z, UV in 1/16 texels, FOG), so
    positions come out in native PS2 pixels whatever the internal resolution. Sprites PCSX2 expands in its vertex
    shader read those vertices as a storage buffer instead and aren't measured."""
    actions = {}

    def walk(acts):
        for a in acts:
            if a.flags & rd.ActionFlags.Drawcall:
                actions[a.eventId] = a
            walk(a.children)

    walk(ctrl.GetRootActions())
    events = sorted(e for e in uses if e in actions and actions[e].numIndices >= 3)
    if not events:
        return {}
    ctrl.SetFrameEvent(events[0], True)
    pipe = ctrl.GetPipelineState()
    vbufs, ibuf = pipe.GetVBuffers(), pipe.GetIBuffer()
    if not vbufs or vbufs[0].byteStride != 32 or ibuf.byteStride not in (2, 4):
        return {}
    vb_id, ib_id, istride = vbufs[0].resourceId, ibuf.resourceId, ibuf.byteStride
    as_vertices = {u.eventId for u in ctrl.GetUsage(vb_id) if u.usage == rd.ResourceUsage.VertexBuffer}
    events = [e for e in events if e in as_vertices]

    # The buffers are rings written as the frame goes: their contents are read at the end of each stretch where the
    # write offset only grows, before a wrap could overwrite them.
    runs, run = [], []
    for e in events:
        prev = actions[run[-1]] if run else None
        if prev and (actions[e].indexOffset < prev.indexOffset or actions[e].baseVertex < prev.baseVertex):
            runs.append(run)
            run = []
        run.append(e)
    runs.append(run)

    tris = {}
    for run in runs:
        ctrl.SetFrameEvent(run[-1], True)
        for e in run:
            a = actions[e]
            ib = bytes(ctrl.GetBufferData(ib_id, a.indexOffset * istride, a.numIndices * istride))
            idx = struct.unpack("<%d%s" % (a.numIndices, "H" if istride == 2 else "I"), ib)
            lo, hi = min(idx), max(idx)
            vb = bytes(ctrl.GetBufferData(vb_id, (a.baseVertex + lo) * 32, (hi - lo + 1) * 32))
            if len(vb) < (hi - lo + 1) * 32:
                continue
            raw = []
            for i in idx:
                s, t, q, xy, uv = struct.unpack_from("<2f4xfI4xI", vb, (i - lo) * 32)
                raw.append(((xy & 0xFFFF) / 16.0, (xy >> 16) / 16.0, s, t, q, (uv & 0xFFFF) / 16.0, (uv >> 16) / 16.0))
            # Integer texel coordinates (FST: sprites, UI) vary in UV; perspective ones (STQ) leave UV at a constant
            fst = len({(r[5], r[6]) for r in raw}) > 1
            for name in uses[e]:
                tw, th = dims.get(name, (0, 0))
                if not tw or not th:
                    continue
                if fst:
                    verts = [(r[0], r[1], r[5] / tw, r[6] / th) for r in raw]
                else:
                    verts = [(r[0], r[1], r[2] / r[4], r[3] / r[4]) if r[4] else (r[0], r[1], 0.0, 0.0) for r in raw]
                triangle_mags(verts, tw, th, tris.setdefault(name, []))
    mags = {}
    for name, ms in tris.items():
        if ms:
            ms.sort()
            mags[name] = ms[min(len(ms) - 1, int(len(ms) * MAG_PERCENTILE))]
    return mags


def analyse(path, index, res):
    """Counts pixel-shader reads per known texture in one capture, and records how magnified each was.
    Returns (known textures, draws) seen."""
    cap = rd.OpenCaptureFile()
    if cap.OpenFile(path, "", None) != rd.ResultCode.Succeeded:
        raise RuntimeError("can't open " + path)
    result, ctrl = cap.OpenCapture(rd.ReplayOptions(), None)
    if result != rd.ResultCode.Succeeded:
        cap.Shutdown()
        raise RuntimeError("can't replay %s: %s" % (path, result))
    try:
        seen, uses = {}, {}
        for tex in ctrl.GetTextures():
            events = [u.eventId for u in ctrl.GetUsage(tex.resourceId) if u.usage == rd.ResourceUsage.PS_Resource]
            draws = len(events)
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
                for e in events:
                    uses.setdefault(e, []).append(name)
        try:
            mags = measure_magnification(ctrl, uses, index.get("dims", {}))
        except Exception as e:  # counting still holds without it
            log("magnification not measured: %r" % e)
            mags = {}
        for name, draws in seen.items():
            t = res["textures"].setdefault(name, {"frames": 0, "draws": 0})
            t["frames"] += 1
            t["draws"] += draws
            if name in mags and mags[name] > t.get("max_mag", 0):
                t["max_mag"] = round(mags[name], 2)
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
    log("PCSX2 running; capturing every %ss of gameplay" % cfg["interval"])

    pine = Pine(cfg["pine"]) if cfg.get("pine") else None
    waiting = None  # reason logged for the current wait, so it's printed once
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
            reason = why_not_capture(pine)
            if reason is None:
                if waiting is not None:
                    log("game running again, capturing")
                    waiting = None
                    # the interval restarts so a frame isn't grabbed the instant the game resumes
                    next_capture = time.time() + cfg["interval"]
                    continue
                target.TriggerCapture(1)
                next_capture = time.time() + 60  # re-armed when the capture arrives; retried if it never does
            else:
                if reason != waiting:
                    log(reason)
                    waiting = reason
                next_capture = time.time() + 1
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
