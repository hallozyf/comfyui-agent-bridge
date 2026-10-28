#!/usr/bin/env python3
"""Minimal, dependency-free client for ComfyUI Agent Bridge.

Works with the `comfyui-agent-bridge` custom node (`/agent/*`), and falls back to
the raw ComfyUI API if the node is not installed.

    python comfy_agent.py status
    python comfy_agent.py models
    python comfy_agent.py gen --preset flux --prompt "a fox in snow" --out out.png
    python comfy_agent.py gen --preset sdxl --ckpt my_model.safetensors --steps 30 --out out.png
    python comfy_agent.py gen --workflow my.api.json --out out.png
    python comfy_agent.py free

Env: COMFY_URL (default http://127.0.0.1:8188)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
UA = "comfyui-agent-bridge-client/1.2"


def api(path: str, payload=None, timeout: int = 60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"User-Agent": UA,
                                          **({"Content-Type": "application/json"} if data else {})})
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with op.open(req, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:800]
        raise SystemExit(f"HTTP {e.code} on {path}: {detail}")
    return json.loads(body) if body else None


def alive() -> bool:
    try:
        api("/system_stats", timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


def cmd_status(_a) -> int:
    if not alive():
        print(f"ComfyUI: DOWN ({BASE})")
        return 1
    st = api("/system_stats")
    dev = (st.get("devices") or [{}])[0]
    print(f"ComfyUI: up at {BASE}")
    sysinfo = st.get("system") or {}
    print(f"  torch {sysinfo.get('pytorch_version') or '?'}  "
          f"cuda {sysinfo.get('cuda_version') or 'n/a'}")
    print(f"  gpu: {dev.get('name')}  "
          f"vram_free={dev.get('vram_free', 0) / (1 << 30):.2f}GB / "
          f"{dev.get('vram_total', 0) / (1 << 30):.2f}GB")
    try:
        h = api("/agent/health", timeout=10)
        print(f"  bridge: {h.get('bridge')} v{h.get('version')}  gguf={h.get('gguf_nodes')}")
    except Exception:  # noqa: BLE001
        print("  bridge: not installed (raw API mode)")
    return 0


def cmd_models(_a) -> int:
    try:
        d = api("/agent/models")
    except SystemExit:
        d = {}
        for folder in ("checkpoints", "diffusion_models", "vae", "text_encoders"):
            try:
                d[folder] = api(f"/models/{folder}")
            except SystemExit:
                pass
    for k in sorted(d or {}):
        print(f"{k}:")
        for f in d[k]:
            print(f"  - {f}")
    return 0


def _wait(prompt_id: str, timeout: float) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = api(f"/agent/result/{prompt_id}", timeout=30)
        except SystemExit:
            r = None
        if r and r.get("state") in ("done", "error"):
            return r
        if not r:  # bridge missing -> native history
            try:
                h = api(f"/history/{prompt_id}", timeout=30).get(prompt_id)
            except Exception:  # noqa: BLE001
                h = None
            if h:
                st = h.get("status") or {}
                imgs = [i for o in (h.get("outputs") or {}).values() for i in o.get("images", [])]
                if st.get("completed") and imgs:
                    return {"state": "done", "images": imgs}
                if st.get("status_str") == "error":
                    return {"state": "error", "messages": st.get("messages")}
        time.sleep(1.5)
    return {"state": "timeout"}


def cmd_gen(a) -> int:
    if not alive():
        raise SystemExit(f"ComfyUI is not running at {BASE}")
    if a.workflow:
        wf = json.load(open(a.workflow, encoding="utf-8"))
        graph = wf.get("prompt", wf)
        for pair in a.set or []:
            key, _, val = pair.partition("=")
            nid, _, field = key.partition(".")
            try:
                val = json.loads(val)
            except json.JSONDecodeError:
                pass
            graph[nid]["inputs"][field] = val
        prompt_id = api("/prompt", {"prompt": graph, "client_id": "agent-client"})["prompt_id"]
        print(f"submitted workflow {a.workflow} -> {prompt_id}")
    else:
        params = {k: v for k, v in {
            "prompt": a.prompt, "negative": a.negative, "width": a.width, "height": a.height,
            "steps": a.steps, "cfg": a.cfg, "guidance": a.guidance, "seed": a.seed,
            "batch_size": a.batch, "ckpt_name": a.ckpt, "unet_name": a.unet,
            "unet_gguf": a.gguf, "t5_name": a.t5,
        }.items() if v is not None}
        r = api("/agent/txt2img", {"preset": a.preset, "params": params}, timeout=120)
        if "error" in r:
            raise SystemExit("bridge rejected the request: " + json.dumps(r, ensure_ascii=False)[:800])
        prompt_id = r["prompt_id"]
        print(f"submitted preset={a.preset} seed={r.get('seed')} -> {prompt_id}")

    t0 = time.time()
    res = _wait(prompt_id, a.timeout)
    dt = time.time() - t0
    if res.get("state") != "done":
        print(f"generation {res.get('state')} after {dt:.0f}s")
        if res.get("messages"):
            print(json.dumps(res["messages"], ensure_ascii=False)[:1200])
        return 2

    out = a.out or f"out_{int(time.time())}.png"
    for i, img in enumerate(res.get("images") or []):
        q = urllib.parse.urlencode({"filename": img.get("filename", ""),
                                    "subfolder": img.get("subfolder", ""),
                                    "type": img.get("type", "output")})
        req = urllib.request.Request(f"{BASE}/view?{q}", headers={"User-Agent": UA})
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with op.open(req, timeout=120) as r:
            data = r.read()
        dst = out if len(res["images"]) == 1 else os.path.splitext(out)[0] + f"_{i}.png"
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        open(dst, "wb").write(data)
        print(f"saved {dst} ({len(data) / 1024:.0f} KB)")
    print(f"done in {dt:.1f}s")
    return 0


def cmd_free(_a) -> int:
    try:
        api("/agent/free", {}, timeout=60)
    except SystemExit:
        api("/free", {"unload_models": True, "free_memory": True}, timeout=120)
    print("models unloaded, cache freed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ComfyUI Agent Bridge client")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("models").set_defaults(func=cmd_models)
    sub.add_parser("free").set_defaults(func=cmd_free)
    g = sub.add_parser("gen")
    g.add_argument("--preset", default="sdxl", choices=["sd15", "sdxl", "flux"])
    g.add_argument("--prompt")
    g.add_argument("--negative")
    g.add_argument("--width", type=int)
    g.add_argument("--height", type=int)
    g.add_argument("--steps", type=int)
    g.add_argument("--cfg", type=float)
    g.add_argument("--guidance", type=float)
    g.add_argument("--seed", type=int)
    g.add_argument("--batch", type=int)
    g.add_argument("--ckpt")
    g.add_argument("--unet")
    g.add_argument("--gguf")
    g.add_argument("--t5")
    g.add_argument("--workflow", help="API-format JSON instead of a preset")
    g.add_argument("--set", action="append", metavar="ID.FIELD=VALUE")
    g.add_argument("--out")
    g.add_argument("--timeout", type=float, default=1800)
    g.set_defaults(func=cmd_gen)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
