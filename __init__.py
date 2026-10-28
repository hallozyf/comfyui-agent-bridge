"""ComfyUI Agent Bridge
====================

Agent-friendly REST endpoints for ComfyUI, as a normal custom node.

Why: driving ComfyUI from an external agent usually means assembling an
"API format" prompt graph by hand, guessing which loader nodes this install
actually has, and remembering that quantized models need a different loader
class. This node collapses all of that into one HTTP call:

    POST /agent/txt2img   {"preset": "flux", "params": {...}}  ->  {"prompt_id": ...}
    GET  /agent/result/{prompt_id}                              ->  {"state": "done", "images": [...]}

Endpoints
---------
    GET  /agent/health              device, VRAM, torch, available presets, GGUF status
    GET  /agent/models              model files that really exist on disk (incl. .gguf)
    POST /agent/workflow            {preset, params} -> {"prompt": <API graph>} (no queueing)
    POST /agent/txt2img             {preset, params} -> {"prompt_id", "seed", "workflow"}
    GET  /agent/result/{prompt_id}  poll: pending / done / error, with image names
    POST /agent/free                unload models, free VRAM

`/dsh/...` is kept as an alias of every route so older callers keep working.

Presets: `sd15`, `sdxl`, `flux` (safetensors UNET, GGUF UNET or an all-in-one
checkpoint -- chosen automatically from what is installed).

The graph builder validates every input against the live node registry
(nodes.NODE_CLASS_MAPPINGS), so a renamed widget raises a clear error instead of
producing a workflow that fails later with "Required input is missing".

It also registers the `.gguf` extension with folder_paths: current
ComfyUI-GGUF builds no longer do that, and without it quantized models are
invisible in the node dropdowns and to get_filename_list().

MIT licensed. See README.md for install and usage.
"""

import json
import os
import time
import uuid

import folder_paths
import nodes as comfy_nodes
from aiohttp import web
from server import PromptServer

# Quantized checkpoints must be visible to folder_paths (and therefore to every
# loader dropdown in the GUI) -- current ComfyUI-GGUF no longer adds this itself.
_GGUF_READY = False


def ensure_gguf_visible(force: bool = False) -> bool:
    """Register .gguf with folder_paths and drop the cached file lists."""
    global _GGUF_READY
    if _GGUF_READY and not force:
        return True
    try:
        folder_paths.supported_pt_extensions.add(".gguf")
        for key in ("diffusion_models", "unet", "text_encoders", "clip",
                    "checkpoints", "loras", "vae"):
            entry = folder_paths.folder_names_and_paths.get(key)
            if entry:
                dirs = list(entry[0])
                exts = set(entry[1]) | {".gguf"}
                folder_paths.folder_names_and_paths[key] = (dirs, exts)
        if hasattr(folder_paths, "filename_list_cache"):
            folder_paths.filename_list_cache.clear()
        if hasattr(folder_paths, "cache_helper"):
            folder_paths.cache_helper.clear()
        _GGUF_READY = True
    except Exception as e:  # noqa: BLE001
        print(f"[agent-bridge] .gguf visibility patch failed: {e}")
        return False
    return True


ensure_gguf_visible()


NODE_CLASS = "AgentBridgeInfo"
NODE_NAME = "Agent Bridge Info"

# ---------------------------------------------------------------- node class


class AgentBridgeInfo:
    """Shows up in the node menu so the bridge is discoverable from the GUI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {}}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "agent_bridge"
    OUTPUT_NODE = False

    def noop(self):
        return {}


NODE_CLASS_MAPPINGS = {NODE_CLASS: AgentBridgeInfo}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_CLASS: NODE_NAME}


# ------------------------------------------------------------------- helpers

def _node_input_names(class_type: str):
    """Full (required+optional) input name set for a node class, or None."""
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_type)
    if cls is None:
        return None
    try:
        spec = cls.INPUT_TYPES()
    except Exception:  # noqa: BLE001
        return None
    names = {}
    for section in ("required", "optional"):
        for key, val in (spec.get(section) or {}).items():
            names[key] = val
    return names


def _clean(class_type: str, inputs: dict) -> dict:
    """Drop inputs the installed node does not declare; fill required defaults."""
    names = _node_input_names(class_type)
    if names is None:
        raise ValueError(f"node '{class_type}' is not installed in this ComfyUI")
    out = {k: v for k, v in inputs.items() if k in names}
    missing = [k for k in names if k not in out and _is_required(class_type, k)]
    if missing and not _has_default(class_type, missing[0]):
        raise ValueError(f"node '{class_type}' requires inputs not provided: {missing}")
    for k in missing:
        spec = names[k]
        if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) \
                and "default" in spec[1]:
            out[k] = spec[1]["default"]
    return out


def _is_required(class_type: str, name: str) -> bool:
    cls = comfy_nodes.NODE_CLASS_MAPPINGS[class_type]
    try:
        return name in (cls.INPUT_TYPES().get("required") or {})
    except Exception:  # noqa: BLE001
        return False


def _has_default(class_type: str, name: str) -> bool:
    cls = comfy_nodes.NODE_CLASS_MAPPINGS[class_type]
    try:
        spec = (cls.INPUT_TYPES().get("required") or {})[name]
    except Exception:  # noqa: BLE001
        return False
    return isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) \
        and "default" in spec[1]


MODEL_EXTS = (".safetensors", ".gguf", ".ckpt", ".pt", ".pt2", ".pth", ".bin", ".sft", ".pkl")


def _file_list(folder: str) -> list[str]:
    """Model files for a ComfyUI folder, merged with a direct disk scan.

    The disk scan guarantees .gguf files show up even if the folder_paths
    extension registration or its cache is out of date.
    """
    ensure_gguf_visible()
    found: set[str] = set()
    try:
        found.update(folder_paths.get_filename_list(folder))
    except Exception:  # noqa: BLE001
        pass
    dirs = []
    try:
        dirs = list(folder_paths.get_folder_paths(folder))
    except Exception:  # noqa: BLE001
        dirs = []
    if not dirs:
        entry = folder_paths.folder_names_and_paths.get(folder)
        if entry:
            dirs = list(entry[0])
    for root_dir in dirs:
        if not os.path.isdir(root_dir):
            continue
        for root, _subdirs, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(MODEL_EXTS):
                    rel = os.path.relpath(os.path.join(root, f), root_dir).replace("\\", "/")
                    found.add(rel)
    return sorted(found)


def _first(folder: str, patterns: list[str], any_if_none: bool = False) -> str | None:
    """First existing filename matching one of the patterns (substring, ci)."""
    files = _file_list(folder)
    for pat in patterns:
        for real in files:
            if pat in real.lower():
                return real
    if any_if_none:
        return files[0] if files else None
    return None


# ------------------------------------------------------------- graph builder

def build_workflow(preset: str, p: dict) -> dict:
    """Return an API-format prompt graph for the requested preset."""
    preset = (preset or "sdxl").lower()
    prompt = p.get("prompt") or "a photograph of a red fox in a snowy forest, golden hour"
    negative = p.get("negative", "")
    width = int(p.get("width") or (512 if preset == "sd15" else 1024))
    height = int(p.get("height") or (512 if preset == "sd15" else 1024))
    seed = int(p.get("seed", -1))
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "big") % (2 ** 31)
    prefix = p.get("filename_prefix", "dsh/%s" % preset)
    batch = int(p.get("batch_size", 1))

    if preset == "flux":
        # resolved inside the flux branch below
        pass
    elif preset == "sd15":
        width = int(p.get("width") or 512)
        height = int(p.get("height") or 768)
    elif preset == "sdxl":
        width = int(p.get("width") or 1024)
        height = int(p.get("height") or 1024)

    g: dict[str, dict] = {}

    if preset == "flux":
        unet_sd = p.get("unet_name")
        unet_gguf = p.get("unet_gguf") or _first("diffusion_models", ["flux", "gguf"])
        steps = int(p.get("steps", 20))
        cfg = float(p.get("cfg", 1.0))
        sampler = p.get("sampler_name", "euler")
        sched = p.get("scheduler", "simple")
        guidance = float(p.get("guidance", 3.5))

        if unet_sd:
            g["1"] = {"class_type": "UNETLoader",
                      "inputs": _clean("UNETLoader", {
                          "unet_name": unet_sd,
                          "weight_dtype": p.get("weight_dtype", "default")})}
        elif unet_gguf and "UnetLoaderGGUF" in comfy_nodes.NODE_CLASS_MAPPINGS:
            g["1"] = {"class_type": "UnetLoaderGGUF",
                      "inputs": _clean("UnetLoaderGGUF", {"unet_name": unet_gguf})}
        else:
            ckpt = p.get("ckpt_name") or _first("checkpoints", ["flux1"])
            if not ckpt:
                raise ValueError(
                    "no FLUX diffusion model found: put a .gguf in models/diffusion_models "
                    "or a flux1-*.safetensors checkpoint in models/checkpoints")
            g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}

        if g["1"]["class_type"] == "CheckpointLoaderSimple":
            model_ref, clip_ref, vae_ref = ["1", 0], ["1", 1], ["1", 2]
        else:
            vae = p.get("vae_name") or _first("vae", ["ae.safetensors", "ae", "flux"], True)
            g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae}}
            model_ref, vae_ref = ["1", 0], ["3", 0]
            t5 = p.get("t5_name") or _first("text_encoders", ["t5"], True)
            clip_l = p.get("clip_l_name") or _first("text_encoders", ["clip_l"])
            if t5 and t5.lower().endswith(".gguf") and "DualCLIPLoaderGGUF" in comfy_nodes.NODE_CLASS_MAPPINGS:
                g["2"] = {"class_type": "DualCLIPLoaderGGUF",
                          "inputs": _clean("DualCLIPLoaderGGUF",
                                           {"clip_name1": clip_l, "clip_name2": t5,
                                            "type": p.get("clip_type", "flux")})}
            else:
                g["2"] = {"class_type": "DualCLIPLoader",
                          "inputs": _clean("DualCLIPLoader",
                                           {"clip_name1": clip_l, "clip_name2": t5,
                                            "type": p.get("clip_type", "flux"),
                                            "device": p.get("clip_device", "default")})}
            clip_ref = ["2", 0]

        g["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_ref}}
        g["6"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}
        pos_src = ["4", 0]
        if "FluxGuidance" in comfy_nodes.NODE_CLASS_MAPPINGS:
            g["7"] = {"class_type": "FluxGuidance",
                      "inputs": {"conditioning": pos_src, "guidance": guidance}}
            pos_src = ["7", 0]
        g["8"] = {"class_type": "EmptySD3LatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": batch}}
        g["9"] = {"class_type": "KSampler",
                  "inputs": _clean("KSampler", {
                      "seed": seed, "steps": steps, "cfg": cfg,
                      "sampler_name": sampler, "scheduler": sched, "denoise": 1.0,
                      "model": model_ref, "positive": pos_src,
                      "negative": ["6", 0], "latent_image": ["8", 0]})}
        g["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": vae_ref}}
        g["11"] = {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}}
        return g

    # classic single-file checkpoints (sd15 / sdxl)
    if preset == "sd15":
        ckpt = p.get("ckpt_name") or _first("checkpoints",
                                            ["realistic_vision", "dreamshaper", "v1-5"], True)
        steps, cfg = int(p.get("steps", 30)), float(p.get("cfg", 7.0))
    else:
        ckpt = p.get("ckpt_name") or _first("checkpoints",
                                            ["sd_xl_base", "juggernaut", "xl"], True)
        steps, cfg = int(p.get("steps", 28)), float(p.get("cfg", 6.5))
    sampler = p.get("sampler_name", "dpmpp_2m")
    sched = p.get("scheduler", "karras")
    if not ckpt:
        raise ValueError(f"no checkpoint available for preset '{preset}'")

    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}
    g["2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}}
    g["3"] = {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative or "blurry, low quality, watermark, text", "clip": ["1", 1]}}
    g["4"] = {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": batch}}
    g["5"] = {"class_type": "KSampler",
              "inputs": _clean("KSampler", {
                  "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler,
                  "scheduler": sched, "denoise": 1.0, "model": ["1", 0],
                  "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]})}
    g["6"] = {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}}
    g["7"] = {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}}
    return g


# ------------------------------------------------------------ self-HTTP glue

def _base_url() -> str:
    srv = PromptServer.instance
    addr = getattr(srv, "address", "127.0.0.1") or "127.0.0.1"
    if addr in ("0.0.0.0", "::"):
        addr = "127.0.0.1"
    return f"http://{addr}:{getattr(srv, 'port', 8188)}"


async def _call(method: str, path: str, payload: dict | None = None, timeout: int = 30):
    import aiohttp

    url = _base_url() + path
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as sess:
        async with sess.request(method, url, json=payload) as resp:
            return resp.status, await resp.json(content_type=None)


# ---------------------------------------------------------------- endpoints

@PromptServer.instance.routes.get("/dsh/health")
async def dsh_health(request):
    import torch

    info = {
        "ok": True,
        "bridge": "comfyui-agent-bridge",
        "version": "1.2.0",
        "pid": os.getpid(),
    }
    try:
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "vram_total_gb": round(total / (1 << 30), 2),
                "vram_free_gb": round(free / (1 << 30), 2),
            }
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = repr(e)[:200]
    info["gguf_nodes"] = [n for n in ("UnetLoaderGGUF", "DualCLIPLoaderGGUF")
                          if n in comfy_nodes.NODE_CLASS_MAPPINGS]
    info["gguf_extension_registered"] = ".gguf" in folder_paths.supported_pt_extensions
    info["presets"] = ["sd15", "sdxl", "flux"]
    return web.json_response(info)


@PromptServer.instance.routes.get("/dsh/models")
async def dsh_models(request):
    out = {}
    for folder in ("checkpoints", "diffusion_models", "unet", "vae", "text_encoders",
                   "loras", "controlnet", "upscale_models", "clip_vision"):
        files = _file_list(folder)
        if files:
            out[folder] = files
    return web.json_response(out)


@PromptServer.instance.routes.post("/dsh/workflow")
async def dsh_workflow(request):
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return web.json_response({"error": "invalid json"}, status=400)
    ensure_gguf_visible(force=True)  # pick up model files added since startup
    try:
        g = build_workflow(body.get("preset", "sdxl"), body.get("params", {}))
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response({"preset": body.get("preset", "sdxl"), "prompt": g})


@PromptServer.instance.routes.post("/dsh/txt2img")
async def dsh_txt2img(request):
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return web.json_response({"error": "invalid json"}, status=400)
    preset = body.get("preset", "sdxl")
    params = body.get("params", {})
    ensure_gguf_visible(force=True)  # pick up model files added since startup
    prefix = params.get("filename_prefix") or f"dsh/{preset}"
    params["filename_prefix"] = f"{prefix}/{uuid.uuid4().hex[:8]}"
    try:
        g = build_workflow(preset, params)
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": str(e)}, status=400)

    payload = {"prompt": g, "client_id": f"dsh-bridge-{uuid.uuid4().hex[:8]}"}
    try:
        status, data = await _call("POST", "/prompt", payload, timeout=60)
    except Exception as e:  # noqa: BLE001
        return web.json_response({"error": f"queue call failed: {e!r}"}, status=502)
    if status != 200:
        return web.json_response({"error": "comfyui rejected the prompt", "detail": data}, status=400)
    return web.json_response({
        "prompt_id": data.get("prompt_id"),
        "number": data.get("number"),
        "preset": preset,
        "seed": next((n["inputs"].get("seed") for n in g.values()
                      if n["class_type"] in ("KSampler", "KSamplerAdvanced")
                      and "seed" in n["inputs"]), None),
        "workflow": g,
    })


@PromptServer.instance.routes.get("/dsh/result/{prompt_id}")
async def dsh_result(request):
    prompt_id = request.match_info["prompt_id"]
    try:
        status, hist = await _call("GET", f"/history/{prompt_id}", timeout=30)
    except Exception as e:  # noqa: BLE001
        return web.json_response({"state": "error", "error": repr(e)[:200]}, status=502)
    if status != 200:
        return web.json_response({"state": "unknown", "http": status}, status=502)
    entry = (hist or {}).get(prompt_id)
    if not entry:
        return web.json_response({"state": "pending", "prompt_id": prompt_id})
    st = entry.get("status") or {}
    done = st.get("completed") or st.get("status_str") in ("success", "error")
    images = []
    for _node_id, out in (entry.get("outputs") or {}).items():
        for img in out.get("images", []) or []:
            images.append(img)
    state = "done" if (done and images) else ("error" if st.get("status_str") == "error" else "pending")
    return web.json_response({
        "state": state,
        "prompt_id": prompt_id,
        "images": images,
        "messages": st.get("messages") or [],
        "status_str": st.get("status_str"),
    })


@PromptServer.instance.routes.post("/dsh/free")
async def dsh_free(request):
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return web.json_response({"ok": True, "unloaded": True, "at": time.time()})


# -- canonical /agent/* routes (the /dsh/* decorators above stay as aliases) --
_routes = PromptServer.instance.routes
_routes.get("/agent/health")(dsh_health)
_routes.get("/agent/models")(dsh_models)
_routes.post("/agent/workflow")(dsh_workflow)
_routes.post("/agent/txt2img")(dsh_txt2img)
_routes.get("/agent/result/{prompt_id}")(dsh_result)
_routes.post("/agent/free")(dsh_free)


WEB_DIRECTORY = None
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
