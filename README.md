# ComfyUI Agent Bridge

> **DSH-only plugin / 为 DSH 打造的插件**
>
> This node was written **for [DSH (DeepSeek Harness)](https://github.com/hallozyf)** —
> the desktop agent runtime whose LLM tool loop drives ComfyUI. Its endpoint shapes
> (`/agent/txt2img` → `prompt_id` → `/agent/result/{id}`), preset defaults, automatic
> loader selection and error messages all assume **an agent is driving, not a human
> clicking the canvas**. It is installed like any custom node, but treat it as a
> DSH-side integration: the `dsh/*` output subfolder, the auto model picking and the
> "one call in, image out" contract are not aimed at interactive use.
>
> It is not technically locked to DSH — any HTTP client (curl, Python, another agent
> framework) can call the same endpoints — but DSH is the only environment it was
> built and verified in. If you are looking for a GUI-side helper node, this is not it.
>
> 中文说明见文末「关于 DSH」一节。

Agent-friendly REST endpoints for [ComfyUI](https://github.com/comfyanonymous/ComfyUI),
packaged as a normal custom node. Text-to-image becomes **one HTTP call** instead of
hand-assembling an API-format prompt graph.

```bash
curl -X POST http://127.0.0.1:8188/agent/txt2img \
  -H "Content-Type: application/json" \
  -d '{"preset":"sdxl","params":{"prompt":"a red fox in a snowy forest, golden hour"}}'
# -> {"prompt_id":"...","seed":123456,"workflow":{...}}

curl http://127.0.0.1:8188/agent/result/<prompt_id>
# -> {"state":"done","images":[{"filename":"agent_00001_.png","subfolder":"dsh/sdxl","type":"output"}]}
```

* No pip dependencies, no extra process, no extra port.
* Picks the right loader nodes for what you actually have installed
  (`CheckpointLoaderSimple` / `UNETLoader` / `UnetLoaderGGUF` + `DualCLIPLoader(GGUF)`).
* Validates the graph against the live node registry before queueing, so a renamed
  widget produces `node 'DualCLIPLoader' requires inputs not provided: ['device']`
  instead of a mysterious 400 later.
* Fixes a real gap: current `ComfyUI-GGUF` builds no longer register `.gguf` with
  `folder_paths`, so quantized models disappear from node dropdowns. This node
  registers it (and clears the file-list cache) at startup and per request.

---

## Why not an MCP server?

MCP is great when the agent already speaks MCP. For a plain script/LLM tool loop it adds a
second process, a second protocol and a second failure surface for no new capability — the
useful part is always the same: `POST /prompt`, then wait, then `/view`. This node wraps
exactly that, server-side, where the node registry is available. If you do want MCP, wrap
these two endpoints in your server of choice in ~30 lines.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hallozyf/comfyui-agent-bridge
# restart ComfyUI
```

There is nothing to `pip install`. Requirements:

| | |
|---|---|
| ComfyUI | any recent version (tested on 0.42.0) |
| Python | 3.10+ (tested on 3.14) |
| Optional | [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) for `.gguf` presets |

Check it loaded: `curl http://127.0.0.1:8188/agent/health`

```json
{
  "ok": true, "bridge": "comfyui-agent-bridge", "version": "1.2.0",
  "torch": "2.9.1+cu128", "cuda_available": true,
  "gpu": {"name": "NVIDIA GeForce RTX 5060 Laptop GPU", "capability": "12.0",
          "vram_total_gb": 7.93, "vram_free_gb": 6.82},
  "gguf_nodes": ["UnetLoaderGGUF", "DualCLIPLoaderGGUF"],
  "gguf_extension_registered": true,
  "presets": ["sd15", "sdxl", "flux"]
}
```

## API

| Method | Path | Body / Query | Returns |
|---|---|---|---|
| GET | `/agent/health` | — | device, VRAM, torch, presets, GGUF status |
| GET | `/agent/models` | — | `{checkpoints: [...], diffusion_models: [...], vae: [...], text_encoders: [...]}` |
| POST | `/agent/workflow` | `{preset, params}` | `{prompt: <API graph>}` — builds only, does not queue |
| POST | `/agent/txt2img` | `{preset, params}` | `{prompt_id, number, preset, seed, workflow}` |
| GET | `/agent/result/{prompt_id}` | — | `{state: pending\|done\|error, images, messages}` |
| POST | `/agent/free` | `{}` | unloads models, frees VRAM |

`/dsh/...` is an alias of every route, so older scripts keep working.

### `params`

| key | default | notes |
|---|---|---|
| `prompt` | built-in sample | required in practice |
| `negative` | quality negatives (sd15/sdxl) | ignored for `flux` (FluxGuidance instead) |
| `width` / `height` | 512×512 sd15, 1024×1024 sdxl/flux | Flux uses `EmptySD3LatentImage` |
| `steps` | 30 sd15, 28 sdxl, 20 flux | |
| `cfg` | 7.0 sd15, 6.5 sdxl, 1.0 flux | flux wants 1.0 |
| `guidance` | 3.5 | flux only (`FluxGuidance`) |
| `sampler_name` / `scheduler` | dpmpp_2m/karras, flux: euler/simple | |
| `seed` | random | returned in the response so a run is reproducible |
| `batch_size` | 1 | |
| `ckpt_name` | auto | put a file in `models/checkpoints` and pass it explicitly to override |
| `unet_name` | auto | `models/diffusion_models/*.safetensors` |
| `unet_gguf` | auto | `models/diffusion_models/*.gguf` |
| `vae_name` / `t5_name` / `clip_l_name` | auto | |
| `filename_prefix` | `dsh/<preset>` | a uuid is appended per request |

### Auto-selection

The `flux` preset walks this list and uses the first thing available:

1. `unet_name` given → `UNETLoader` (+ `DualCLIPLoader`, `VAELoader`)
2. a `.gguf` present (or `unet_gguf` given) → `UnetLoaderGGUF` (+ `DualCLIPLoaderGGUF`)
3. otherwise a `flux1*` checkpoint → `CheckpointLoaderSimple`

If the T5 encoder is a `.gguf`, that decides `DualCLIPLoaderGGUF` vs `DualCLIPLoader`
automatically. `clip_l.safetensors` + `t5.gguf` mixed in one GGUF dual loader is fine.

## Examples

`examples/` holds two API-format graphs you can post straight to ComfyUI's own
`/prompt` (no bridge needed) or pass the same node ids to `--set` overrides:

```bash
# SDXL, plain checkpoint
curl -X POST http://127.0.0.1:8188/prompt \
  -H "Content-Type: application/json" \
  -d @examples/sdxl.api.json
```

## Standalone client

`client/comfy_agent.py` is a dependency-free CLI (stdlib only) that talks to these
endpoints, so a shell/agent can generate and save a PNG in one command:

```bash
python client/comfy_agent.py status
python client/comfy_agent.py models
python client/comfy_agent.py gen --preset flux --prompt "a fox in snow" --out out.png
python client/comfy_agent.py gen --preset sdxl --ckpt "juggernaut_fp8.safetensors" --steps 30 --out out.png
python client/comfy_agent.py free
```

## Low-VRAM notes

The bridge itself is memory-neutral, but ComfyUI's defaults are not friendly to small cards.
On an 8 GB / 16 GB box these flags were needed for SDXL and Flux:

```
python main.py --reserve-vram 0.6 --cache-none --disable-pinned-memory
```

`--disable-pinned-memory` matters most: the default pins several GB of system RAM, which
turns into `DefaultCPUAllocator: not enough memory` on a 16 GB machine.
`--normalvram` no longer exists and `--highvram` will OOM an 8 GB card.

## Compatibility

Tested end to end on: Windows 11, ComfyUI 0.42.0, Python 3.14, torch 2.9.1+cu128,
RTX 5060 Laptop (8 GB), with SD1.5 / SDXL / SDXL fp8 / FLUX.1-dev Q4 / FLUX.1-schnell Q4.
Nothing in the code is Windows specific; the graph builder only reads ComfyUI's own
registries, so it follows the installed version rather than assuming one.

## License

MIT — see [LICENSE](LICENSE).

---

# 中文说明

## 关于 DSH（重要）

**这个插件是给 DSH（DeepSeek Harness）专用的**，不是通用 ComfyUI 插件：

- 它是在 DSH 里写出来并验证的，接口形状（一次提交拿 `prompt_id`、再轮询 `result`）就是按
  LLM 工具循环设计的，目的是让智能体少绕弯路；
- 默认值、模型自动选型、报错文案都按「智能体在驱动」而不是「人在点界面」来设计
  （比如自动挑 GGUF 还是 safetensors、自动补 `filename_prefix`、错误直接告诉缺哪个节点）；
- 出图默认落在 `dsh/<preset>` 子目录，这是 DSH 会话的归档约定。

它**不锁定** DSH：任何 HTTP 客户端（curl / Python / 别的 agent 框架）都能直接调同样接口；
但 DSH 是唯一经过完整验证的环境。如果你要的是「在界面上点着用」的节点，这个不是。

## 这是什么

把「让智能体驱动 ComfyUI 出图」收敛成几次 HTTP 调用：插件挂在 ComfyUI 自己的 web 服务上，
自动根据你装了哪些加载器节点（普通 safetensors / GGUF / 单体 checkpoint）拼好工作流，
不用外部脚本手写 API 格式的节点图。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hallozyf/comfyui-agent-bridge
# 重启 ComfyUI
```

无需 `pip install`，无需额外进程和端口。要跑 FLUX 量化模型请另外装
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)。

## 用法

```bash
# 健康检查（显卡、显存、可用预设、GGUF 是否就绪）
curl http://127.0.0.1:8188/agent/health

# 出图（sdxl / sd15 / flux 三种预设）
curl -X POST http://127.0.0.1:8188/agent/txt2img -H "Content-Type: application/json" \
  -d '{"preset":"flux","params":{"prompt":"a red fox in a snowy forest","steps":4}}'

# 拿结果（pending / done / error）
curl http://127.0.0.1:8188/agent/result/<prompt_id>

# 取图
curl "http://127.0.0.1:8188/view?filename=<filename>&subfolder=<subfolder>&type=output" -o out.png
```

配套的命令行客户端在 `client/comfy_agent.py`，只用标准库：

```bash
python client/comfy_agent.py gen --preset flux --prompt "雪地里的红狐" --out out.png
```

## 顺手修的一个坑

当前 `ComfyUI-GGUF` 不再把 `.gguf` 注册进 `folder_paths.supported_pt_extensions`，
导致 GGUF 模型在节点下拉框和 `get_filename_list()` 里**根本看不见**。本插件在启动时和
每次请求时补注册并清缓存，所以量化模型在下拉框里正常可见。

## 小显存提示

插件本身不占显存，但 ComfyUI 默认参数对小卡不友好。8GB 显存 / 16GB 内存建议：

```
python main.py --reserve-vram 0.6 --cache-none --disable-pinned-memory
```

`--disable-pinned-memory` 最关键：默认会预留数 GB 的内存缓冲，16GB 机器上会直接报
`DefaultCPUAllocator: not enough memory`。
