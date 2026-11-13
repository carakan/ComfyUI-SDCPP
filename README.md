# ComfyUI-SDCPP

A custom node for **ComfyUI** that integrates the [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) inference engine: quantized `gguf` models inside ComfyUI, driven by an `sd-cli` binary you already have working on the machine. The plugin is backend-agnostic — it never talks to the GPU itself, the engine does, so it runs on Metal, Vulkan, CUDA, SYCL or CPU, whichever backend your binary was built with.

## 🌟 Features

- **Two backends, one plugin**:
  - **sd-cli**: **`SD.cpp Sampler (sd-cli)`** drives an already-compiled
    `stable-diffusion.cpp` binary as a subprocess. No bindings, no build step,
    and it runs *your* build — the one that already works on this machine, on
    any backend.
  - **Python bindings** (upstream): **`SD.cpp Flux (UNet/VAE/CLIP)`** drives
    the `stable_diffusion_cpp` bindings in-process.
- **GGUF support in ComfyUI**: the plugin adds `.gguf` to the extensions
  ComfyUI accepts for `unet`, `clip` and `vae` folders, so GGUF files show up
  in the dropdowns next to the safetensors ones.
- **One engine process at a time**: every render spawns its own `sd-cli`, but
  a render waits for the previous one to fully exit before it launches, and
  **Cancel** stops the running engine before the prompt is aborted. A
  cancelled render can never leave a stale engine holding the GPU under the
  next queued one.
- **Flash Attention**: sd.cpp's `--flash-attn` shrinks the activation
  footprint and speeds sampling up.
- **ComfyUI integration**: step progress bar, working Cancel, and a plain
  `IMAGE` tensor out — ready for any downstream ComfyUI node.
- **Model families**: the sd-cli backend handles whatever your binary
  supports — Flux, SD1.5/SDXL, Wan, Qwen-Image 1.x/2.1, Z-Image, Hunyuan and
  friends.

## 📦 Installation

### sd-cli backend (no Python dependencies)

1. Get a working `stable-diffusion.cpp` `sd-cli` binary for your platform and
   tell the node where it is:
   ```bash
   export SDCPP_BIN=/path/to/sd-cli
   ```
   Lookup order: `$SDCPP_BIN` →
   `/Applications/ToshLLM.app/Contents/Resources/bin-image/sd-cli` → `sd-cli`
   on `$PATH`.
2. Optionally add model directories (default `~/models/imagen`), separated by
   the platform path separator (`:` on macOS/Linux, `;` on Windows):
   ```bash
   export SDCPP_MODEL_DIRS=/path/to/models:/other/models
   ```
3. Clone the plugin:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/carakan/ComfyUI-SDCPP.git
   ```
4. Boot up ComfyUI. The node registers `SD.cpp Diffusion Model (sd-cli)`,
   `SD.cpp VAE (sd-cli)`, `SD.cpp Text Encoder (sd-cli)` and
   `SD.cpp Sampler (sd-cli)`.

The env vars are read by the server process, so a change needs a ComfyUI
restart. A wrong `SDCPP_BIN` fails fast with
`FileNotFoundError: sd-cli not found at ...`.

### macOS on Intel, engine from ToshLLM

The machine this plugin was developed on is an Intel Mac with an AMD RX 6800
XT: ComfyUI runs there on a custom PyTorch MPS build, and the engine is the
Metal `sd-cli` shipped inside the ToshLLM app build — no Apple Silicon
anywhere. `ComfyUI/start_comfyui.sh` wires it up:

```bash
export SDCPP_BIN="$HOME/development/toshllm/dist/ToshLLM.app/Contents/Resources/bin-image/sd-cli"
export SDCPP_MODEL_DIRS="$HOME/models/imagen"
```

The launcher only fills a value that is not already set, so exporting these
before starting it overrides it. Things worth knowing on that setup:

- ComfyUI runs the *dist* build, not the copy installed under
  `/Applications` — that one is refreshed by hand and drifts behind. Which
  binary ran is logged as `[sd-cli] cmd:` on every render.
- On the 16 GiB card, set the sampler's `max_vram` to `8` for the big models
  (Qwen-Image 2.1) instead of the `0` = use-everything default; a full canvas
  otherwise fails weight preparation and returns flat noise.
- On macOS Intel with the ToshLLM binary, the plugin exports `TOSH_FA_AMD=1`
  to every `sd-cli` call — that build gates its AMD Metal flash-attention
  kernels behind the variable. Nothing to configure; point `SDCPP_BIN`
  elsewhere and it stays unset.

### Python bindings backend

1. Install the bindings:
   ```bash
   pip install stable-diffusion-cpp-python
   ```
   There is no macOS x86_64 wheel — on Intel Macs the bindings must be built
   from source, and the sd-cli backend above is the way to go.
2. Clone the plugin as shown above and boot ComfyUI.

The Flux node is only registered when the bindings import successfully, so the
plugin still loads cleanly on platforms where they are unavailable.

## 🚀 Usage

### sd-cli backend

1. Add **`SD.cpp Diffusion Model (sd-cli)`**, **`SD.cpp VAE (sd-cli)`** and
   **`SD.cpp Text Encoder (sd-cli)`**, wire their outputs into
   **`SD.cpp Sampler (sd-cli)`**, then into any image node (`SaveImage`,
   upscalers, …).
2. Model dropdowns list ComfyUI's own model folders plus `$SDCPP_MODEL_DIRS`.
   The text encoder is routed to the right sd.cpp flag automatically
   (`--llm` for Qwen/Mistral/Llama, `--t5xxl` for T5/UMT5, `--clip_l`
   otherwise).
3. Set prompt, resolution and steps, then **Queue Prompt**.

Each render runs its own `sd-cli` process, but only one at a time: a render
waits for the previous one to fully exit before it launches, and **Cancel**
stops the running `sd-cli` (SIGTERM first, SIGKILL if it ignores that) before
the prompt is aborted, so a cancelled render never leaves the engine holding
the GPU underneath the next queued one.

Optional sampler inputs:

| input | what it does |
|---|---|
| `text_encoder_2` | second encoder for models that need two (Qwen-Image 1.x style: T5 **and** CLIP-L) |
| `vision_encoder` | the Qwen3-VL vision tower (`mmproj`), needed for reference-image (edit) workflows |
| `lora` | a LoRA from **`SD.cpp LoRA (sd-cli)`** — see the LoRA section below |
| `reference_images` | an `IMAGE` batch; each image is passed to sd.cpp as `--ref-image` (image 1 = person, image 2 = outfit, …) |
| `reference_image_1` / `reference_image_2` / `reference_image_3` | one `IMAGE` each, passed as `--ref-image` in that order. Use these instead of the batch when the references have different dimensions: every image in a batch has to be the same size, separate sockets do not. The numbered sockets come first, then the batch. sd.cpp still wants each reference to be a multiple of 32 in both directions |
| `max_vram` | GPU budget in GiB for the model weights. `0` = use all free VRAM. Lower it (e.g. `8`) when the big models do not fit |
| `randomize_prompt` | keep one option of every `{a\|b\|c}` group in the positive prompt, picked at random — on by default, see below |

### Randomizing parts of the prompt

Write alternatives in braces in the positive prompt: `a woman wearing a
{red|blue|white} top`. Every group keeps one option, and each run picks
another one. `randomize_prompt` is **on by default** (like ComfyUI core's
dynamic prompts); switch it off to send the braces to the model literally.

- Nested groups work: `{red|{blue|white}}`.
- A fixed `seed` keeps the pick reproducible — the same seed always chooses the
  same option. `seed -1` (sd-cli picks the noise seed itself) gives a new
  option on every run.
- The expanded prompt is printed to the log (`[sd-cli] randomized prompt: …`) so
  you can see which option was used. Nothing is printed when the prompt has no
  group.
- With **ComfyUI-Impact-Pack** installed its expander is used, which also
  understands weights (`{5::red|1::blue}`) and wildcard files (`__colours__`).
  Without that pack a plain `{a|b|c}` expander is used instead.
- The prompt reaches the engine with the braces already resolved, so sd.cpp's
  own weighting syntax (`(text:1.5)`) still applies on top of it.

**Z-Image example** (896×1152, 8 steps, ~60 s measured on an Intel Mac with an
AMD RX 6800 XT, Metal backend):

| node | value |
|---|---|
| Diffusion Model | `z_image_turbo-Q4_0.gguf` |
| VAE | `z_image_ae-f16.gguf` |
| Text Encoder | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` |
| Sampler | `euler`, `discrete`, `cfg 1.0`, `flow_shift -1` (off), `vae_tiling on` |

**Qwen-Image 2.1 example** (needs a Qwen-family LLM text encoder):

| node | value |
|---|---|
| Diffusion Model | `qwen_image_2.1-Q6_K.gguf` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` |
| Text Encoder | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` |
| Sampler | `euler`, `simple`, `cfg 1.0`, `guidance 3.5` (sd.cpp default), `vae_tiling on` |

Notes:

- `vae_tiling` **must stay on**: without it the VAE decode leaves a band of
  garbage across the top of the image (clearest above 768 px).
- The text encoder is routed to the right sd.cpp flag by file name: `--llm`
  for Qwen/Mistral/Llama, `--t5xxl` for T5/UMT5, `--clip_l` otherwise. Models
  that need two encoders (Qwen-Image 1.x style: T5 **and** CLIP-L) take the
  second one in the sampler's optional `text_encoder_2` input.
- `flow_shift` matches ComfyUI's `ModelSamplingAuraFlow(shift=N)`; leave it at
  `-1` for the distilled turbo checkpoints.
- The diffusion model can be quantized (`Q4_0`, `Q6_K`); the base `bf16`
  safetensors work too but need far more memory.

### Python bindings backend

Inside ComfyUI:
1. Double click the workspace (or search the node list) and add: **`SD.cpp Flux (UNet/VAE/CLIP)`**.
2. Place your downloaded `.gguf` files into the standard ComfyUI directories:
   - `ComfyUI/models/unet/`
   - `ComfyUI/models/clip/`
   - `ComfyUI/models/vae/`
   *(Node specifically ensures any `qwen` / `mistral` / `llama` named text encoders route to the `llm_path` for correct Flux handling).*

3. Select your UNet, CLIP, and VAE.
4. Set your prompt, resolution, and steps, then click **Queue Prompt**.

## LoRA

stable-diffusion.cpp supports LoRAs the A1111 way: the weights live in
`--lora-model-dir` and the prompt carries `<lora:file_stem:strength>`. The
**`SD.cpp LoRA (sd-cli)`** node does that for you: it adds the directory and
the tag to the prompt.

**The key format must be sd.cpp's.** sd.cpp looks a tensor up as
`lora.<model tensor name>.lora_down` / `.lora_up`. ComfyUI writes
`diffusion_model.<module>.lora_A.weight` / `lora_B.weight`. A ComfyUI LoRA
therefore applies **0 of its tensors** — and with
`--lora-apply-mode immediately` the process aborts with a native backtrace.
Convert the file first:

```sh
python convert_lora_for_sdcpp.py <in.safetensors> [out.safetensors] [dtype]
```

The converter renames the keys and casts the weights to f16. f16 is the
default because it is small and every backend has a fast f16 path; bf16 LoRAs
work too on current engine builds, so pass `bf16` as the dtype to keep it.

After conversion the LoRA applies: a run with and without the same LoRA differ
in ~99 % of the pixels. LoRAs are applied at runtime, which costs roughly
double the time per step.

### Qwen-Image 2.1 outfit swap (edit workflow)

| node | value |
|---|---|
| Diffusion Model | `qwen_image_2.1-Q6_K.gguf` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` |
| Text Encoder | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` |
| Vision Encoder | `mmproj-Qwen3VL-8B-Instruct-F16.gguf` |
| LoRA | the converted `qwen-image-2.1-outfit-swap.sdcpp.safetensors`, strength 1.0 |
| Sampler | `euler`, `simple`, `cfg 1.0`, 25 steps, `max_vram 8`, `vae_tiling` on |
| Prompt | `Dress the person in image 1 in the outfit shown in image 2. Keep their face, hair, hands, pose and the background exactly the same.` |

The two reference images go into the sampler's `reference_image_1` and
`reference_image_2` sockets: image 1 is the person, image 2 is the outfit.
Use the numbered sockets rather than `ImageBatch` whenever the references are
not the same size — every image in a batch has to share one size, the
separate sockets do not.

Note that the prompt names the references in words ("image 1", "image 2")
rather than with `<image1>`/`<image2>` markers. sd.cpp does not substitute a
marker in the text with the image at that position: it writes its own marker
in front of every reference and puts the text after them. A prompt that says
`<image1>` names nothing, and the model falls back to copying whichever
reference is most prominent.

## 🔧 Acknowledgements
Powered by [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and [ggml](https://github.com/ggerganov/ggml).
