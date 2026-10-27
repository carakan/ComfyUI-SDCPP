# ComfyUI-SDCPP

A custom node for **ComfyUI** that integrates the powerful [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) inference engine. This node brings high-performance, `gguf` quantized model support to ComfyUI, with massive optimization specifically for Apple Silicon (Metal) devices to efficiently run huge models like Flux!

## 🌟 Features

- **Two backends, one plugin**:
  - **Python bindings** (upstream): `SD.cpp Flux (UNet/VAE/CLIP)` drives the
    `stable_diffusion_cpp` bindings in-process.
  - **sd-cli** (added in this fork): `SD.cpp Sampler (sd-cli)` drives an
    already-compiled `stable-diffusion.cpp` binary as a subprocess. No
    bindings, no build step, and it runs *your* build — the one you already
    have working on the machine.
- **GGUF Support In ComfyUI**: Automatically patches ComfyUI's model loaders to recognize `.gguf` extensions. You can effortlessly load `unet`, `vae`, and `clip` files in GGUF format!
- **Apple Silicon / Metal Optimization**: 
  - Forces GGML to correctly locate the `ggml-metal.metal` shader files, preventing silent CPU fallbacks.
  - Implements `enable_mmap` and `offload_params_to_cpu` for native zero-copy streaming directly from ssd to Unified Memory. Massively reduces swapping out RAM, preventing OOMs (Out Of Memory) entirely.
- **Flash Attention**: Greatly speeds up generation and reduces VRAM/RAM load footprint.
- **ComfyUI Compatibility**: Fully integrates with ComfyUI's step progress bar, interruption triggers (Cancel button), and robustly translates the output directly into perfectly contiguous standard `IMAGE` tensors, seamlessly compatible with downstream ComfyUI PyTorch nodes.
- **Model families beyond Flux**: the sd-cli backend handles whatever your
  binary supports — Flux, SD1.5/SDXL, Wan, Qwen-Image and **Z-Image**.

## 📦 Installation

### sd-cli backend (no Python dependencies)

1. Make sure a `stable-diffusion.cpp` `sd-cli` binary exists and works on
   your machine.
2. Tell the node where it is (optional — the defaults are tried first):
   ```bash
   export SDCPP_BIN=/path/to/sd-cli
   ```
   Lookup order: `$SDCPP_BIN` →
   `/Applications/ToshLLM.app/Contents/Resources/bin-image/sd-cli` → `sd-cli`
   on `$PATH`.
3. Optionally add a directory of extra models (default `~/models/imagen`):
   ```bash
   export SDCPP_MODEL_DIRS=/path/to/models:/other/models
   ```
4. Clone the plugin:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/carakan/ComfyUI-SDCPP.git
   ```
5. Boot up ComfyUI. The node registers `SD.cpp Diffusion Model (sd-cli)`,
   `SD.cpp VAE (sd-cli)`, `SD.cpp Text Encoder (sd-cli)` and
   `SD.cpp Sampler (sd-cli)`.

### Python bindings backend (Apple Silicon)

1. Ensure your Python environment has the `stable_diffusion_cpp` bindings installed:
   ```bash
   pip install stable-diffusion-cpp-python
   ```
   There is **no macOS x86_64 wheel** — on that platform the bindings have to
   be built from source, and the sd-cli backend above is the way to go.
2. Clone the plugin as shown above.
3. Boot up ComfyUI.

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

Optional sampler inputs:

| input | what it does |
|---|---|
| `text_encoder_2` | second encoder for models that need two (Qwen-Image 1.x style: T5 **and** CLIP-L) |
| `vision_encoder` | the Qwen3-VL vision tower (`mmproj`), needed for reference-image (edit) workflows |
| `lora` | a LoRA from **`SD.cpp LoRA (sd-cli)`** — see the LoRA section below |
| `reference_images` | an `IMAGE` batch; each image is passed to sd.cpp as `--ref-image` (image 1 = person, image 2 = outfit, …) |
| `max_vram` | GPU budget in GiB for the model weights. `0` = use all free VRAM. Lower it (e.g. `8`) when the big models do not fit — Qwen-Image 2.1 wants ~14 GiB and this GPU has 16 GiB shared with everything else |

**Z-Image example** (896×1152, 8 steps, ~60 s on an RX 6800 XT via Metal):

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

Two things bite, and both are silent or fatal rather than obvious:

1. **The key format must be sd.cpp's.** sd.cpp looks a tensor up as
   `lora.<model tensor name>.lora_down` / `.lora_up`. ComfyUI writes
   `diffusion_model.<module>.lora_A.weight` / `lora_B.weight`. A ComfyUI
   LoRA therefore applies **0 of its tensors** — and with
   `--lora-apply-mode immediately` the process aborts with a native
   backtrace. Convert the file first:

   ```sh
   python convert_lora_for_sdcpp.py <in.safetensors> [out.safetensors] [dtype]
   ```

   The converter renames the keys and casts the weights to f16.

2. **bf16 LoRAs crash the Metal backend.** The precompiled `default.metallib`
   has no bf16 matmul kernel, so a bf16 LoRA dies with
   `Function kernel_mul_mm_bf16_f16 was not found in the library`. Cast to f16
   (the converter does that by default).

With both fixed the LoRA applies: a run with and without the same LoRA differ
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

The two reference images come from two `LoadImage` nodes joined with
`ImageBatch` into the sampler's `reference_images` input: image 1 is the
person, image 2 is the outfit.

## 🔧 Acknowledgements
Powered by [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and [ggml](https://github.com/ggerganov/ggml).
