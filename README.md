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

**Z-Image example** (1024×1024, 8 steps, ~66 s on an RX 6800 XT via Metal):

| node | value |
|---|---|
| Diffusion Model | `z_image_turbo-Q4_0.gguf` |
| VAE | `z_image_ae-f16.gguf` |
| Text Encoder | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` |
| Sampler | `res_multistep`, `simple`, `cfg 1.0`, `flow_shift 3`, `vae_tiling on` |

Notes for Z-Image:

- `vae_tiling` **must stay on** for 1024 px: without it the VAE decode leaves
  a band of garbage across the top of the image. At ≤768 px it makes no
  difference.
- `flow_shift 3` matches ComfyUI's `ModelSamplingAuraFlow(shift=3)`.
- The diffusion model is quantized (`Q4_0`); the base `bf16` safetensors work
  too but need far more memory.

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

## 🔧 Acknowledgements
Powered by [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and [ggml](https://github.com/ggerganov/ggml).
