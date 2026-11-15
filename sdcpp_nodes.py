import torch
import numpy as np
import random
import folder_paths
import comfy.utils
import comfy.model_management

# --- stable-diffusion.cpp Python bindings (optional) -------------------------
# The Flux node below needs the `stable_diffusion_cpp` bindings. They are not
# published for every platform (there is no macOS x86_64 wheel), so the import
# is optional: when they are missing this module still loads and the sd-cli
# backend stays usable.
try:
    from stable_diffusion_cpp import StableDiffusion
    HAVE_BINDINGS = True
except ImportError:
    StableDiffusion = None
    HAVE_BINDINGS = False

# --- Apple Silicon Metal Fix ---
# Force GGML to look inside the stable_diffusion_cpp installation directory
# for the ggml-metal.metal shader file, otherwise it silently falls back to CPU!
import os
if HAVE_BINDINGS:
    import stable_diffusion_cpp
    metal_path = os.path.dirname(stable_diffusion_cpp.__file__)
    os.environ["GGML_METAL_PATH_RESOURCES"] = metal_path
# -------------------------------

# --- THE FIX: Tell ComfyUI to look for .gguf files ---
# We inject .gguf into the allowed extensions for these specific folders
if "unet" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["unet"][1].add(".gguf")
    folder_paths.folder_names_and_paths["unet"][1].add(".GGUF")
if "clip" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["clip"][1].add(".gguf")
    folder_paths.folder_names_and_paths["clip"][1].add(".GGUF")
if "vae" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["vae"][1].add(".gguf")
    folder_paths.folder_names_and_paths["vae"][1].add(".GGUF")

# Purge ComfyUI's internal model cache so it immediately rescans the folders for our injected .GGUF files
if hasattr(folder_paths, "filename_list_cache"):
    folder_paths.filename_list_cache.clear()
if hasattr(folder_paths, "cache_helper") and hasattr(folder_paths.cache_helper, "clear"):
    folder_paths.cache_helper.clear()
# -----------------------------------------------------

# --- prompt randomization ---------------------------------------------------
def _split_group(content):
    """Split the body of a brace group on the pipes outside nested groups."""
    parts = []
    depth = 0
    current = ""
    for char in content:
        if char == "{":
            depth += 1
            current += char
        elif char == "}":
            depth -= 1
            current += char
        elif char == "|" and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return parts


def _expand_groups(text, rng):
    """Keep one alternative of every {a|b|c} group in `text`.

    Nested groups work: {red|{blue|white}} picks red, blue or white. A group
    with a single option loses its braces, and a brace that never closes is
    left alone.
    """
    out = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "{":
            depth = 0
            end = index
            while end < length:
                if text[end] == "{":
                    depth += 1
                elif text[end] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                end += 1
            if end >= length:
                out.append(text[index:])
                break
            chosen = rng.choice(_split_group(text[index + 1:end]))
            out.append(_expand_groups(chosen, rng))
            index = end + 1
        else:
            out.append(char)
            index += 1
    return "".join(out)


def randomize_prompt_groups(text, seed=None):
    """Local brace expander, used when ComfyUI-Impact-Pack is not installed."""
    rng = random.Random(seed) if seed is not None else random.Random()
    return _expand_groups(text, rng)


_wildcard_expander = None


def expand_prompt_groups(text, seed=None):
    """Keep one option of every {a|b|c} group in `text`.

    ComfyUI-Impact-Pack ships a seeded expander that also understands weights
    ({5::red|1::blue}) and wildcard files, so it is used when that pack is
    installed. The local expander keeps this node working without it.
    """
    global _wildcard_expander
    if _wildcard_expander is None:
        try:
            # Imported here, not at module scope: custom node packs are loaded
            # in filesystem order, so Impact Pack may not have put its modules
            # directory on sys.path yet.
            from impact.wildcards import process
            _wildcard_expander = process
        except ImportError:
            _wildcard_expander = randomize_prompt_groups
    return _wildcard_expander(text, seed)
# -----------------------------------------------------------------------------


class SDCppCustomFlux:
    def __init__(self):
        self.model = None
        self.current_unet = ""
        self.current_vae = ""
        self.current_clip = ""

    @classmethod
    def INPUT_TYPES(cls):
        # Because we added .gguf above, these lists will now include your GGUF files!
        clip_list = folder_paths.get_filename_list("clip")
        
        # Sort so Qwen3 (or any Qwen) automatically bubbles up and becomes the default selection!
        clip_list = sorted(clip_list, key=lambda x: 0 if "qwen" in x.lower() else 1)
        
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("unet"), ),
                "vae_name": (folder_paths.get_filename_list("vae"), ),
                "clip_name": (clip_list, ),
                
                "prompt": ("STRING", {"multiline": True, "default": "A highly detailed portrait..."}),
                
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "n_threads": ("INT", {"default": -1, "min": -1, "max": 64, "step": 1}), 
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "sampling/sdcpp"

    def generate(self, unet_name="", vae_name="", clip_name="", prompt="", width=1024, height=1024, steps=20, cfg_scale=1.0, seed=-1, n_threads=-1):
        if not HAVE_BINDINGS:
            raise RuntimeError(
                "The stable_diffusion_cpp Python bindings are not installed. "
                "Install them with: pip install stable-diffusion-cpp-python"
            )
        
        # Resolve the actual file paths on your Mac
        unet_path = folder_paths.get_full_path("unet", unet_name)
        vae_path = folder_paths.get_full_path("vae", vae_name)
        clip_path = folder_paths.get_full_path("clip", clip_name)

        # Check if you changed models in the dropdown
        paths_changed = (
            self.current_unet != unet_path or
            self.current_vae != vae_path or
            self.current_clip != clip_path
        )

        # Load the models into memory
        if self.model is None or paths_changed:
            if self.model is not None:
                try:
                    # Free the C++ memory on Apple Silicon to prevent OOM
                    import gc
                    if hasattr(self.model, "close"):
                        self.model.close()
                    self.model = None
                    gc.collect()
                except Exception:
                    pass

            print(f"Loading Flux UNet from {unet_path}...")
            
            # Route the text encoder to the correct sd.cpp argument
            kwargs = {
                "diffusion_model_path": unet_path,
                "vae_path": vae_path,
                "n_threads": n_threads,
                # Apple Silicon Optimization: Massively reduce unified memory footprint
                "flash_attn": True,
                # Pure Metal Memory Optimization: Native zero-copy streaming from SSD mapping to avoid RAM duplication
                "enable_mmap": True,
                "offload_params_to_cpu": True
            }
            
            if "qwen" in clip_name.lower() or "mistral" in clip_name.lower() or "llama" in clip_name.lower():
                kwargs["llm_path"] = clip_path
            else:
                kwargs["t5xxl_path"] = clip_path
                
            self.model = StableDiffusion(**kwargs)
            
            # Cache the paths so it doesn't reload on every generation
            self.current_unet = unet_path
            self.current_vae = vae_path
            self.current_clip = clip_path

        print("Generating image on Apple Silicon...")
        
        # Initialize ComfyUI progress bar and interruption checks
        import time

        start_time = time.time()
        pbar = comfy.utils.ProgressBar(steps)
        def progress_callback(step: int, steps_total: int, ttime: float):
            # Check if user clicked Cancel correctly
            comfy.model_management.throw_exception_if_processing_interrupted()
            
            now = time.time()
            elapsed = now - start_time
            print(f"Step {step}/{steps_total} completed in {elapsed:.1f}s total (this step approx {ttime:.2f}s internally if tracked)")
            if hasattr(pbar, "update_absolute"):
                pbar.update_absolute(step, steps_total)
            else:
                pbar.update(1)
        
        # FIX: The method is generate_image, not txt2img
        result = self.model.generate_image(
            prompt=prompt,
            negative_prompt="", # Flux ignores negative prompts
            width=width,
            height=height,
            sample_steps=steps,
            cfg_scale=cfg_scale,
            seed=seed if seed != -1 else 42,
            progress_callback=progress_callback
        )
        
        # Final pass check if cancelled exactly as execution returned
        comfy.model_management.throw_exception_if_processing_interrupted()

        # Convert back to ComfyUI Image format
        pil_image = result[0].convert("RGB")
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).contiguous()

        return (img_tensor,)


# =============================================================================
# sd-cli backend
#
# The bindings above need a prebuilt `stable_diffusion_cpp` wheel, which does
# not exist for every platform (macOS x86_64 has none). The nodes below drive a
# compiled stable-diffusion.cpp `sd-cli` binary instead: no Python bindings, no
# build step, and it runs the exact build you already have on the machine.
#
# The binary is looked up in this order:
#   1. $SDCPP_BIN
#   2. /Applications/ToshLLM.app/Contents/Resources/bin-image/sd-cli
#   3. sd-cli on $PATH
#
# Models are searched in ComfyUI's own model folders plus $SDCPP_MODEL_DIRS
# (default: ~/models/imagen).
# =============================================================================

import json
import platform
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time

from PIL import Image

# Only one sd-cli may run at a time. Every render spawns its own process, and a
# cancel can abandon one mid-run; without this guard the next queued render
# starts a second sd-cli on top of it, and every abandoned one keeps holding
# the GPU until the machine gives up. The lock serializes renders, and the
# handle below lets a new render put down a leftover process first.
_sdcli_lock = threading.Lock()
_sdcli_process = None


def _sdcli_stop(process, grace=15.0):
    """Shut a running sd-cli down, politely first, then hard."""
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=grace)
    except OSError:
        pass

SDCLI_FALLBACK = "/Applications/ToshLLM.app/Contents/Resources/bin-image/sd-cli"

MODEL_EXTENSIONS = (".gguf", ".safetensors", ".sft")
TOKENIZER_EXTENSIONS = (".json",)

UNET_KEYWORDS = ("z_image", "flux", "qwen_image", "wan", "hunyuan", "sdxl", "juggernaut")
# Where a diffusion model may live. "checkpoints" lets an SDXL/SD1.5 checkpoint that
# ComfyUI already loads be reused for sd-cli instead of duplicating the weights.
UNET_FOLDERS = ["diffusion_models", "unet", "checkpoints"]
VAE_KEYWORDS = ("vae", "ae.safetensors", "ae-f16")
TEXT_ENCODER_KEYWORDS = ("qwen3-4b", "t5", "umt5", "clip", "llm")
TOKENIZER_KEYWORDS = ("tokenizer",)
# ComfyUI's own text encoder folder holds the CLIP vocabulary, which sd.cpp
# cannot use as a tokenizer, so only the extra model directory is searched.
TOKENIZER_FOLDERS = ["tokenizers"]

SAMPLING_METHODS = [
    "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
    "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd", "res_multistep",
    "res_2s", "er_sde", "euler_cfg_pp", "euler_a_cfg_pp", "euler_ge",
    "dpm++2m_sde", "dpm++2m_sde_bt", "lms",
]

SCHEDULERS = [
    "discrete", "karras", "exponential", "ays", "gits", "sgm_uniform", "simple",
    "smoothstep", "kl_optimal", "lcm", "bong_tangent", "ltx2", "logit_normal",
    "flux2", "flux", "beta", "llada_image",
]


def sdcli_path():
    configured = os.environ.get("SDCPP_BIN")
    if configured:
        return configured
    if os.path.isfile(SDCLI_FALLBACK):
        return SDCLI_FALLBACK
    return "sd-cli"

def sdcli_env():
    """Env for the sd-cli subprocess.

    The ToshLLM engine build on macOS Intel (AMD over Metal):
    - TOSH_FA_AMD=1 routes flash attention to the _amd kernels - essential
      when --flash-attn is on.
    - TOSH_MM_WIDE_DISABLE=1 turns off the wide matmul tile; it is LLM-tuned
      and costs on diffusion.
    - GGML_METAL_NCB=3 splits Metal command buffers so big canvases stay
      under the GPU watchdog.
    """
    env = dict(os.environ)
    if (sys.platform == "darwin" and platform.machine() == "x86_64"
            and "toshllm" in sdcli_path().lower()):
        env["TOSH_FA_AMD"] = "1"
        env["TOSH_MM_WIDE_DISABLE"] = "1"
        env["GGML_METAL_NCB"] = "3"
    return env


def extra_model_dirs():
    raw = os.environ.get("SDCPP_MODEL_DIRS", os.path.expanduser("~/models/imagen"))
    return [d for d in raw.split(os.pathsep) if d]


def model_roots(folder_names):
    roots = []
    for name in folder_names:
        try:
            paths = folder_paths.get_folder_paths(name)
        except KeyError:
            # ComfyUI does not know every folder the node looks in, e.g. the
            # tokenizer folder that only exists in the extra model directory.
            continue
        for path in paths:
            roots.append(("comfy", path))
    for path in extra_model_dirs():
        roots.append((os.path.basename(path.rstrip("/")) or "extra", path))
    return roots


def list_models(folder_names, role_keywords, extensions=MODEL_EXTENSIONS):
    """Every model file the node can load.

    The extra model directory is flat and mixes roles, so entries matching the
    role come first: the combo default is the first item.
    """
    seen = set()
    names = []
    for label, root in model_roots(folder_names):
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.lower().endswith(extensions):
                    continue
                path = os.path.join(dirpath, filename)
                if not os.path.isfile(path):
                    continue
                rel = os.path.relpath(path, root)
                display = f"{label}/{rel}"
                if display in seen:
                    continue
                seen.add(display)
                names.append(display)
    names.sort(key=lambda name: (0 if any(tag in name.lower() for tag in role_keywords) else 1, name))
    return names


def resolve_model(display, folder_names):
    """Map a display name back to a path, refusing anything outside the roots."""
    label, _, rel = display.partition("/")
    for root_label, root in model_roots(folder_names):
        if root_label != label:
            continue
        candidate = os.path.join(root, rel)
        if not os.path.abspath(candidate).startswith(os.path.abspath(root) + os.sep):
            raise ValueError(f"model path escapes its root: {display}")
        # The model may be a symlink into the central store, so the real path is
        # allowed to live outside the root. Only the link itself has to sit
        # inside it, which keeps ".." out.
        path = os.path.realpath(candidate)
        if os.path.isfile(path):
            return path
    raise ValueError(f"unknown model: {display}")


def text_encoder_flag(display):
    """sd.cpp takes the text encoder under different flags per family."""
    name = os.path.basename(display).lower()
    if any(tag in name for tag in ("qwen", "mistral", "llama")):
        return "--llm"
    if any(tag in name for tag in ("t5", "umt5")):
        return "--t5xxl"
    return "--clip_l"


def is_full_checkpoint(path):
    """True when a safetensors file bundles its text encoders with the unet.

    SDXL/SD1.5 checkpoints store them under ``conditioner.`` or
    ``cond_stage_model.``; a split Comfy-Org unet does not. sd.cpp needs
    --model for the former and --diffusion-model for the latter.
    """
    if not path.lower().endswith(".safetensors"):
        return False
    try:
        with open(path, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            if size > 64 * 1024 * 1024:
                return False
            header = json.loads(handle.read(size))
    except (OSError, ValueError, struct.error):
        return False
    return any(key.startswith(("conditioner.", "cond_stage_model.")) for key in header)


LORA_EXTENSIONS = (".safetensors", ".gguf")


def extra_lora_dirs():
    raw = os.environ.get("SDCPP_LORA_DIRS", os.path.expanduser("~/Downloads"))
    return [d for d in raw.split(os.pathsep) if d]


def lora_roots():
    roots = []
    for path in folder_paths.get_folder_paths("loras"):
        roots.append(("comfy", path))
    for path in extra_lora_dirs():
        roots.append((os.path.basename(path.rstrip("/")) or "extra", path))
    return roots


def list_loras():
    seen = set()
    names = []
    for label, root in lora_roots():
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.lower().endswith(LORA_EXTENSIONS):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, filename), root)
                display = f"{label}/{rel}"
                if display in seen:
                    continue
                seen.add(display)
                names.append(display)
    return sorted(names)


def resolve_lora(display):
    label, _, rel = display.partition("/")
    for root_label, root in lora_roots():
        if root_label != label:
            continue
        candidate = os.path.join(root, rel)
        if not os.path.abspath(candidate).startswith(os.path.abspath(root) + os.sep):
            raise ValueError(f"lora path escapes its root: {display}")
        # Like the model loaders, allow a symlink whose target lives in the
        # central store: only the link itself has to sit inside the root.
        path = os.path.realpath(candidate)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"lora file is gone: {path}")
        return path
    raise ValueError(f"unknown lora: {display}")


class SDCPPModel:
    """A single file handed to sd.cpp: diffusion model, VAE or text encoder.

    ``full_checkpoint`` marks an SDXL/SD1.5 checkpoint that also carries its
    text encoders and VAE; sd.cpp loads those with --model instead of
    --diffusion-model plus separate --clip_l/--vae files.
    """

    def __init__(self, path, display, full_checkpoint=False):
        self.path = path
        self.display = display
        self.full_checkpoint = full_checkpoint


class SDCPPLora:
    """A LoRA file plus its strength.

    sd.cpp takes LoRAs the A1111 way: the file lives in --lora-model-dir and
    the prompt carries `<lora:file_stem:strength>`.
    """

    def __init__(self, path, strength):
        self.path = path
        self.strength = strength
        self.stem = os.path.splitext(os.path.basename(path))[0]


class SDCPPLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lora": (list_loras(),),
            "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("SDCPP_LORA",)
    FUNCTION = "load"
    CATEGORY = "sampling/sdcpp"

    def load(self, lora, strength):
        return (SDCPPLora(resolve_lora(lora), strength),)


class SDCPPUnetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # "checkpoints" is included so the SDXL/SD1.5 checkpoints the rest of
        # ComfyUI already uses can be handed to sd-cli directly. sd.cpp reads a
        # full SDXL checkpoint as its diffusion model, so there is no reason to
        # ask for a second copy of the weights.
        return {"required": {"diffusion_model": (list_models(UNET_FOLDERS, UNET_KEYWORDS),)}}

    RETURN_TYPES = ("SDCPP_MODEL",)
    FUNCTION = "load"
    CATEGORY = "sampling/sdcpp"

    def load(self, diffusion_model):
        path = resolve_model(diffusion_model, UNET_FOLDERS)
        return (SDCPPModel(path, diffusion_model, full_checkpoint=is_full_checkpoint(path)),)


class SDCPPVaeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": (list_models(["vae"], VAE_KEYWORDS),)}}

    RETURN_TYPES = ("SDCPP_MODEL",)
    FUNCTION = "load"
    CATEGORY = "sampling/sdcpp"

    def load(self, vae):
        return (SDCPPModel(resolve_model(vae, ["vae"]), vae),)


class SDCPPTextEncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text_encoder": (list_models(["text_encoders", "clip"], TEXT_ENCODER_KEYWORDS),)}}

    RETURN_TYPES = ("SDCPP_MODEL",)
    FUNCTION = "load"
    CATEGORY = "sampling/sdcpp"

    def load(self, text_encoder):
        return (SDCPPModel(resolve_model(text_encoder, ["text_encoders", "clip"]), text_encoder),)


class SDCPPTokenizerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # sd.cpp wants a tokenizer.json, not the CLIP vocabulary files that
        # happen to sit in the same folders.
        candidates = list_models(TOKENIZER_FOLDERS, TOKENIZER_KEYWORDS,
                                 extensions=TOKENIZER_EXTENSIONS)
        tokenizers = [name for name in candidates
                      if os.path.basename(name).lower().startswith("tokenizer")
                      and not any(part.startswith("clip")
                                  for part in name.lower().split("/"))]
        return {"required": {"tokenizer": (tokenizers,)}}

    RETURN_TYPES = ("SDCPP_MODEL",)
    FUNCTION = "load"
    CATEGORY = "sampling/sdcpp"

    def load(self, tokenizer):
        return (SDCPPModel(resolve_model(tokenizer, TOKENIZER_FOLDERS), tokenizer),)


class SDCPPSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusion_model": ("SDCPP_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A highly detailed photograph"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                # sd-cli parses the seed with std::stoll, which throws on anything
                # above 2**63-1, so the widget stops there rather than at the
                # full unsigned 64-bit range.
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7fffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise. "
                                            "-1 lets sd-cli pick one."}),
                "sampling_method": (SAMPLING_METHODS, {"default": "euler"}),
                "scheduler": (SCHEDULERS, {"default": "discrete"}),
                "flow_shift": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1}),
                "vae_tiling": ("BOOLEAN", {"default": True}),
                "max_vram": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 1.0,
                                       "tooltip": "GPU budget in GiB for model weights. 0 lets sd.cpp "
                                                  "use all free VRAM; lower it when the big models do not "
                                                  "fit (Qwen-Image 2.1 needs ~14 GiB on this GPU)."}),
                "offload_to_cpu": ("BOOLEAN", {"default": False,
                                               "tooltip": "Keep the model weights in RAM and stream them "
                                                          "to the GPU. Slower, but the 12 GB models fit "
                                                          "on a GPU shared with everything else."}),
            },
            "optional": {
                # Both are left out when a full checkpoint carries them: sd.cpp
                # reads those straight from the file passed with --model.
                "vae": ("SDCPP_MODEL",),
                "text_encoder": ("SDCPP_MODEL",),
                "text_encoder_2": ("SDCPP_MODEL",),
                "tokenizer": ("SDCPP_MODEL",),
                "vision_encoder": ("SDCPP_MODEL",),
                "lora": ("SDCPP_LORA",),
                "reference_image_1": ("IMAGE", {"tooltip": "First reference image. "
                                                            "sd.cpp takes one --ref-image per image, in order."}),
                "reference_image_2": ("IMAGE", {"tooltip": "Second reference image."}),
                "reference_image_3": ("IMAGE", {"tooltip": "Third reference image."}),
                "input_image": ("IMAGE", {"tooltip": "Denoise starting from this image instead of "
                                                     "noise (img2img); strength below 1 keeps more of it."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "flash_attention": ("BOOLEAN", {"default": False,
                                                "tooltip": "sd.cpp's flash attention: smaller activations "
                                                           "and faster sampling on Metal."}),
                "sigmas": ("STRING", {"default": "",
                                      "tooltip": "Explicit sigma schedule for distilled models, e.g. the "
                                                 "Qwen-Image Turbo list. Empty uses the scheduler's own."}),
                "model_args": ("STRING", {"default": "",
                                          "tooltip": "Extra model args passed to sd.cpp as --model-args, "
                                                     "e.g. qwen_image_2_1_prefix_cache_type=q8_0 to shrink "
                                                     "the Qwen-Image 2.1 prefix cache on small GPUs."}),
                "randomize_prompt": ("BOOLEAN", {"default": True,
                                                "tooltip": "Keep one option of every {a|b|c} group in the "
                                                           "positive prompt, picked at random. A fixed seed "
                                                           "keeps the pick reproducible, seed -1 gives a new "
                                                           "one on every run. Turn it off to send the braces "
                                                           "to the model literally."}),
                # New widgets must be appended at the END of this dict: saved
                # workflows map widget values positionally, so inserting one
                # earlier shifts every later value into the wrong widget.
                "text_encoder_cpu": ("BOOLEAN", {"default": True,
                                                 "tooltip": "Run the text encoder on the CPU "
                                                            "(--params-backend te=cpu): it never sits in "
                                                            "VRAM, leaving more of the card to the "
                                                            "diffusion model and LoRAs. Slower prompt "
                                                            "encoding."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "sampling/sdcpp"

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        """A random seed must not be served from the cache.

        ComfyUI keys its cache on the node's inputs, so a prompt that always
        carries the same seed -1 would answer every run with the previous
        image even though sd-cli picks a new one. Returning NaN makes the
        signature differ every time, so the node runs again.
        """
        if seed < 0:
            return float("NaN")
        return seed

    def generate(self, diffusion_model, prompt, negative_prompt,
                 width, height, steps, cfg_scale, seed, sampling_method, scheduler,
                 flow_shift, vae_tiling, vae=None, text_encoder=None, max_vram=0.0,
                 offload_to_cpu=False, text_encoder_2=None, tokenizer=None, vision_encoder=None, lora=None,
                 input_image=None, strength=1.0, flash_attention=False, text_encoder_cpu=True,
                 reference_image_1=None, reference_image_2=None, reference_image_3=None,
                 sigmas="", model_args="", randomize_prompt=True):
        if randomize_prompt:
            # A fixed seed keeps the choice reproducible, seed -1 (sd-cli picks
            # the noise seed itself) gets a new one on every run.
            expanded = expand_prompt_groups(prompt, seed if seed >= 0 else None)
            if expanded != prompt:
                print(f"[sd-cli] randomized prompt: {expanded}")
                prompt = expanded

        binary = sdcli_path()
        if not os.path.isfile(binary):
            raise FileNotFoundError(
                f"sd-cli not found at {binary}. Set SDCPP_BIN to a stable-diffusion.cpp build."
            )

        encoders = [encoder for encoder in (text_encoder, text_encoder_2) if encoder is not None]

        # sd.cpp takes over the GPU for the whole run. Hand it a clean slate:
        # torch's caching allocator keeps freed blocks reserved, which on this
        # 16 GiB card is the difference between the big models fitting and not.
        try:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache(force=True)
        except Exception as error:
            print(f"[sd-cli] could not release the torch GPU cache: {error}")

        with tempfile.TemporaryDirectory() as workdir:
            output = os.path.join(workdir, "out.png")
            cmd = [binary]
            if diffusion_model.full_checkpoint:
                # The file also holds the text encoders and the VAE, so sd.cpp
                # loads it whole instead of taking them from separate files.
                cmd += ["--model", diffusion_model.path]
            else:
                cmd += ["--diffusion-model", diffusion_model.path]
            if vae is not None:
                cmd += ["--vae", vae.path]
            if tokenizer is not None:
                # Qwen3.5 and friends ship their own vocabulary, so the
                # tokenizer.json has to travel with the text encoder.
                cmd += ["--tokenizer", tokenizer.path]
            for encoder in encoders:
                cmd += [text_encoder_flag(encoder.display), encoder.path]
            if vision_encoder is not None:
                cmd += ["--llm_vision", vision_encoder.path]
            if lora is not None:
                cmd += ["--lora-model-dir", os.path.dirname(lora.path)]
                if "<lora:" not in prompt:
                    prompt = f"<lora:{lora.stem}:{lora.strength}> {prompt}"
            cmd += [
                "-p", prompt,
                "-n", negative_prompt,
                "-W", str(width), "-H", str(height),
                "--steps", str(steps),
                "--cfg-scale", str(cfg_scale),
                "--sampling-method", sampling_method,
                "--scheduler", scheduler,
                "-s", str(max(-1, min(seed, 0x7fffffffffffffff))),
                "-o", output,
            ]
            if flow_shift >= 0:
                cmd += ["--flow-shift", str(flow_shift)]
            if sigmas.strip():
                # Distilled schedules (Qwen-Image Turbo) are lists of sigmas, not
                # a scheduler name; sd.cpp takes them verbatim.
                cmd += ["--sigmas", sigmas.strip()]
            if model_args.strip():
                cmd += ["--model-args", model_args.strip()]
            if vae_tiling:
                cmd += ["--vae-tiling"]
            if max_vram > 0:
                cmd += ["--max-vram", str(max_vram)]
            if offload_to_cpu:
                cmd += ["--offload-to-cpu"]
            if flash_attention:
                cmd += ["--flash-attn", "off,diffusion=on"]
            if text_encoder_cpu:
                # Encode on the CPU: the encoder never sits in VRAM, leaving
                # more of the card to the diffusion model and LoRAs.
                cmd += ["--params-backend", "te=cpu"]
            if input_image is not None:
                array = (input_image[0].cpu().numpy() * 255.0).round().astype("uint8")
                input_path = os.path.join(workdir, "input.png")
                Image.fromarray(array).save(input_path)
                cmd += ["-i", input_path, "--strength", str(strength)]
            if any(image is not None for image in
                   (reference_image_1, reference_image_2, reference_image_3)):
                # Separate tensors, so the numbered references do not have to
                # share their dimensions - a portrait and a landscape reference
                # can go in side by side. sd.cpp still wants each of them to be
                # a multiple of 32 in both directions.
                groups = [image for image in (reference_image_1, reference_image_2, reference_image_3)
                          if image is not None]
                index = 0
                for group in groups:
                    for tensor in group:
                        array = (tensor.cpu().numpy() * 255.0).round().astype("uint8")
                        path = os.path.join(workdir, f"ref_{index}.png")
                        Image.fromarray(array).save(path)
                        cmd += ["--ref-image", path]
                        index += 1

            progress = comfy.utils.ProgressBar(steps)
            lines = []
            print(f"[sd-cli] cmd: {' '.join(cmd)}")
            step_re = re.compile(r"\|\s*(\d+)/(\d+)\s*-\s*[\d.]+s/it")

            global _sdcli_process
            # One render at a time, for the whole life of the process. ComfyUI
            # aborts a cancelled prompt and immediately starts the next queued
            # one; without the lock that next render launches a second sd-cli
            # on top of the abandoned one.
            with _sdcli_lock:
                # A cancel that arrived while the previous render was winding
                # down must not launch a fresh one.
                comfy.model_management.throw_exception_if_processing_interrupted()
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        env=sdcli_env())
                _sdcli_process = proc
                try:
                    while True:
                        # InterruptProcessingException inherits from BaseException,
                        # so it escapes `except Exception` guards - the finally
                        # below is the only thing that stops sd-cli on a cancel.
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        line = proc.stdout.readline()
                        if line:
                            lines.append(line.rstrip())
                            match = step_re.search(line)
                            if match:
                                progress.update(int(match.group(1)))
                            elif any(tag in line for tag in ("[I]", "[W]", "[E]", "[INFO", "[WARN", "[ERROR")):
                                print(f"[sd-cli] {line.rstrip()}")
                            continue
                        if proc.poll() is not None:
                            break
                        time.sleep(0.05)
                finally:
                    # Cancel, failure, executor shutdown: sd-cli never outlives
                    # its render. terminate first so Metal can clean up, kill if
                    # it ignores that.
                    _sdcli_stop(proc)
                    _sdcli_process = None
                if proc.returncode != 0 or not os.path.isfile(output):
                    raise RuntimeError("sd-cli failed:\n" + "\n".join(lines[-20:]))

            image = Image.open(output).convert("RGB")
            array = np.asarray(image).astype(np.float32) / 255.0
            return (torch.from_numpy(array)[None],)
