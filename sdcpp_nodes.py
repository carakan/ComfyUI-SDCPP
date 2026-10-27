import torch
import numpy as np
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

import re
import subprocess
import tempfile

from PIL import Image

SDCLI_FALLBACK = "/Applications/ToshLLM.app/Contents/Resources/bin-image/sd-cli"

MODEL_EXTENSIONS = (".gguf", ".safetensors", ".sft")

UNET_KEYWORDS = ("z_image", "flux", "qwen_image", "wan", "hunyuan", "sdxl", "juggernaut")
VAE_KEYWORDS = ("vae", "ae.safetensors", "ae-f16")
TEXT_ENCODER_KEYWORDS = ("qwen3-4b", "t5", "umt5", "clip", "llm")

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


def extra_model_dirs():
    raw = os.environ.get("SDCPP_MODEL_DIRS", os.path.expanduser("~/models/imagen"))
    return [d for d in raw.split(os.pathsep) if d]


def model_roots(folder_names):
    roots = []
    for name in folder_names:
        for path in folder_paths.get_folder_paths(name):
            roots.append(("comfy", path))
    for path in extra_model_dirs():
        roots.append((os.path.basename(path.rstrip("/")) or "extra", path))
    return roots


def list_models(folder_names, role_keywords):
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
                if not filename.lower().endswith(MODEL_EXTENSIONS):
                    continue
                path = os.path.join(dirpath, filename)
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
        path = os.path.realpath(os.path.join(root, rel))
        if not path.startswith(os.path.realpath(root) + os.sep):
            raise ValueError(f"model path escapes its root: {display}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"model file is gone: {path}")
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
        path = os.path.realpath(os.path.join(root, rel))
        if not path.startswith(os.path.realpath(root) + os.sep):
            raise ValueError(f"lora path escapes its root: {display}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"lora file is gone: {path}")
        return path
    raise ValueError(f"unknown lora: {display}")


class SDCPPModel:
    """A single file handed to sd.cpp: diffusion model, VAE or text encoder."""

    def __init__(self, path, display):
        self.path = path
        self.display = display


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
        return {"required": {"diffusion_model": (list_models(["diffusion_models", "unet"], UNET_KEYWORDS),)}}

    RETURN_TYPES = ("SDCPP_MODEL",)
    FUNCTION = "load"
    CATEGORY = "sampling/sdcpp"

    def load(self, diffusion_model):
        return (SDCPPModel(resolve_model(diffusion_model, ["diffusion_models", "unet"]), diffusion_model),)


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


class SDCPPSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusion_model": ("SDCPP_MODEL",),
                "vae": ("SDCPP_MODEL",),
                "text_encoder": ("SDCPP_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A highly detailed photograph"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "sampling_method": (SAMPLING_METHODS, {"default": "euler"}),
                "scheduler": (SCHEDULERS, {"default": "discrete"}),
                "flow_shift": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1}),
                "vae_tiling": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "text_encoder_2": ("SDCPP_MODEL",),
                "vision_encoder": ("SDCPP_MODEL",),
                "lora": ("SDCPP_LORA",),
                "reference_images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "sampling/sdcpp"

    def generate(self, diffusion_model, vae, text_encoder, prompt, negative_prompt,
                 width, height, steps, cfg_scale, seed, sampling_method, scheduler,
                 flow_shift, vae_tiling, text_encoder_2=None, vision_encoder=None,
                 lora=None, reference_images=None):
        binary = sdcli_path()
        if not os.path.isfile(binary):
            raise FileNotFoundError(
                f"sd-cli not found at {binary}. Set SDCPP_BIN to a stable-diffusion.cpp build."
            )

        encoders = [text_encoder]
        if text_encoder_2 is not None:
            encoders.append(text_encoder_2)

        with tempfile.TemporaryDirectory() as workdir:
            output = os.path.join(workdir, "out.png")
            cmd = [
                binary,
                "--diffusion-model", diffusion_model.path,
                "--vae", vae.path,
            ]
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
                "-s", str(seed),
                "-o", output,
            ]
            if flow_shift >= 0:
                cmd += ["--flow-shift", str(flow_shift)]
            if vae_tiling:
                cmd += ["--vae-tiling"]
            if reference_images is not None:
                for index, tensor in enumerate(reference_images):
                    array = (tensor.cpu().numpy() * 255.0).round().astype("uint8")
                    path = os.path.join(workdir, f"ref_{index}.png")
                    Image.fromarray(array).save(path)
                    cmd += ["--ref-image", path]

            progress = comfy.utils.ProgressBar(steps)
            lines = []
            step_re = re.compile(r"\|\s*(\d+)/(\d+)\s*-\s*[\d.]+s/it")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                try:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                except Exception:
                    proc.kill()
                    raise
                line = proc.stdout.readline()
                if line:
                    lines.append(line.rstrip())
                    match = step_re.search(line)
                    if match:
                        progress.update(int(match.group(1)))
                    continue
                if proc.poll() is not None:
                    break
            if proc.returncode != 0 or not os.path.isfile(output):
                raise RuntimeError("sd-cli failed:\n" + "\n".join(lines[-20:]))

            image = Image.open(output).convert("RGB")
            array = np.asarray(image).astype(np.float32) / 255.0
            return (torch.from_numpy(array)[None],)
