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
import io
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

def sdcli_env(binary=None):
    """Env for the sd-cli / sd-server subprocess.

    The ToshLLM engine build on macOS Intel (AMD over Metal):
    - TOSH_FA_AMD=1 routes flash attention to the _amd kernels - essential
      when --flash-attn is on.
    - TOSH_MM_WIDE_DISABLE=1 turns off the wide matmul tile; it is LLM-tuned
      and costs on diffusion.
    - GGML_METAL_NCB=3 splits Metal command buffers so big canvases stay
      under the GPU watchdog.

    The knobs apply to the ToshLLM build of *either* binary. Callers pass the
    binary that is being launched, otherwise the check keys off the CLI path.
    """
    env = dict(os.environ)
    binary = binary or sdcli_path()
    if (sys.platform == "darwin" and platform.machine() == "x86_64"
            and "toshllm" in binary.lower()):
        env["TOSH_FA_AMD"] = "1"
        env["TOSH_MM_WIDE_DISABLE"] = "1"
        env["GGML_METAL_NCB"] = "3"
    return env


# =============================================================================
# sd-server backend
#
# The spawn path above pays for a fresh process, a fresh Metal init and a
# fresh weight load on every render. The server backend keeps one sd-server
# process alive and renders through its native async API, so the model stays
# resident and the second render of a workflow starts sampling right away.
#
# The mode is a process environment switch, read once at import:
#
#   SDCPP_MODE=standalone   spawn sd-cli per render (default, unchanged)
#   SDCPP_MODE=server       keep one sd-server alive and render through it
#
# Other knobs (all optional):
#   SDCPP_SERVER_BIN        path to sd-server (default: next to $SDCPP_BIN)
#   SDCPP_SERVER_HOST       default 127.0.0.1
#   SDCPP_SERVER_PORT       default 12727 (upstream's own default is 1234)
#   SDCPP_SERVER_IDLE_MINUTES  unload the model after this many idle minutes
#                            (default 10; 0 disables the idle unload)
#   SDCPP_SERVER_CONDITIONING_CACHE  prompt cache size (default 8)
#
# A resident server holds VRAM, so the model is unloaded once the machine goes
# idle: the process stays up (no restart cost) but the card is free again.
#
# The plugin never depends on the server: if it cannot be started, or a
# request fails, the render falls back to the spawn path.
# =============================================================================

import base64
import urllib.error
import atexit
import signal
import socket
import urllib.request

SDSERVER_FALLBACK = "/Applications/ToshLLM.app/Contents/Resources/bin-image/sd-server"

_server = None


def server_mode_enabled():
    """True when SDCPP_MODE asks for the long-running server path."""
    return os.environ.get("SDCPP_MODE", "standalone").strip().lower() == "server"


def sdserver_path():
    configured = os.environ.get("SDCPP_SERVER_BIN")
    if configured:
        return configured
    sibling = os.path.join(os.path.dirname(sdcli_path()), "sd-server")
    if os.path.isfile(sibling):
        return sibling
    if os.path.isfile(SDSERVER_FALLBACK):
        return SDSERVER_FALLBACK
    return "sd-server"


def server_endpoint():
    host = os.environ.get("SDCPP_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("SDCPP_SERVER_PORT", "12727"))
    return host, port


def server_idle_minutes():
    try:
        return float(os.environ.get("SDCPP_SERVER_IDLE_MINUTES", "10"))
    except ValueError:
        return 10.0


def _b64_png(tensor):
    """One ComfyUI image tensor ([H, W, C] float 0..1) as a base64 PNG string.

    Server APIs accept images as base64 (or a data URL); they do not read
    file paths. The old request builder saved PNGs to a tempdir and sent the
    path, which sd-server always rejected.
    """
    array = (tensor.detach().cpu().numpy() * 255.0).round().astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_images(tensors):
    return [_b64_png(t) for t in tensors]


class SDServer:
    """One sd-server process, driven through its native async API.

    Lifecycle
    ---------
    - `ensure` reuses the running server only when both the process is alive
      AND the currently loaded model matches the requested context. The idler
      unloads (but never kills) the model, so the context key must include
      "currently loaded" rather than "started with".
    - Idle watch unloads the model via the API after N minutes with no active
      render; the process stays up, keeping the cheaper warm-resident startup.
    """

    def __init__(self):
        self.process = None
        self.context_args = []
        self.idle_thread = None
        self.log_handle = None
        self.log_path = None
        self.last_used = 0.0
        # A single process-wide lock is still needed to avoid spawning two
        # sd-servers racing on the same GPU, but it should only be held
        # during launch/teardown, not across a whole generation.
        self.lock = threading.RLock()
        self.generation_lock = threading.Lock()
        self.in_flight = 0
        self.model_loaded = False

    # -- process -----------------------------------------------------------
    def alive(self):
        return self.process is not None and self.process.poll() is None

    def _log(self, message):
        print(f"[sd-server] {message}")

    def _http(self, method, path, body=None, timeout=30):
        host, port = server_endpoint()
        url = f"http://{host}:{port}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                return response.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as error:
            raw = error.read() if error.fp else b""
            try:
                payload = json.loads(raw) if raw else {}
            except Exception:
                payload = {"error": raw.decode("utf-8", errors="replace")}
            return error.code, payload

    def _status(self):
        """Return the /models/status payload, or None when the server is away."""
        try:
            status, payload = self._http("GET", "/sdcpp/v1/models/status", timeout=5)
        except Exception:
            # The socket is not listening yet (or already gone). The caller
            # decides what to do with a server that is not there.
            return None
        return payload if status == 200 else None

    def stop(self):
        with self.lock:
            if self.process is None:
                return
            self._log("stopping the server")
            _sdcli_stop(self.process)
            self.process = None
            self.model_loaded = False
            self.context_args = []
            if getattr(self, "log_handle", None) is not None:
                try:
                    self.log_handle.close()
                except OSError:
                    pass
                self.log_handle = None

    def _wait_ready(self, timeout=600):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    "sd-server exited while loading the model; see "
                    + (self.log_path or "the server log file")
                )
            status = self._status()
            if status:
                if status.get("state") == "ready":
                    self._log("model loaded")
                    self.model_loaded = True
                    self.context_args = list(self._pending_args)
                    self.last_used = time.time()
                    return
                if status.get("state") == "error" or status.get("error"):
                    raise RuntimeError(f"the server could not load the model: {status.get('error')}")
            time.sleep(1.0)
        raise RuntimeError("sd-server did not load the model in time")

    def _port_is_free(self, host, port):
        """True when nothing listens on the port, so a new server owns it.

        sd-server shares a port with SO_REUSEPORT: several servers on the same
        port answer each other's requests with their own model.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                return False
        return True

    def _start(self, context_args, env):
        binary = sdserver_path()
        if not os.path.isfile(binary):
            raise FileNotFoundError(
                f"sd-server not found at {binary}. Set SDCPP_SERVER_BIN to a stable-diffusion.cpp server build."
            )
        host, port = server_endpoint()
        if not self._port_is_free(host, port):
            raise RuntimeError(
                f"port {port} is taken by another process. A second sd-server on the same "
                "port answers with the wrong model, so the server path is skipped."
            )
        cmd = [
            binary,
            "--listen-ip", host,
            "--listen-port", str(port),
            "--log-level", os.environ.get("SDCPP_SERVER_LOG_LEVEL", "warn"),
            "--conditioning-cache-size", os.environ.get("SDCPP_SERVER_CONDITIONING_CACHE", "8"),
        ]
        # Let sd-server resolve LoRA stems from its own model roots.
        for path in sorted({p for p in os.environ.get("SDCPP_MODEL_DIRS", os.path.expanduser("~/models/imagen")).split(os.pathsep) if p}):
            cmd += ["--model-dir", path]
        cmd += context_args
        self._log(f"starting: {' '.join(cmd)}")
        self.log_path = os.path.join(tempfile.gettempdir(), "sdcpp-sd-server.log")
        self.log_handle = open(self.log_path, "w")
        self.process = subprocess.Popen(cmd, stdout=self.log_handle, stderr=subprocess.STDOUT,
                                        text=True, env=env)
        self._pending_args = list(context_args)
        self._start_idle_watch()
        self._wait_ready()

    def ensure(self, context_args, env):
        """Reuse the running server only when the loaded model matches."""
        with self.lock:
            if self.alive() and self.model_loaded and self.context_args == list(context_args):
                # Cheap re-confirmation: the idler may have unloaded between renders.
                status = self._status()
                if status and status.get("state") in ("ready", "loading"):
                    if status.get("state") == "ready":
                        self.last_used = time.time()
                        return True
            if self.alive() and self.model_loaded:
                status = self._status()
                if status and status.get("state") == "empty":
                    # Idler already unloaded the old context; nothing to stop.
                    self.stop()
                else:
                    self.stop()
            self._start(context_args, env)
            self._start_idle_watch()
            return True

    def _start_idle_watch(self):
        if self.idle_thread is not None:
            return

        def watch():
            while True:
                time.sleep(30)
                try:
                    minutes = server_idle_minutes()
                    if minutes <= 0 or not self.alive():
                        continue
                    with self.lock:
                        if self.in_flight > 0:
                            continue
                        if time.time() - self.last_used < minutes * 60:
                            continue
                        if self._status() and self._status().get("state") == "empty":
                            continue
                        self._log("idle: unloading the model to free VRAM")
                        status, payload = self._http("POST", "/sdcpp/v1/models/unload", {}, timeout=30)
                        if status == 409 or (status == 503 and payload.get("code") == "model_busy"):
                            # A job is still pending; try again next tick.
                            self.last_used = time.time()
                            continue
                        if status == 200:
                            self.model_loaded = False
                            self._log("idle unload done")
                except Exception as error:
                    self._log(f"idle watch error: {error}")

        self.idle_thread = threading.Thread(target=watch, daemon=True)
        self.idle_thread.start()

    # -- render ------------------------------------------------------------
    def render(self, body, steps, timeout=3600):
        """Submit a generation and wait for it. Cancellations cancel the job."""
        status, payload = self._http("POST", "/sdcpp/v1/img_gen", body, timeout=60)
        if status not in (200, 202):
            message = payload.get("error", payload) if isinstance(payload, dict) else payload
            raise RuntimeError(f"sd-server refused the request ({status}): {message}")
        job_id = payload.get("id")
        if not job_id:
            raise RuntimeError(f"sd-server returned no job id: {payload}")

        progress = comfy.utils.ProgressBar(steps)
        deadline = time.time() + timeout
        self.in_flight += 1
        self.last_used = time.time()
        try:
            while time.time() < deadline:
                try:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                except Exception:
                    try:
                        self._http("POST", f"/sdcpp/v1/jobs/{job_id}/cancel", {}, timeout=10)
                    except Exception:
                        pass
                    raise

                status, job = self._http("GET", f"/sdcpp/v1/jobs/{job_id}", timeout=30)
                if status != 200:
                    # The server does not know this job. That happens when a
                    # second sd-server owns the port and answers with its own
                    # job list: waiting for the job would never end.
                    raise RuntimeError(f"sd-server does not know job {job_id} (HTTP {status})")
                state = job.get("status")
                job_progress = job.get("progress") or {}
                total = job_progress.get("total")
                if total:
                    progress.update(int(job_progress.get("completed") or 0))
                if state == "completed":
                    break
                if state in ("failed", "cancelled"):
                    error = job.get("error") or {}
                    raise RuntimeError(f"sd-server job {state}: {error.get('message') or error}")
                time.sleep(0.4)
            else:
                self._http("POST", f"/sdcpp/v1/jobs/{job_id}/cancel", {}, timeout=10)
                raise RuntimeError("sd-server job did not finish in time")
        finally:
            self.in_flight -= 1
            self.last_used = time.time()

        images = (job.get("result") or {}).get("images") or []
        if not images:
            raise RuntimeError("sd-server returned no image")
        out = []
        for entry in images:
            encoded = entry.get("b64_json")
            if not encoded:
                continue
            raw = Image.open(io.BytesIO(base64.b64decode(encoded)))
            raw.load()
            out.append(raw)
        if not out:
            raise RuntimeError("sd-server returned an image without data")
        return out


def get_server():
    global _server
    if _server is None:
        _server = SDServer()
    return _server


@atexit.register
def _stop_server_on_exit():
    """The server holds the GPU and the port; it must not outlive ComfyUI."""
    if _server is not None:
        _server.stop()


def _install_term_cleanup():
    """Stop the server when ComfyUI is terminated.

    ComfyUI handles SIGINT itself (it dumps a traceback on the first Ctrl-C).
    SIGTERM has no handler, so the interpreter would die without running the
    atexit hook and the sd-server would keep the GPU and the port.
    """
    if threading.current_thread() is not threading.main_thread():
        return

    def handler(signum, frame):
        _stop_server_on_exit()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):
        pass


_install_term_cleanup()


LORA_TAG_RE = re.compile(r"<lora:([^:>]+)(?::([^:>]+))?>")


def split_lora_tags(prompt):
    """Pull the A1111 style LoRA tags out of a prompt.

    The server API does not read them: a LoRA travels as a structured field.
    Returns the clean prompt and the list of (name, multiplier) pairs.
    """
    loras = []

    def take(match):
        name = match.group(1).strip()
        try:
            multiplier = float(match.group(2)) if match.group(2) else 1.0
        except ValueError:
            multiplier = 1.0
        loras.append((name, multiplier))
        return ""

    return LORA_TAG_RE.sub(take, prompt).strip(), loras


def parse_sigma_list(sigmas):
    """The sigma schedule as numbers, or None when it is a preset name.

    The server takes the schedule as an array of floats; a preset name like
    `minimax-h3-image` has no server side equivalent, so the caller falls
    back to the spawn path.
    """
    values = []
    for part in sigmas.replace(" ", "").split(","):
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            return None
    return values or None


def model_context_args(diffusion_model, vae, tokenizer, encoders, vision_encoder, lora,
                       max_vram=0.0, offload_to_cpu=False, flash_attention=False,
                       text_encoder_cpu=True, model_args=""):
    """The model context flags, shared by the spawn path and the server start.

    The server receives the same placement/performance knobs as sd-cli, so the
    resident engine behaves identically to a one-shot render on this Mac.
    """
    args = []
    if diffusion_model.full_checkpoint:
        args += ["--model", diffusion_model.path]
    else:
        args += ["--diffusion-model", diffusion_model.path]
    if vae is not None:
        args += ["--vae", vae.path]
    if tokenizer is not None:
        # Qwen3.5 and friends ship their own vocabulary, so the
        # tokenizer.json has to travel with the text encoder.
        args += ["--tokenizer", tokenizer.path]
    for encoder in encoders:
        args += [text_encoder_flag(encoder.display), encoder.path]
    if vision_encoder is not None:
        args += ["--llm_vision", vision_encoder.path]
    if lora is not None:
        args += ["--lora-model-dir", os.path.dirname(lora.path)]
    if max_vram > 0:
        args += ["--max-vram", str(max_vram)]
    if offload_to_cpu:
        args += ["--offload-to-cpu"]
    if flash_attention:
        args += ["--flash-attn", "off,diffusion=on"]
    if text_encoder_cpu:
        args += ["--params-backend", "te=cpu"]
    if model_args.strip():
        args += ["--model-args", model_args.strip()]
    return args


def save_image_tensors(tensors, workdir, prefix):
    """Write image tensors as PNG files and return their paths.

    Only the spawn (sd-cli) path uses files; the server path sends base64.
    """
    paths = []
    for index, tensor in enumerate(tensors):
        array = (tensor.cpu().numpy() * 255.0).round().astype("uint8")
        path = os.path.join(workdir, f"{prefix}_{index}.png")
        Image.fromarray(array).save(path)
        paths.append(path)
    return paths


def merge_lora_entries(entries):
    """Merge (path, multiplier) tuples into one request entry per file.

    stable-diffusion.cpp's JSON parser already sums the multiplier per repeated
    path (the same map behaviour as the cli merging), but doing it here keeps
    the request small and makes intent obvious.
    """
    merged = {}
    order = []
    for name, multiplier in entries:
        if name not in merged:
            order.append(name)
            merged[name] = 0.0
        merged[name] += multiplier
    return [{"path": name, "multiplier": merged[name]} for name in order]


def server_request(prompt, negative_prompt, width, height, steps, cfg_scale, seed,
                   sampling_method, scheduler, flow_shift, sigmas, strength,
                   vae_tiling, lora, prompt_loras, workdir, reference_image_4, reference_images,
                   model_args_str="", batch_count=None):
    """The native img_gen body for one render.

    Images go as base64 PNG strings (the server reads them inline, no paths).
    LoRAs go as the structured `lora` array `[{path, multiplier}]`. Multiple
    LoRAs are supported by repeating entries or by summing per file.
    """
    body = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "seed": max(-1, min(seed, 0x7FFFFFFFFFFFFFFF)),
        "strength": strength,
        "output_format": "png",
        "sample_params": {
            "sample_steps": steps,
            "sample_method": sampling_method,
            "scheduler": scheduler,
            "guidance": {"txt_cfg": cfg_scale},
        },
    }
    if flow_shift >= 0:
        body["sample_params"]["flow_shift"] = flow_shift
    if sigmas.strip():
        values = parse_sigma_list(sigmas)
        if values is None:
            raise RuntimeError(
                f"the server needs the sigma schedule as numbers, not the preset {sigmas.strip()!r}"
            )
        body["sample_params"]["custom_sigmas"] = values
    loras = []
    if lora is not None:
        loras.append((lora.stem, lora.strength))
    loras.extend(prompt_loras)
    if loras:
        body["lora"] = merge_lora_entries(loras)
    if vae_tiling:
        body["vae_tiling_params"] = {"enabled": True}
    if batch_count is not None:
        body["batch_count"] = int(batch_count)
    if reference_image_4 is not None:
        body["init_image"] = _b64_png(reference_image_4[0])
    if reference_images:
        body["ref_images"] = b64_images(reference_images)
    return body


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
                                                            "The vision tower encodes it and it steers the "
                                                            "result; the prompt still decides the scene. "
                                                            "sd.cpp takes one --ref-image per image, in order."}),
                "reference_image_2": ("IMAGE", {"tooltip": "Second reference image."}),
                "reference_image_3": ("IMAGE", {"tooltip": "Third reference image."}),
                "reference_image_4": ("IMAGE", {"tooltip": "Fourth reference image. "
                                                            "Denoise starting from this image instead of "
                                                            "noise (img2img); strength below 1 keeps more of it."}),
                "input_image": ("IMAGE", {"tooltip": "Reference images as a batch: every image in it is "
                                                     "one reference, like the numbered sockets above."}),
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

    def _render_with_server(self, context_args, prompt, negative_prompt, width, height,
                            steps, cfg_scale, seed, sampling_method, scheduler, flow_shift,
                            sigmas, strength, vae_tiling, lora, reference_image_4, references):
        """Render through the resident sd-server process.

        The server keeps the model loaded, so the second render of a workflow
        skips the spawn, the Metal init and the weight load.
        """
        server = get_server()
        clean_prompt, prompt_loras = split_lora_tags(prompt)

        comfy.model_management.throw_exception_if_processing_interrupted()
        # ensure() serialises startup/teardown through its own lock; the GPU
        # is only contended during the launch window, not for the whole render.
        server.ensure(context_args, sdcli_env(sdserver_path()))

        body = server_request(clean_prompt, negative_prompt, width, height, steps,
                              cfg_scale, seed, sampling_method, scheduler, flow_shift,
                              sigmas, strength, vae_tiling, lora, prompt_loras,
                              None, reference_image_4, references)
        images = server.render(body, steps)
        out = []
        for raw in images:
            array = np.asarray(raw.convert("RGB")).astype(np.float32) / 255.0
            out.append(torch.from_numpy(array))
        if not out:
            raise RuntimeError("sd-server returned no image")
        return (torch.stack(out),)

    def generate(self, diffusion_model, prompt, negative_prompt,
                 width, height, steps, cfg_scale, seed, sampling_method, scheduler,
                 flow_shift, vae_tiling, vae=None, text_encoder=None, max_vram=0.0,
                 offload_to_cpu=False, text_encoder_2=None, tokenizer=None, vision_encoder=None, lora=None,
                 reference_image_4=None, input_image=None, strength=1.0,
                 flash_attention=False, text_encoder_cpu=True,
                 reference_image_1=None, reference_image_2=None, reference_image_3=None,
                 sigmas="", model_args="", randomize_prompt=True):
        # The reference slots are batches: every image in them is one reference.
        references = []
        for group in (reference_image_1, reference_image_2, reference_image_3, input_image):
            if group is not None:
                references.extend(group)
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

        # The model context is the same for both paths: the flags that say
        # which model, VAE, encoders and LoRA folder this render uses.
        context_args = model_context_args(
            diffusion_model, vae, tokenizer, encoders, vision_encoder, lora,
            max_vram=max_vram, offload_to_cpu=offload_to_cpu,
            flash_attention=flash_attention, text_encoder_cpu=text_encoder_cpu,
            model_args=model_args,
        )

        # sd.cpp takes over the GPU for the whole run. Hand it a clean slate:
        # torch's caching allocator keeps freed blocks reserved, which on this
        # 16 GiB card is the difference between the big models fitting and not.
        try:
            import gc
            gc.collect()
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache(force=True)
            if torch.backends.mps.is_available():
                # sd-cli can only use what the driver still shows free; these two
                # numbers say whether ComfyUI itself is the process holding it.
                print(f"[sd-cli] MPS after release: tensors {torch.mps.current_allocated_memory() / 2**20:.0f} MB, driver {torch.mps.driver_allocated_memory() / 2**20:.0f} MB")
        except Exception as error:
            print(f"[sd-cli] could not release the torch GPU cache: {error}")

        if server_mode_enabled():
            try:
                return self._render_with_server(context_args, prompt, negative_prompt,
                                                width, height, steps, cfg_scale, seed,
                                                sampling_method, scheduler, flow_shift,
                                                sigmas, strength, vae_tiling, lora,
                                                reference_image_4, references)
            except Exception as error:
                # The server is an optimisation, never a dependency: any
                # failure goes back to the spawn path.
                print(f"[sd-server] server path failed, falling back to sd-cli: {error}")

        with tempfile.TemporaryDirectory() as workdir:
            output = os.path.join(workdir, "out.png")
            cmd = [binary]
            cmd += context_args
            if lora is not None and "<lora:" not in prompt:
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
            if reference_image_4 is not None:
                input_path = save_image_tensors([reference_image_4[0]], workdir, "input")[0]
                cmd += ["-i", input_path, "--strength", str(strength)]
            if references:
                # Separate tensors, so the numbered references do not have to
                # share their dimensions - a portrait and a landscape reference
                # can go in side by side. sd.cpp still wants each of them to be
                # a multiple of 32 in both directions.
                for path in save_image_tensors(references, workdir, "ref"):
                    cmd += ["--ref-image", path]

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
