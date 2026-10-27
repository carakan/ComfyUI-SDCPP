#!/usr/bin/env python3
# PLATFORM: macos-mps — converts ComfyUI-format Qwen-Image LoRAs for stable-diffusion.cpp.
"""Convert a ComfyUI-format LoRA to the key format stable-diffusion.cpp expects.

stable-diffusion.cpp looks up LoRA tensors by name:

    lora.<model tensor name>.lora_down
    lora.<model tensor name>.lora_up
    (optional) lora.<model tensor name>.scale / .alpha

ComfyUI writes them the other way round:

    diffusion_model.<module>.lora_A.weight
    diffusion_model.<module>.lora_B.weight

Hand the converted file to sd.cpp (--lora-model-dir + `<lora:name:strength>`
in the prompt) and it applies; the original file silently applies nothing.

Usage:

    python convert_lora_for_sdcpp.py <in.safetensors> [out.safetensors] [dtype]

Without an output path the result is written next to the input as
<name>.sdcpp.safetensors. Tensor data is copied verbatim, only the keys change.

The dtype argument (default f16) casts the LoRA weights. ai-toolkit writes
bf16, and the precompiled Metal library shipped with the sd.cpp build has no
bf16 matmul kernel — the run dies with "Function kernel_mul_mm_bf16_f16 was
not found in the library". f16 keeps the build's fast path.
"""
import json
import os
import struct
import sys


def read_header(path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size)), size


def convert_key(name):
    """ComfyUI key -> sd.cpp key. Returns None for keys that are not LoRA weights.

    sd.cpp looks a LoRA tensor up as `lora.<model tensor name>.lora_down`, where
    the model tensor name is the one in the GGUF (e.g.
    `transformer_blocks.0.attn.to_k.weight`). ComfyUI writes the module path
    with a `diffusion_model.` prefix and the rank tensors as lora_A/lora_B.
    """
    if name.endswith(".lora_A.weight"):
        base = name[: -len(".lora_A.weight")]
    elif name.endswith(".lora_B.weight"):
        base = name[: -len(".lora_B.weight")]
    elif name.endswith(".alpha"):
        base = name[: -len(".alpha")]
    else:
        return None
    for prefix in ("model.diffusion_model.", "diffusion_model.", "model.", "lora."):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    suffix = "lora_down" if name.endswith(".lora_A.weight") else (
        "lora_up" if name.endswith(".lora_B.weight") else "alpha")
    return f"lora.{base}.weight.{suffix}"


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    source = argv[0]
    target = argv[1] if len(argv) > 1 else os.path.splitext(source)[0] + ".sdcpp.safetensors"
    dtype = argv[2] if len(argv) > 2 else "f16"
    if not os.path.isfile(source):
        print(f"not a file: {source}")
        return 1

    header, header_size = read_header(source)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    metadata = header.get("__metadata__", {})

    mapping = {}
    skipped = []
    for name in tensors:
        converted = convert_key(name)
        if converted is None:
            skipped.append(name)
        else:
            mapping[name] = converted

    if not mapping:
        print(f"{source}: no ComfyUI LoRA tensors found (lora_A/lora_B)")
        return 1

    from safetensors.torch import load_file, save_file
    import torch

    target_dtype = {"f16": torch.float16, "f32": torch.float32, "bf16": torch.bfloat16}[dtype]
    data = load_file(source)
    converted = {}
    for name, tensor in data.items():
        if name not in mapping:
            continue
        if tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype)
        converted[mapping[name]] = tensor
    extra = {name: tensor for name, tensor in data.items() if name not in mapping}
    if extra:
        print(f"  keeping {len(extra)} non-LoRA tensors unchanged")
    save_file(converted, target, metadata=metadata)

    print(f"{source} -> {target}")
    print(f"  {len(mapping)} LoRA tensors renamed, {len(skipped)} left alone, dtype {dtype}")
    if skipped:
        print(f"  sample skipped: {skipped[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
