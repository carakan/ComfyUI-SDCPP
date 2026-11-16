#!/usr/bin/env python3
"""Check the node schemas of the ComfyUI-SDCPP plugin.

ComfyUI hands the inputs to a node by name, so a renamed or removed input
breaks every saved workflow that uses it, and a widget that moves inside the
schema shifts the widget values of the workflows that were saved before it.
Both are silent failures: the node loads and then produces something else.

The checks:

  * every declared input is a parameter of the node's function, and every
    parameter is declared (a mismatch is a runtime error on the first call),
  * the sampler's input names are pinned, so a rename shows up as a diff here
    and not as a broken workflow,
  * the widget order is pinned. The widget values in a saved workflow are a
    positional array, so a new widget has to be appended at the end and the
    image sockets in between do not move anything.

Run it from the ComfyUI directory:

    ./venv/bin/python custom_nodes/ComfyUI-SDCPP/tests/test_node_schema.py
"""

import inspect
import os
import sys

PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.dirname(os.path.dirname(PLUGIN))
for path in (COMFY, PLUGIN):
    if path not in sys.path:
        sys.path.insert(0, path)

import sdcpp_nodes  # noqa: E402

# The widget types of ComfyUI. Everything else (IMAGE, MASK, LATENT, the
# custom model types) is a socket and does not take a widget value.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}

# The pinned sampler schema. Change it on purpose when the node changes.
SAMPLER_REQUIRED = [
    "diffusion_model", "prompt", "negative_prompt", "width", "height", "steps",
    "cfg_scale", "seed", "sampling_method", "scheduler", "flow_shift",
    "vae_tiling", "max_vram", "offload_to_cpu",
]
SAMPLER_OPTIONAL = [
    "vae", "text_encoder", "text_encoder_2", "tokenizer", "vision_encoder",
    "lora", "reference_image_1", "reference_image_2", "reference_image_3",
    "reference_image_4", "input_image", "strength", "flash_attention", "sigmas", "model_args",
    "randomize_prompt", "text_encoder_cpu", "resolution",
]
# The widget values of a saved workflow are positional, so this order matters.
# diffusion_model is a socket (SDCPP_MODEL), not a widget: the loader node
# hands the model over.
SAMPLER_WIDGETS = [
    "prompt", "negative_prompt", "width", "height", "steps",
    "cfg_scale", "seed", "sampling_method", "scheduler", "flow_shift",
    "vae_tiling", "max_vram", "offload_to_cpu", "strength", "flash_attention",
    "sigmas", "model_args", "randomize_prompt", "text_encoder_cpu", "resolution",
]
# The image inputs and what they do.
IMAGE_INPUTS = {
    "reference_image_4": "img2img: the image is the canvas, the prompt re-textures it",
    "input_image": "reference images as a batch",
    "reference_image_1": "reference: the vision tower steers the result",
    "reference_image_2": "reference: the vision tower steers the result",
    "reference_image_3": "reference: the vision tower steers the result",
}

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)


def is_widget(spec):
    if not isinstance(spec, (tuple, list)) or not spec:
        return False
    kind = spec[0]
    if isinstance(kind, str):
        return kind in WIDGET_TYPES
    # A combo: the first entry is the list of choices.
    return isinstance(kind, (list, tuple))


def widgets_of(schema):
    names = []
    for section in ("required", "optional"):
        for name, spec in schema.get(section, {}).items():
            if is_widget(spec):
                names.append(name)
    return names


def check_node(name, cls):
    schema = cls.INPUT_TYPES()
    check(isinstance(schema, dict), f"{name}: INPUT_TYPES is not a dict")
    check("required" in schema, f"{name}: INPUT_TYPES has no required section")
    for section in ("required", "optional"):
        entries = schema.get(section, {})
        check(isinstance(entries, dict), f"{name}: {section} is not a dict")
        for input_name, spec in entries.items():
            check(isinstance(spec, (tuple, list)) and spec,
                  f"{name}: input {input_name} has no type")

    declared = list(schema.get("required", {}).keys()) + list(schema.get("optional", {}).keys())
    check(len(declared) == len(set(declared)), f"{name}: an input name is declared twice")

    function = getattr(cls, cls.FUNCTION, None)
    check(function is not None, f"{name}: FUNCTION {cls.FUNCTION} does not exist")
    if function is None:
        return
    parameters = [p for p in inspect.signature(function).parameters if p != "self"]
    missing = sorted(set(declared) - set(parameters))
    extra = sorted(set(parameters) - set(declared))
    check(not missing, f"{name}: declared but not accepted by {cls.FUNCTION}: {missing}")
    check(not extra, f"{name}: accepted but not declared: {extra}")


def main():
    for name, cls in sdcpp_nodes.__dict__.items():
        if not inspect.isclass(cls) or not hasattr(cls, "INPUT_TYPES"):
            continue
        if not hasattr(cls, "FUNCTION"):
            continue
        check_node(cls.__name__, cls)

    sampler = sdcpp_nodes.SDCPPSampler
    schema = sampler.INPUT_TYPES()
    check(list(schema["required"].keys()) == SAMPLER_REQUIRED,
          f"SDCPPSampler required inputs changed:\n  now:  {list(schema['required'].keys())}\n"
          f"  want: {SAMPLER_REQUIRED}")
    check(list(schema["optional"].keys()) == SAMPLER_OPTIONAL,
          f"SDCPPSampler optional inputs changed:\n  now:  {list(schema['optional'].keys())}\n"
          f"  want: {SAMPLER_OPTIONAL}")
    check(widgets_of(schema) == SAMPLER_WIDGETS,
          f"the widget order changed, the saved workflows shift their values:\n"
          f"  now:  {widgets_of(schema)}\n  want: {SAMPLER_WIDGETS}")

    for name, role in IMAGE_INPUTS.items():
        check(name in schema.get("optional", {}), f"SDCPPSampler lost the input {name}")
        spec = schema.get("optional", {}).get(name)
        if spec:
            check(spec[0] == "IMAGE", f"SDCPPSampler input {name} is {spec[0]}, not IMAGE")

    check("strength" in schema.get("optional", {}),
          "SDCPPSampler lost the strength input of the img2img path")

    # The resolution menu and the sizes it stands for.
    resolutions = sdcpp_nodes.RESOLUTIONS
    spec = schema.get("optional", {}).get("resolution")
    check(spec is not None and list(spec[0]) == list(resolutions),
          "the resolution choices do not match RESOLUTIONS")
    for label, size in resolutions.items():
        if size is None:
            continue
        check(label.startswith(f"{size[0]}x{size[1]} ("),
              f"the resolution label {label!r} does not name its size {size}")
        check(size[0] >= 64 and size[1] >= 64,
              f"the resolution {label!r} is smaller than the node allows")

    if failures:
        print("FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"OK: {len(SAMPLER_REQUIRED) + len(SAMPLER_OPTIONAL)} sampler inputs, "
          f"{len(SAMPLER_WIDGETS)} widgets, all node schemas match their functions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
