from .sdcpp_nodes import (
    HAVE_BINDINGS,
    SDCppCustomFlux,
    SDCPPUnetLoader,
    SDCPPVaeLoader,
    SDCPPTextEncoderLoader,
    SDCPPSampler,
)

NODE_CLASS_MAPPINGS = {
    "SDCPPUnetLoader": SDCPPUnetLoader,
    "SDCPPVaeLoader": SDCPPVaeLoader,
    "SDCPPTextEncoderLoader": SDCPPTextEncoderLoader,
    "SDCPPSampler": SDCPPSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDCPPUnetLoader": "SD.cpp Diffusion Model (sd-cli)",
    "SDCPPVaeLoader": "SD.cpp VAE (sd-cli)",
    "SDCPPTextEncoderLoader": "SD.cpp Text Encoder (sd-cli)",
    "SDCPPSampler": "SD.cpp Sampler (sd-cli)",
}

# The Flux node drives the stable_diffusion_cpp Python bindings. They have no
# wheel for every platform (macOS x86_64 has none), so the node is only
# registered when the bindings are importable.
if HAVE_BINDINGS:
    NODE_CLASS_MAPPINGS["SDCppCustomFlux"] = SDCppCustomFlux
    NODE_DISPLAY_NAME_MAPPINGS["SDCppCustomFlux"] = "SD.cpp Flux (UNet/VAE/CLIP)"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
