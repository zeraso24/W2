# start from a clean base image (replace <version> with the desired release)
FROM runpod/worker-comfyui:5.1.0-base

# install custom nodes using comfy-cli
RUN comfy-node-install comfyui-kjnodes comfyui-ic-light comfyui_ipadapter_plus comfyui_essentials ComfyUI-Hangover-Nodes

# download models using comfy-cli
# the "--filename" is what you use in your ComfyUI workflow
RUN comfy model download --url https://huggingface.co/KamCastle/jugg/resolve/main/juggernaut_reborn.safetensors --relative-path models/checkpoints --filename juggernaut_reborn.safetensors
RUN comfy model download --url https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin --relative-path models/ipadapter --filename ip-adapter-plus_sd15.bin
RUN comfy model download --url https://huggingface.co/shiertier/clip_vision/resolve/main/SD15/model.safetensors --relative-path models/clip_vision --filename models.safetensors
RUN comfy model download --url https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fcon.safetensors --relative-path models/diffusion_models --filename iclight_sd15_fcon.safetensors

# Copy local static input files into the ComfyUI input directory (delete if not needed)
# Assumes you have an 'input' folder next to your Dockerfile
COPY input/ /comfyui/input/
