"""
FLUX.2 Flex Edit Node
A dedicated node for FLUX.2 Flex image editing functionality.
Supports multi-reference image editing with customizable sizes.
Independent version - no external dependencies required.
"""

import os
import io
import tempfile
import configparser

import torch
import numpy as np
import requests
from PIL import Image
from fal_client.client import SyncClient


# ==================== FAL API UTILITIES ====================

class FalConfig:
    """Singleton class to handle FAL configuration and client setup."""

    _instance = None
    _client = None
    _key = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FalConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self, api_key_override=None):
        """Initialize configuration and API key."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        try:
            # Priority: 1. API key from node parameter, 2. Environment variable, 3. config.ini
            if api_key_override and api_key_override.strip() and api_key_override != "<your_fal_api_key_here>":
                self._key = api_key_override.strip()
                print("FAL_KEY found in node parameter")
            elif os.environ.get("FAL_KEY") is not None:
                print("FAL_KEY found in environment variables")
                self._key = os.environ["FAL_KEY"]
            else:
                print("FAL_KEY not found in environment variables")
                self._key = config["API"]["FAL_KEY"]
                print("FAL_KEY found in config.ini")
                os.environ["FAL_KEY"] = self._key
                print("FAL_KEY set in environment variables")

            # Check if FAL key is the default placeholder
            if self._key == "<your_fal_api_key_here>" or not self._key or self._key.strip() == "":
                print("WARNING: You are using the default FAL API key placeholder or empty key!")
                print("Please set your actual FAL API key in either:")
                print("1. The node parameter 'api_key' (recommended)")
                print("2. The config.ini file under [API] section")
                print("3. Or as an environment variable named FAL_KEY")
                print("Get your API key from: https://fal.ai/dashboard/keys")
        except KeyError:
            print("Error: FAL_KEY not found in config.ini or environment variables")

    def get_client(self, api_key_override=None):
        """Get or create the FAL client."""
        # If API key override is provided, use it; otherwise use the stored key
        key_to_use = api_key_override if api_key_override and api_key_override.strip() else self._key
        
        if not key_to_use or key_to_use == "<your_fal_api_key_here>":
            raise ValueError("FAL API key is not set. Please provide a valid API key.")
        
        # Create a new client if key is different, or reuse existing client
        if self._client is None or (api_key_override and api_key_override != self._key):
            self._client = SyncClient(key=key_to_use)
        return self._client

    def get_key(self):
        """Get the FAL API key."""
        return self._key


class ImageUtils:
    """Utility functions for image processing."""

    @staticmethod
    def tensor_to_pil(image):
        """Convert image tensor to PIL Image."""
        try:
            # Convert the image tensor to a numpy array
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)

            # Ensure the image is in the correct format (H, W, C)
            if image_np.ndim == 4:
                image_np = image_np.squeeze(0)  # Remove batch dimension if present
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)  # Convert grayscale to RGB
            elif image_np.shape[0] == 3:
                image_np = np.transpose(
                    image_np, (1, 2, 0)
                )  # Change from (C, H, W) to (H, W, C)

            # Normalize the image data to 0-255 range
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).astype(np.uint8)

            # Convert to PIL Image
            return Image.fromarray(image_np)
        except Exception as e:
            print(f"Error converting tensor to PIL: {str(e)}")
            return None

    @staticmethod
    def upload_image(image, api_key=None):
        """Upload image tensor to FAL and return URL."""
        try:
            pil_image = ImageUtils.tensor_to_pil(image)
            if not pil_image:
                return None

            # Save the image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                pil_image.save(temp_file, format="PNG")
                temp_file_path = temp_file.name

            # Upload the temporary file
            fal_config = FalConfig()
            client = fal_config.get_client(api_key_override=api_key)
            image_url = client.upload_file(temp_file_path)
            return image_url
        except Exception as e:
            print(f"Error uploading image: {str(e)}")
            return None
        finally:
            # Clean up the temporary file
            if "temp_file_path" in locals():
                os.unlink(temp_file_path)
                
    @staticmethod
    def upload_file(file_path, api_key=None):
        """Upload a file to FAL and return URL."""
        try:
            fal_config = FalConfig()
            client = fal_config.get_client(api_key_override=api_key)
            file_url = client.upload_file(file_path)
            return file_url
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
            return None
        
    @staticmethod
    def mask_to_image(mask):
        """Convert mask tensor to image tensor."""
        result = (
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            .movedim(1, -1)
            .expand(-1, -1, -1, 3)
        )
        return result


class ResultProcessor:
    """Utility functions for processing API results."""

    @staticmethod
    def process_image_result(result):
        """Process image generation result and return tensor."""
        try:
            images = []
            for img_info in result["images"]:
                img_url = img_info["url"]
                img_response = requests.get(img_url)
                img = Image.open(io.BytesIO(img_response.content))
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)

            # Stack the images along a new first dimension
            stacked_images = np.stack(images, axis=0)

            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(stacked_images)

            return (img_tensor,)
        except Exception as e:
            print(f"Error processing image result: {str(e)}")
            return ResultProcessor.create_blank_image()

    @staticmethod
    def process_single_image_result(result):
        """Process single image result and return tensor."""
        try:
            img_url = result["image"]["url"]
            img_response = requests.get(img_url)
            img = Image.open(io.BytesIO(img_response.content))
            img_array = np.array(img).astype(np.float32) / 255.0

            # Stack the images along a new first dimension
            stacked_images = np.stack([img_array], axis=0)

            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(stacked_images)
            return (img_tensor,)
        except Exception as e:
            print(f"Error processing single image result: {str(e)}")
            return ResultProcessor.create_blank_image()

    @staticmethod
    def create_blank_image():
        """Create a blank black image tensor."""
        blank_img = Image.new("RGB", (512, 512), color="black")
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)


class ApiHandler:
    """Utility functions for API interactions."""

    @staticmethod
    def submit_and_get_result(endpoint, arguments, api_key=None):
        """Submit job to FAL API and get result."""
        try:
            fal_config = FalConfig()
            client = fal_config.get_client(api_key_override=api_key)
            handler = client.submit(endpoint, arguments=arguments)
            return handler.get()
        except Exception as e:
            print(f"Error submitting to {endpoint}: {str(e)}")
            raise e

    @staticmethod
    def handle_video_generation_error(model_name, error):
        """Handle video generation errors consistently."""
        print(f"Error generating video with {model_name}: {str(error)}")
        return ("Error: Unable to generate video.",)

    @staticmethod
    def handle_image_generation_error(model_name, error):
        """Handle image generation errors consistently."""
        print(f"Error generating image with {model_name}: {str(error)}")
        return ResultProcessor.create_blank_image()

    @staticmethod
    def handle_text_generation_error(model_name, error):
        """Handle text generation errors consistently."""
        print(f"Error generating text with {model_name}: {str(error)}")
        return ("Error: Unable to generate text.",)


# ==================== FLUX.2 FLEX EDIT NODE ====================


class Flux2FlexEdit:
    """
    FLUX.2 Flex Edit Node
    Performs image editing using FLUX.2 [flex] model from Black Forest Labs.
    Supports multi-reference editing with customizable inference steps and enhanced text rendering.
    Requires: image(s) and prompt.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Primary image to be edited"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prompt describing the desired image editing"}),
            },
            "optional": {
                "image_2": ("IMAGE", {"tooltip": "Second reference image for multi-reference editing (optional)"}),
                "image_3": ("IMAGE", {"tooltip": "Third reference image for multi-reference editing (optional)"}),
                "api_key": ("STRING", {"default": "", "tooltip": "FAL API key (optional, overrides config.ini and environment variable)"}),
                "image_size_mode": (["preset", "custom"], {"default": "preset", "tooltip": "Choose between preset sizes or custom width/height"}),
                "image_size_preset": (["auto", "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"], {"default": "auto", "tooltip": "Preset image size. 'auto' lets the model determine the size"}),
                "custom_width": ("INT", {"default": 1280, "min": 128, "max": 4096, "step": 8, "tooltip": "Custom width (only used when image_size_mode is 'custom')"}),
                "custom_height": ("INT", {"default": 720, "min": 128, "max": 4096, "step": 8, "tooltip": "Custom height (only used when image_size_mode is 'custom')"}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "The guidance scale to use for the generation"}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1, "tooltip": "The number of inference steps to perform"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "Random seed used for the generation process (-1 = random)"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True, "tooltip": "Whether to expand the prompt using the model's own knowledge"}),
                "safety_tolerance": (["1", "2", "3", "4", "5"], {"default": "2", "tooltip": "Safety tolerance level (1 = most strict, 5 = most permissive)"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True, "tooltip": "Whether to enable the safety checker"}),
                "output_format": (["jpeg", "png"], {"default": "jpeg", "tooltip": "The format of the output image"}),
                "sync_mode": ("BOOLEAN", {"default": False, "tooltip": "If True, media will be returned as data URI and won't be available in request history"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "plexard_FAL"
    DESCRIPTION = "FLUX.2 Flex Edit - Multi-reference image editing with FLUX.2 [flex]"

    def edit(
        self,
        image,
        prompt,
        image_2=None,
        image_3=None,
        api_key="",
        image_size_mode="preset",
        image_size_preset="auto",
        custom_width=1280,
        custom_height=720,
        guidance_scale=3.5,
        num_inference_steps=28,
        seed=-1,
        enable_prompt_expansion=True,
        safety_tolerance="2",
        enable_safety_checker=True,
        output_format="jpeg",
        sync_mode=False,
    ):
        """
        Perform image editing using FLUX.2 Flex model.
        
        Args:
            image: Primary input image tensor (required)
            prompt: Text prompt describing the desired editing (required)
            image_2: Second reference image for multi-reference editing (optional)
            image_3: Third reference image for multi-reference editing (optional)
            api_key: FAL API key (optional, overrides config.ini and environment variable)
            image_size_mode: "preset" or "custom" (default: "preset")
            image_size_preset: Preset size option (default: "auto")
            custom_width: Custom width when image_size_mode is "custom" (default: 1280)
            custom_height: Custom height when image_size_mode is "custom" (default: 720)
            guidance_scale: Guidance scale for generation (default: 3.5)
            num_inference_steps: Number of inference steps (default: 28)
            seed: Random seed for generation (-1 = random, default: -1)
            enable_prompt_expansion: Enable prompt expansion (default: True)
            safety_tolerance: Safety tolerance level 1-5 (default: "2")
            enable_safety_checker: Enable safety checker (default: True)
            output_format: Output format - "jpeg" or "png" (default: "jpeg")
            sync_mode: Return as data URI (default: False)
            
        Returns:
            Tuple containing the edited image tensor
        """
        model_name = "FLUX.2 Flex Edit"
        
        # Validate and sanitize input parameters
        try:
            # Validate image_size_mode
            valid_modes = ["preset", "custom"]
            if image_size_mode not in valid_modes:
                image_size_mode = "preset"
            
            # Validate image_size_preset
            valid_presets = ["auto", "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"]
            if image_size_preset not in valid_presets:
                image_size_preset = "auto"
            
            # Validate custom dimensions
            if custom_width is None or not isinstance(custom_width, (int, float)):
                custom_width = 1280
            custom_width = int(max(128, min(4096, custom_width)))
            # Round to nearest multiple of 8
            custom_width = (custom_width // 8) * 8
            
            if custom_height is None or not isinstance(custom_height, (int, float)):
                custom_height = 720
            custom_height = int(max(128, min(4096, custom_height)))
            # Round to nearest multiple of 8
            custom_height = (custom_height // 8) * 8
            
            # Validate guidance_scale
            if guidance_scale is None or not isinstance(guidance_scale, (int, float)):
                guidance_scale = 3.5
            guidance_scale = float(max(0.0, min(20.0, guidance_scale)))
            
            # Validate num_inference_steps
            if num_inference_steps is None or not isinstance(num_inference_steps, (int, float)):
                num_inference_steps = 28
            num_inference_steps = int(max(1, min(100, num_inference_steps)))
            
            # Validate safety_tolerance
            valid_tolerances = ["1", "2", "3", "4", "5"]
            if safety_tolerance not in valid_tolerances:
                safety_tolerance = "2"
            
            # Validate output_format
            valid_formats = ["jpeg", "png"]
            if output_format not in valid_formats:
                output_format = "jpeg"
            
            # Validate seed
            if seed == -1:
                seed = None  # API will use random seed
            elif seed is not None:
                seed = int(max(-1, min(2147483647, seed)))
        except Exception as e:
            print(f"Warning: Error validating parameters: {str(e)}, using defaults")
            image_size_mode = "preset"
            image_size_preset = "auto"
            custom_width = 1280
            custom_height = 720
            guidance_scale = 3.5
            num_inference_steps = 28
            safety_tolerance = "2"
            output_format = "jpeg"
            seed = None
        
        # Get API key for uploads (priority: node parameter > environment > config.ini)
        api_key_for_upload = api_key if api_key and api_key.strip() and api_key != "<your_fal_api_key_here>" else None
        
        # Upload input images - FLUX.2 Flex requires image_urls (list)
        image_urls = []
        
        # Upload primary image (required)
        image_url = ImageUtils.upload_image(image, api_key=api_key_for_upload)
        if not image_url:
            print(f"Error: Failed to upload primary image for {model_name}")
            return ResultProcessor.create_blank_image()
        image_urls.append(image_url)
        
        # Upload optional additional images
        if image_2 is not None:
            image_url_2 = ImageUtils.upload_image(image_2, api_key=api_key_for_upload)
            if image_url_2:
                image_urls.append(image_url_2)
            else:
                print(f"Warning: Failed to upload second image for {model_name}, continuing without it")
        
        if image_3 is not None:
            image_url_3 = ImageUtils.upload_image(image_3, api_key=api_key_for_upload)
            if image_url_3:
                image_urls.append(image_url_3)
            else:
                print(f"Warning: Failed to upload third image for {model_name}, continuing without it")

        # Build API arguments according to fal.ai FLUX.2 Flex Edit API schema
        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "enable_prompt_expansion": enable_prompt_expansion,
            "safety_tolerance": safety_tolerance,
            "enable_safety_checker": enable_safety_checker,
            "output_format": output_format,
        }
        
        # Handle image_size based on mode
        if image_size_mode == "custom":
            arguments["image_size"] = {
                "width": custom_width,
                "height": custom_height
            }
        else:
            arguments["image_size"] = image_size_preset
        
        # Add seed only if provided (not -1)
        if seed is not None:
            arguments["seed"] = seed
        
        # Add sync_mode only if True (default is False)
        if sync_mode:
            arguments["sync_mode"] = True

        # Submit request to FAL API
        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/flux-2-flex/edit", arguments, api_key=api_key_for_upload
            )
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            print(f"Error in {model_name}: {str(e)}")
            return ApiHandler.handle_image_generation_error(model_name, e)


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Flux2FlexEdit": Flux2FlexEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2FlexEdit": "FLUX.2 Flex Edit",
}

