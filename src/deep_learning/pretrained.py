"""
Pre-trained comparison model wrappers.

Provides a unified interface for running inference with pre-trained
colorization models for benchmarking against our Zhang 2016 reimplementation.

Model categories (4 for comparison):
    - CNN: Zhang 2016 (our reimplementation, see colorizer.py)
    - Interactive CNN: Zhang 2017 (automatic mode with zero hints)
    - GAN: DeOldify (Self-Attention GAN / NoGAN)
    - Diffusion: ControlNet + Stable Diffusion

Theory papers mapped to practical models:
    - CNN: "Colorful Image Colorization" (Zhang, ECCV 2016)
    - Interactive CNN: "Real-Time User-Guided Image Colorization" (Zhang, SIGGRAPH 2017)
    - GAN: ChromaGAN (Vitoria, WACV 2020) -> practical: DeOldify
    - Diffusion: Palette (Saharia, CVPR 2022) -> practical: ControlNet (Zhang & Agrawala, ICCV 2023)

All wrappers expose the same API:
    colorize(gray_image) -> (result_bgr, info_dict)
"""

import time
import os
import tempfile
from abc import ABC, abstractmethod
import numpy as np
import cv2


class PretrainedColorizer(ABC):
    """Abstract base class for pretrained colorization models."""

    @abstractmethod
    def colorize(self, gray_image):
        """Colorize a grayscale image.

        Args:
            gray_image: np.ndarray (H, W) or (H, W, 3) grayscale image.

        Returns:
            result_bgr: np.ndarray (H, W, 3) uint8 BGR colorized image.
            info_dict: dict with metadata.
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """Return the model name for display."""
        pass

    @property
    def category(self):
        """Return the model category (CNN, Interactive CNN, GAN, Diffusion)."""
        return "unknown"

    @property
    def is_available(self):
        """Check if this model's dependencies are installed."""
        return True


# =============================================================================
# Interactive CNN: Zhang et al. 2017 (SIGGRAPH)
# =============================================================================

class Zhang2017Colorizer(PretrainedColorizer):
    """Interactive CNN colorization using Zhang et al. 2017 in automatic mode.

    The SIGGRAPH 2017 model is designed for user-guided colorization but
    can also run fully automatically with zero hints (input_B=None, mask_B=None).
    In automatic mode it colorizes based solely on learned priors.

    Category: Interactive CNN
    Paper: "Real-Time User-Guided Image Colorization with Learned Deep Priors"
           (Zhang et al., SIGGRAPH 2017)
    Official repo: https://github.com/richzhang/colorization

    Requires: colorizers package (pip install colorizers) OR manual weight loading.
    """

    def __init__(self, device="auto"):
        self._model = None
        self._available = False
        self._device_str = device

        try:
            import torch
            self._torch = torch
            # Try official colorizers package first
            try:
                from colorizers import siggraph17
                self._load_fn = siggraph17
                self._use_package = True
            except ImportError:
                # Fall back to manual loading
                self._use_package = False
            self._available = True
        except ImportError:
            print("PyTorch not available for Zhang 2017")

    @property
    def name(self):
        return "Zhang 2017 (Interactive CNN)"

    @property
    def category(self):
        return "Interactive CNN"

    @property
    def is_available(self):
        return self._available

    def _load_model(self):
        if self._model is not None:
            return

        if self._use_package:
            self._model = self._load_fn(pretrained=True).eval()
        else:
            # Manual loading from downloaded weights
            self._model = self._build_siggraph17_manual()

        if self._device_str == "auto":
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
        else:
            device = self._device_str
        self._model = self._model.to(device)
        self._device = device

    def _build_siggraph17_manual(self):
        """Build Zhang 2017 architecture manually if colorizers package unavailable.

        Downloads weights from the official S3 bucket.
        """
        import torch
        import torch.nn as nn

        # Download weights
        weights_path = os.path.join("models", "pretrained", "zhang17_siggraph.pth")
        if not os.path.exists(weights_path):
            url = "https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth"
            print(f"Downloading Zhang 2017 weights from {url}...")
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            import requests
            resp = requests.get(url, stream=True)
            with open(weights_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved to {weights_path}")

        # Try loading via colorizers-style architecture
        # If colorizers package isn't available, we need a minimal reimplementation
        # For now, raise an informative error
        raise ImportError(
            "Manual Zhang 2017 loading not implemented. "
            "Install the colorizers package: pip install colorizers"
        )

    def colorize(self, gray_image):
        start = time.time()

        if not self._available:
            raise RuntimeError("Zhang 2017 model is not available")

        self._load_model()

        import torch
        from PIL import Image
        from skimage import color as skcolor

        h, w = gray_image.shape[:2]

        # Prepare input: convert to Lab L channel
        if gray_image.ndim == 2:
            # Single channel grayscale
            rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)

        # Resize to 256x256 for the model
        rgb_resized = cv2.resize(rgb, (256, 256))

        # Convert to Lab
        img_float = rgb_resized.astype(np.float64) / 255.0
        lab = skcolor.rgb2lab(img_float)
        L = lab[:, :, 0]  # [0, 100]

        # Normalize L to [-50, 50] (Zhang's convention) then to tensor
        L_input = L - 50.0
        L_tensor = torch.from_numpy(L_input).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)
        L_tensor = L_tensor.to(self._device)

        # Run model with NO hints (automatic mode)
        with torch.no_grad():
            if self._use_package:
                # colorizers package API: model(input_l) returns ab
                ab_output = self._model(L_tensor).cpu()
            else:
                ab_output = self._model(L_tensor).cpu()

        # Post-process: combine L + predicted ab -> RGB
        ab_np = ab_output[0].permute(1, 2, 0).numpy()  # (H, W, 2)

        # Resize ab to original resolution
        ab_full = cv2.resize(ab_np, (w, h), interpolation=cv2.INTER_LINEAR)

        # Get L at original resolution
        if gray_image.ndim == 2:
            rgb_orig = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_orig = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)
        img_orig_float = rgb_orig.astype(np.float64) / 255.0
        lab_orig = skcolor.rgb2lab(img_orig_float)
        L_orig = lab_orig[:, :, 0]

        # Reconstruct
        lab_result = np.zeros((h, w, 3), dtype=np.float64)
        lab_result[:, :, 0] = L_orig
        lab_result[:, :, 1:] = ab_full
        rgb_result = skcolor.lab2rgb(lab_result)
        rgb_result = np.clip(rgb_result * 255, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2BGR)

        info_dict = {
            "elapsed_sec": time.time() - start,
            "model_name": "Zhang2017-SIGGRAPH",
            "category": "Interactive CNN",
            "mode": "automatic (zero hints)",
            "using_package": self._use_package,
        }
        return result_bgr, info_dict


# =============================================================================
# GAN: DeOldify
# =============================================================================

class DeOldifyColorizer(PretrainedColorizer):
    """GAN-based colorization using DeOldify.

    DeOldify uses a Self-Attention GAN (NoGAN) architecture for
    restoring and colorizing old photographs. It is the industry
    standard for old photo restoration.

    Category: GAN
    Theory reference: ChromaGAN (Vitoria et al., WACV 2020)
    Practical model: DeOldify (Jason Antic)
    Official repo (ChromaGAN): https://github.com/pvitoria/ChromaGAN

    Requires: deoldify package and fastai.
    """

    def __init__(self, model_path=None, render_factor=35):
        self._model = None
        self._model_path = model_path
        self._render_factor = render_factor
        self._available = False

        try:
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            from deoldify import device as deoldify_device
            from deoldify.device_id import DeviceId
            deoldify_device.set(device=DeviceId.GPU0)
            from deoldify.visualize import get_image_colorizer
            self._get_colorizer = get_image_colorizer
            self._available = True
        except ImportError:
            print("DeOldify not available. Install with: pip install deoldify")

    @property
    def name(self):
        return "DeOldify (GAN)"

    @property
    def category(self):
        return "GAN"

    @property
    def is_available(self):
        return self._available

    def _load_model(self):
        if self._model is None and self._available:
            self._model = self._get_colorizer(artistic=True)

    def colorize(self, gray_image):
        start = time.time()

        if not self._available:
            raise RuntimeError("DeOldify is not installed")

        self._load_model()

        # DeOldify expects a file path, so save to temp
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            if gray_image.ndim == 2:
                cv2.imwrite(tmp_path, gray_image)
            else:
                cv2.imwrite(tmp_path, gray_image)

        try:
            result_pil = self._model.get_transformed_image(
                tmp_path, render_factor=self._render_factor
            )
            result_rgb = np.array(result_pil)
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        finally:
            os.unlink(tmp_path)

        # Resize to match input
        h, w = gray_image.shape[:2]
        result_bgr = cv2.resize(result_bgr, (w, h))

        info_dict = {
            "elapsed_sec": time.time() - start,
            "model_name": "DeOldify-Artistic",
            "category": "GAN",
            "render_factor": self._render_factor,
        }
        return result_bgr, info_dict


# =============================================================================
# Diffusion: ControlNet + Stable Diffusion
# =============================================================================

class ControlNetColorizer(PretrainedColorizer):
    """Diffusion-based colorization using ControlNet + Stable Diffusion.

    Uses ControlNet to condition Stable Diffusion on a grayscale image,
    producing colorized output through the diffusion denoising process.

    Category: Diffusion
    Theory reference: Palette (Saharia et al., CVPR 2022)
    Practical model: ControlNet (Zhang & Agrawala, ICCV 2023)
    Official repo: https://github.com/lllyasviel/ControlNet

    Requires: diffusers, transformers packages.
    VRAM: ~6-8 GB with FP16 and memory optimizations.
    """

    def __init__(self, controlnet_model="neurallove/controlnet-sd21-colorization-diffusers",
                 sd_model="stabilityai/stable-diffusion-2-1-base",
                 num_inference_steps=30, guidance_scale=7.5, device="auto"):
        self._pipe = None
        self._controlnet_model = controlnet_model
        self._sd_model = sd_model
        self._num_steps = num_inference_steps
        self._guidance_scale = guidance_scale
        self._available = False
        self._device_str = device

        try:
            import torch
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
            self._torch = torch
            self._Pipeline = StableDiffusionControlNetPipeline
            self._ControlNetModel = ControlNetModel
            self._available = True
        except ImportError:
            print("ControlNet not available. Install with: pip install diffusers transformers accelerate")

    @property
    def name(self):
        return "ControlNet (Diffusion)"

    @property
    def category(self):
        return "Diffusion"

    @property
    def is_available(self):
        return self._available

    def _load_model(self):
        if self._pipe is None and self._available:
            try:
                print(f"Loading ControlNet: {self._controlnet_model}")
                controlnet = self._ControlNetModel.from_pretrained(
                    self._controlnet_model,
                    torch_dtype=self._torch.float16
                )

                print(f"Loading SD pipeline: {self._sd_model}")
                self._pipe = self._Pipeline.from_pretrained(
                    self._sd_model,
                    controlnet=controlnet,
                    torch_dtype=self._torch.float16,
                    safety_checker=None,
                )

                # Memory optimizations for 6GB VRAM
                if self._device_str == "auto":
                    if self._torch.cuda.is_available():
                        self._pipe.enable_model_cpu_offload()
                    else:
                        self._pipe = self._pipe.to("cpu")
                else:
                    self._pipe = self._pipe.to(self._device_str)

                try:
                    self._pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    self._pipe.enable_attention_slicing()

                print("ControlNet pipeline loaded successfully")
            except Exception as e:
                print(f"Could not load ControlNet: {e}")
                self._available = False

    def colorize(self, gray_image):
        start = time.time()

        if not self._available:
            raise RuntimeError("ControlNet is not available")

        self._load_model()

        if self._pipe is None:
            raise RuntimeError("ControlNet pipeline failed to load")

        from PIL import Image

        h, w = gray_image.shape[:2]

        # Convert grayscale to PIL RGB
        if gray_image.ndim == 2:
            pil_image = Image.fromarray(gray_image).convert("RGB")
        else:
            rgb = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

        # Resize to a multiple of 8 (required by SD)
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)

        # Run diffusion
        with self._torch.no_grad():
            result = self._pipe(
                prompt="high quality, detailed, colorful photograph",
                image=pil_image,
                num_inference_steps=self._num_steps,
                guidance_scale=self._guidance_scale,
                controlnet_conditioning_scale=1.0,
            )

        # Convert result to BGR numpy
        result_rgb = np.array(result.images[0])
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        # Resize back to original
        result_bgr = cv2.resize(result_bgr, (w, h))

        info_dict = {
            "elapsed_sec": time.time() - start,
            "model_name": "ControlNet+SD",
            "category": "Diffusion",
            "num_inference_steps": self._num_steps,
            "guidance_scale": self._guidance_scale,
            "controlnet_model": self._controlnet_model,
        }
        return result_bgr, info_dict


# =============================================================================
# Factory
# =============================================================================

def get_comparison_models(cfg):
    """Factory function to create available comparison model instances.

    Creates instances of all enabled comparison models from config.
    Models that fail to load (missing dependencies) are silently skipped.

    Args:
        cfg: Config dict with cfg["deep_learning"]["comparison"] section.

    Returns:
        dict mapping model_name -> PretrainedColorizer instance.
    """
    models = {}
    comparison_cfg = cfg.get("deep_learning", {}).get("comparison", {})

    # Interactive CNN: Zhang 2017 (automatic mode)
    if comparison_cfg.get("zhang2017", {}).get("enabled", False):
        model = Zhang2017Colorizer()
        if model.is_available:
            models[model.name] = model

    # GAN: DeOldify
    if comparison_cfg.get("deoldify", {}).get("enabled", False):
        model = DeOldifyColorizer(
            model_path=comparison_cfg["deoldify"].get("model_path")
        )
        if model.is_available:
            models[model.name] = model

    # Diffusion: ControlNet
    if comparison_cfg.get("controlnet", {}).get("enabled", False):
        cn_cfg = comparison_cfg["controlnet"]
        model = ControlNetColorizer(
            controlnet_model=cn_cfg.get("controlnet_model",
                                        "neurallove/controlnet-sd21-colorization-diffusers"),
            sd_model=cn_cfg.get("sd_model", "stabilityai/stable-diffusion-2-1-base"),
            num_inference_steps=cn_cfg.get("num_inference_steps", 30),
            guidance_scale=cn_cfg.get("guidance_scale", 7.5),
        )
        if model.is_available:
            models[model.name] = model

    return models
