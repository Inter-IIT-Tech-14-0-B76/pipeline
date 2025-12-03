"""
Unified Model Cache - Single source of truth for all models
Eliminates duplicate model loading across different tasks.
"""

import torch
import sys

try:
    from rembg import remove, new_session
except ImportError:
    print("[ERROR]: 'rembg' library is missing.")
    print("   Install: pip install rembg")
    print("   For GPU: pip install onnxruntime-gpu")
    sys.exit(1)


class UnifiedModelCache:
    """
    Singleton cache for ALL models used across tasks.
    Ensures each model is loaded only once.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Device settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Shared models (used by multiple tasks)
        self._rembg_session = None
        self._controlnet_pipeline = (
            None  # Shared by style_transfer_text & style_transfer_ref
        )

        # Style transfer text specific
        self._style_pipeline = None  # SDXL for generating style from text

        # Color grading
        self._mask2former_processor = None
        self._mask2former_model = None

        # AI suggestions
        self._moondream_model = None
        self._moondream_tokenizer = None

        # Prompt classifier
        self._flan_t5_model = None
        self._flan_t5_tokenizer = None

        # Track what's loaded
        self._loaded = {
            "rembg": False,
            "controlnet_pipeline": False,
            "style_pipeline": False,
            "mask2former": False,
            "moondream": False,
            "flan_t5": False,
            "sam2": False,
        }

        # SAM2 models cache (keyed by checkpoint path)
        self._sam2_models = {}

        print(f"[INFO]: UnifiedModelCache initialized on device: {self.device}")

        # Simple GPU profiler helper for measuring memory differences
        class _GPUProfiler:
            def __init__(self, label, enabled=True):
                self.label = label
                self.enabled = enabled and torch.cuda.is_available()

            def __enter__(self):
                if not self.enabled:
                    return self
                torch.cuda.synchronize()
                self.before_alloc = torch.cuda.memory_allocated(0)
                self.before_reserved = torch.cuda.memory_reserved(0)
                print(f"[GPU-PROFILER] {self.label} - before alloc={self.before_alloc/1e9:.3f}GB reserved={self.before_reserved/1e9:.3f}GB")
                return self

            def __exit__(self, exc_type, exc, tb):
                if not self.enabled:
                    return False
                torch.cuda.synchronize()
                after_alloc = torch.cuda.memory_allocated(0)
                after_reserved = torch.cuda.memory_reserved(0)
                delta_alloc = after_alloc - self.before_alloc
                delta_reserved = after_reserved - self.before_reserved
                print(f"[GPU-PROFILER] {self.label} - after  alloc={after_alloc/1e9:.3f}GB reserved={after_reserved/1e9:.3f}GB")
                print(f"[GPU-PROFILER] {self.label} - delta  alloc={delta_alloc/1e9:.3f}GB reserved={delta_reserved/1e9:.3f}GB")
                return False

        # attach profiler class to instance for use in methods
        self._GPUProfiler = _GPUProfiler

    # =========================================================================
    # REMBG (shared by style transfer tasks)
    # =========================================================================
    def get_rembg_session(self):
        """Get or create GPU-accelerated rembg session."""
        if self._rembg_session is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                self._rembg_session = new_session("u2net", providers=providers)
                print("[INFO]: rembg session created")
            except Exception as e:
                print(f"[WARN]: Could not create GPU rembg session: {e}")
                self._rembg_session = new_session("u2net")
            self._loaded["rembg"] = True
        return self._rembg_session

    # =========================================================================
    # CONTROLNET + IP-ADAPTER PIPELINE (shared by both style transfer tasks)
    # =========================================================================
    def get_controlnet_pipeline(self):
        """Get or create SDXL + ControlNet + IP-Adapter pipeline."""
        if self._controlnet_pipeline is None:
            from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

            sdxl_model = "stabilityai/stable-diffusion-xl-base-1.0"
            controlnet_model = "diffusers/controlnet-canny-sdxl-1.0"
            ip_adapter = "h94/IP-Adapter"

            print("[INFO]: Loading ControlNet model...")
            with self._GPUProfiler("ControlNet: load ControlNetModel", enabled=(self.device=="cuda")):
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_model,
                    torch_dtype=self.dtype,
                )

            print("[INFO]: Loading SDXL+ControlNet pipeline...")
            with self._GPUProfiler("ControlNet: load Pipeline", enabled=(self.device=="cuda")):
                self._controlnet_pipeline = (
                    StableDiffusionXLControlNetPipeline.from_pretrained(
                        sdxl_model,
                        controlnet=controlnet,
                        torch_dtype=self.dtype,
                    )
                )

            if self.device == "cuda":
                with self._GPUProfiler("ControlNet: move pipeline to CUDA", enabled=True):
                    self._controlnet_pipeline = self._controlnet_pipeline.to("cuda")
                    self._controlnet_pipeline.enable_vae_slicing()
                    self._controlnet_pipeline.enable_vae_tiling()

            # Load IP Adapter
            try:
                print("[INFO]: Loading IP-Adapter...")
                self._controlnet_pipeline.load_ip_adapter(
                    ip_adapter,
                    subfolder="sdxl_models",
                    weight_name="ip-adapter_sdxl.bin",
                )
                self._controlnet_pipeline.set_ip_adapter_scale(0.8)
            except Exception as e:
                print(f"[WARN]: IP-Adapter load warning: {e}")

            self._loaded["controlnet_pipeline"] = True
            print("[INFO]: ControlNet + IP-Adapter pipeline loaded and cached")

        return self._controlnet_pipeline

    # =========================================================================
    # SDXL STYLE PIPELINE (for generating style from text)
    # =========================================================================
    def get_style_pipeline(self):
        """Get or create SDXL pipeline for style generation from text."""
        if self._style_pipeline is None:
            from diffusers import StableDiffusionXLPipeline

            sdxl_model = "stabilityai/stable-diffusion-xl-base-1.0"
            print("[INFO]: Loading SDXL style pipeline...")

            with self._GPUProfiler("SDXL Style: from_pretrained", enabled=(self.device=="cuda")):
                self._style_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    sdxl_model,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                )

            if self.device == "cuda":
                with self._GPUProfiler("SDXL Style: move to CUDA", enabled=True):
                    self._style_pipeline = self._style_pipeline.to("cuda")
                    self._style_pipeline.enable_vae_slicing()
                    self._style_pipeline.enable_vae_tiling()

            self._loaded["style_pipeline"] = True
            print("[INFO]: SDXL style pipeline loaded and cached")

        return self._style_pipeline

    # =========================================================================
    # MASK2FORMER (for color grading)
    # =========================================================================
    def get_mask2former(self):
        """Get or create Mask2Former model for segmentation."""
        if self._mask2former_model is None:
            from transformers import (
                Mask2FormerImageProcessor,
                Mask2FormerForUniversalSegmentation,
            )

            model_name = "facebook/mask2former-swin-base-coco-panoptic"
            print("[INFO]: Loading Mask2Former model...")

            self._mask2former_processor = Mask2FormerImageProcessor.from_pretrained(
                model_name
            )
            self._mask2former_model = (
                Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(
                    self.device
                )
            )
            self._mask2former_model.eval()

            self._loaded["mask2former"] = True
            print("[INFO]: Mask2Former model loaded and cached")

        return self._mask2former_processor, self._mask2former_model

    # =========================================================================
    # MOONDREAM (for AI suggestions)
    # =========================================================================
    def get_moondream(self):
        """Get or create moondream2 model for image analysis."""
        if self._moondream_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "vikhyatk/moondream2"
            print("[INFO]: Loading moondream2 model...")

            self._moondream_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._moondream_model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True
            ).to(self.device)

            self._loaded["moondream"] = True
            print("[INFO]: moondream2 model loaded and cached")

        return self._moondream_model, self._moondream_tokenizer

    # =========================================================================
    # FLAN-T5 (for prompt classification)
    # =========================================================================
    def get_flan_t5(self):
        """Get or create FLAN-T5 model for prompt classification."""
        if self._flan_t5_model is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            model_name = "google/flan-t5-base"
            print("[INFO]: Loading FLAN-T5 model...")

            self._flan_t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Keep FLAN-T5 on CPU to save GPU memory (it's small and fast enough)
            self._flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._flan_t5_model.eval()

            self._loaded["flan_t5"] = True
            print("[INFO]: FLAN-T5 model loaded and cached (on CPU)")

        return self._flan_t5_model, self._flan_t5_tokenizer

    # =========================================================================
    # SAM2 (segment-anything v2)
    # =========================================================================
    def get_sam2(self, config_path: str, checkpoint_path: str):
        """Get or create a SAM2 model instance (cached by checkpoint path).

        Returns the built model object. Building is deferred to the
        segmentation_sam2.sam2.build_sam module to avoid duplicating logic.
        """
        key = checkpoint_path or "default"
        if key not in self._sam2_models:
            try:
                # Ensure the bundled `sam2` package is importable as top-level `sam2`.
                # This helps Hydra resolve class-path strings like
                # 'sam2.modeling.backbones.hieradet.Hiera'. Compute project root
                # relative to this file.
                import os
                import importlib

                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                bundled_sam2 = os.path.join(root_dir, "segmentation_sam2", "sam2")
                if os.path.isdir(bundled_sam2) and bundled_sam2 not in sys.path:
                    sys.path.insert(0, bundled_sam2)

                # Try importing builder from top-level sam2 first (preferred),
                # otherwise fall back to package-relative import.
                build_sam2 = None
                try:
                    mod = importlib.import_module("sam2.build_sam")
                    build_sam2 = getattr(mod, "build_sam2")
                except Exception:
                    try:
                        mod = importlib.import_module("segmentation_sam2.sam2.build_sam")
                        build_sam2 = getattr(mod, "build_sam2")
                    except Exception as e:
                        # Print debug information for easier diagnosis
                        print(f"[ERROR]: Could not import sam2.build_sam: {e}")
                        print("[DEBUG] sys.path (first 10):")
                        for p in sys.path[:10]:
                            print("  ", p)
                        raise

                print(f"[INFO]: Building SAM2 model: cfg={config_path} ckpt={checkpoint_path}")
                # Resolve checkpoint path: allow passing bare filename and try
                # common locations (workspace root, segmentation_sam2 folder,
                # segmentation_sam2/sam2). This helps when users place the
                # checkpoint next to the bundled package.
                import os

                resolved_ckpt = checkpoint_path
                tried = []
                if checkpoint_path:
                    if os.path.isabs(checkpoint_path) and os.path.exists(checkpoint_path):
                        resolved_ckpt = checkpoint_path
                    else:
                        # paths to try
                        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                        candidates = [
                            checkpoint_path,
                            os.path.join(root_dir, checkpoint_path),
                            os.path.join(root_dir, "segmentation_sam2", checkpoint_path),
                            os.path.join(root_dir, "segmentation_sam2", "sam2", checkpoint_path),
                        ]
                        for c in candidates:
                            tried.append(c)
                            if os.path.exists(c):
                                resolved_ckpt = c
                                break

                # If checkpoint is expected but not found, raise clear error
                if checkpoint_path and not os.path.exists(resolved_ckpt):
                    msg = (
                        f"SAM2 checkpoint not found: '{checkpoint_path}'.\n"
                        f"Tried locations:\n  " + "\n  ".join(tried)
                    )
                    print("[ERROR]: " + msg)
                    raise FileNotFoundError(msg)

                # Report which checkpoint path will be used (helps debugging)
                print(f"[INFO]: SAM2 resolved checkpoint path: {resolved_ckpt}")

                with self._GPUProfiler("SAM2: build_sam2 (from_pretrained)", enabled=(self.device=="cuda")):
                    model = build_sam2(
                        config_file=config_path,
                        ckpt_path=resolved_ckpt,
                        device=self.device,
                        mode="eval",
                    )

                # Move model to device if needed
                try:
                    if self.device == "cuda":
                        with self._GPUProfiler("SAM2: move model to CUDA", enabled=True):
                            model = model.to(self.device)
                except Exception:
                    pass

                try:
                    model.eval()
                except Exception:
                    pass

                self._sam2_models[key] = model
                self._loaded["sam2"] = True
                print("[INFO]: SAM2 model built and cached")
            except Exception as e:
                print(f"[ERROR]: Failed to build SAM2 model: {e}")
                raise

        return self._sam2_models[key]

    def preload_sam(self, config_path: str = "configs/sam2.1/sam2.1_hiera_l.yaml", checkpoint_path: str = "sam2.1_hiera_large.pt"):
        """Preload a default SAM2 model into the cache."""
        print("[INFO]: Preloading SAM2 model...")
        self.get_sam2(config_path, checkpoint_path)
        print("[SUCCESS]: SAM2 model preloaded")

    # =========================================================================
    # PRELOAD METHODS
    # =========================================================================
    def preload_style_transfer(self):
        """Preload models needed for style transfer (both text and ref)."""
        print("[INFO]: Preloading style transfer models...")
        self.get_rembg_session()
        self.get_style_pipeline()
        self.get_controlnet_pipeline()
        print("[SUCCESS]: Style transfer models ready!")

    def preload_color_grading(self):
        """Preload models needed for color grading."""
        print("[INFO]: Preloading color grading models...")
        self.get_mask2former()
        print("[SUCCESS]: Color grading models ready!")

    def preload_ai_suggestions(self):
        """Preload models needed for AI suggestions."""
        print("[INFO]: Preloading AI suggestions models...")
        self.get_moondream()
        print("[SUCCESS]: AI suggestions models ready!")

    def preload_prompt_classifier(self):
        """Preload models needed for prompt classification."""
        print("[INFO]: Preloading prompt classifier models...")
        self.get_flan_t5()
        print("[SUCCESS]: Prompt classifier models ready!")

    def preload_all(self):
        """Preload all models."""
        print("=" * 60)
        print("[INFO]: Preloading ALL models...")
        print(f"[INFO]: Device: {self.device}")
        if self.device == "cuda":
            print(f"[INFO]: GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"[INFO]: VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        print("=" * 60)

        self.preload_style_transfer()
        self.preload_color_grading()
        self.preload_ai_suggestions()
        self.preload_prompt_classifier()
        # Preload SAM2 model as well (optional, may be large)
        try:
            self.preload_sam()
        except Exception as e:
            print(f"[WARN]: Could not preload SAM2: {e}")

        print("\n" + "=" * 60)
        print("[SUCCESS]: ALL MODELS LOADED!")
        print("=" * 60)

    # =========================================================================
    # STATUS & CLEANUP
    # =========================================================================
    def get_status(self):
        """Get status of all models."""
        return {
            "device": self.device,
            "models_loaded": self._loaded.copy(),
        }

    def is_loaded(self, model_name: str) -> bool:
        """Check if a specific model is loaded."""
        return self._loaded.get(model_name, False)

    def all_loaded(self) -> bool:
        """Check if all models are loaded."""
        return all(self._loaded.values())

    def clear_cache(self):
        """Clear all cached models to free memory."""
        self._rembg_session = None
        self._controlnet_pipeline = None
        self._style_pipeline = None
        self._mask2former_processor = None
        self._mask2former_model = None
        self._moondream_model = None
        self._moondream_tokenizer = None
        self._flan_t5_model = None
        self._flan_t5_tokenizer = None

        for key in self._loaded:
            self._loaded[key] = False

        if self.device == "cuda":
            torch.cuda.empty_cache()

        print("[INFO]: All model caches cleared")


# Global cache instance
_cache = None


def get_model_cache() -> UnifiedModelCache:
    """Get the global unified model cache."""
    global _cache
    if _cache is None:
        _cache = UnifiedModelCache()
    return _cache
