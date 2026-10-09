import base64
import io
import os

import comfy.model_management as model_management
import folder_paths
import torch
from PIL import Image

folder_paths.folder_names_and_paths["hunyuanocr"] = (
    [os.path.join(folder_paths.models_dir, "hunyuanocr")],
    {".gguf"},
)

DEFAULT_PROMPT = "Describe this image in detail."


def _list_gguf():
    files = folder_paths.get_filename_list("hunyuanocr")
    mmprojs, models = [], []
    for f in files:
        if os.path.basename(f).lower().startswith("mmproj"):
            mmprojs.append(f)
        else:
            models.append(f)
    return models or ["None"], mmprojs or ["None"]


def _encode_image(image: torch.Tensor) -> str:
    arr = (image.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class HunyuanOCR:
    def __init__(self, model_path, mmproj_path, ctx_size, gpu_layers):
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import MTMDChatHandler
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is required. Install it into the ComfyUI python environment "
                "(see ComfyUI-HunyuanOCR/README.md)."
            )

        use_gpu = gpu_layers != 0
        model_management.unload_all_models()
        self.llm = Llama(
            model_path=model_path,
            chat_handler=MTMDChatHandler(clip_model_path=mmproj_path, use_gpu=use_gpu),
            n_ctx=ctx_size,
            n_gpu_layers=gpu_layers,
            flash_attn=use_gpu,
            verbose=False,
        )

    def caption(self, prompt, images, max_tokens, temperature, repeat_penalty):
        results = []
        for image in images:
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": _encode_image(image)}},
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                repeat_penalty=repeat_penalty,
            )
            results.append(response["choices"][0]["message"]["content"])
        model_management.soft_empty_cache()
        return results


class LoadHunyuanOCR:
    @classmethod
    def INPUT_TYPES(cls):
        models, mmprojs = _list_gguf()
        return {
            "required": {
                "model_name": (models, {"tooltip": "HunyuanOCR language model GGUF, from ggml-org/HunyuanOCR-GGUF"}),
                "mmproj_name": (mmprojs, {"tooltip": "Matching vision projector GGUF"}),
                "device": (["gpu", "cpu"], {"tooltip": "Offload the model to the GPU or keep it on the CPU"}),
                "context_size": ("INT", {"default": 4096, "min": 1024, "max": 32768, "step": 512}),
            }
        }

    RETURN_TYPES = ("HUNYUAN_OCR",)
    RETURN_NAMES = ("hunyuan_ocr",)
    FUNCTION = "load"
    CATEGORY = "HunyuanOCR"

    def load(self, model_name, mmproj_name, device, context_size):
        if model_name == "None" or mmproj_name == "None":
            raise RuntimeError(
                f"No GGUF files found in {folder_paths.get_folder_paths('hunyuanocr')[0]}. "
                "See ComfyUI-HunyuanOCR/README.md for the download commands."
            )
        return (HunyuanOCR(
            folder_paths.get_full_path_or_raise("hunyuanocr", model_name),
            folder_paths.get_full_path_or_raise("hunyuanocr", mmproj_name),
            context_size,
            -1 if device == "gpu" else 0,
        ),)


class HunyuanOCRCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hunyuan_ocr": ("HUNYUAN_OCR",),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_PROMPT,
                        "tooltip": 'Instruction for the model. Use "OCR" to extract raw text.',
                    },
                ),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 32768, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "repeat_penalty": ("FLOAT", {"default": 1.08, "min": 1.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "caption"
    CATEGORY = "HunyuanOCR"
    OUTPUT_NODE = True

    def caption(self, hunyuan_ocr, image, prompt, max_tokens, temperature, repeat_penalty):
        results = hunyuan_ocr.caption(prompt, image, max_tokens, temperature, repeat_penalty)
        return ("\n\n".join(results),)


NODE_CLASS_MAPPINGS = {
    "LoadHunyuanOCR": LoadHunyuanOCR,
    "HunyuanOCRCaption": HunyuanOCRCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHunyuanOCR": "Load HunyuanOCR (GGUF)",
    "HunyuanOCRCaption": "HunyuanOCR Caption",
}