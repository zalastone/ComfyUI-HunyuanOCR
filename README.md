# ComfyUI HunyuanOCR

Image captioning and OCR with [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR)
GGUF weights through `llama-cpp-python`. Runs on CPU or GPU, no PyTorch inference needed.

## Nodes

- **Load HunyuanOCR (GGUF)** — picks the language model and vision projector.
- **HunyuanOCR Caption** — takes an IMAGE batch and returns the generated text.

## Setup

### 1. Install `llama-cpp-python`

A recent version (0.3.30+) is required for the `mtmd` multimodal handler.

CPU only, prebuilt wheel:

```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

CUDA (needs the CUDA toolkit and a compiler):

```
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

### 2. Download the models

Place both files in `ComfyUI/models/hunyuanocr/`:

```
huggingface-cli download ggml-org/HunyuanOCR-GGUF HunyuanOCR-Q8_0.gguf mmproj-HunyuanOCR-Q8_0.gguf --local-dir ComfyUI/models/hunyuanocr
```

Alternative GGUF repository: [prithivMLmods/HunyuanOCR-1.5-GGUF-Updated](https://huggingface.co/prithivMLmods/HunyuanOCR-1.5-GGUF-Updated/tree/main)

`HunyuanOCR-bf16.gguf` + `mmproj-HunyuanOCR-bf16.gguf` is the higher quality, larger pair.
The text model and the projector must be the same quantization.

## Usage

Load an image, connect it to **HunyuanOCR Caption**, and run. Each image in the batch
produces one paragraph of text, joined by blank lines.

The `prompt` field takes any instruction. Useful defaults:

| Prompt | Result |
| --- | --- |
| `Describe this image in detail.` | General caption |
| `OCR` | Raw text extraction |
| `提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进[...]