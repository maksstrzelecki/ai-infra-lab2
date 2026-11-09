# EfficientNet-B7 inference API

Image classification using pretrained TorchVision weights and FastAPI.

## Model

Source: [official EfficientNet-B7 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b7.html).

| Property | Value |
| --- | --- |
| Weights | `EfficientNet_B7_Weights.DEFAULT` (`IMAGENET1K_V1`) |
| Parameters | 66,347,960 |
| Model input | Float tensor `(N, 3, 600, 600)` after preprocessing; one image: `(1, 3, 600, 600)` |
| Model output | Logits `(N, 1000)` for ImageNet-1K classes |
| API output | Top-k class IDs, names, and softmax probabilities |
| Example class names | `tench`, `goldfish`, `great white shark` |

The weights' preprocessing accepts PIL images or image tensors, resizes the
shorter edge to 600 using bicubic interpolation, and center-crops to 600 × 600.
Pixels are scaled to `[0, 1]`, then normalized with mean
`[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
This project converts uploaded images to RGB and uses `weights.transforms()`.
Softmax is applied across all 1,000 logits before selecting top-k results.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

Startup loads the model once and downloads approximately 255 MB of weights on
first use into the standard PyTorch cache (`~/.cache/torch/hub/checkpoints`).
Internet access is required for that first download. CPU is the default;
set `MODEL_DEVICE=cuda` or `MODEL_DEVICE=mps` for supported hardware.

Interactive API documentation: <http://127.0.0.1:8000/docs>.

```bash
curl http://127.0.0.1:8000/health
curl -X POST 'http://127.0.0.1:8000/predict?top_k=3' -F 'file=@image.jpg'
```

`POST /predict` accepts a multipart image upload up to 10 MiB and `top_k`
between 1 and 1000 (default 5). Invalid images return 400, oversized uploads
return 413, and invalid query values return 422. Each prediction contains
`class_id` (zero-based), `class_name`, and `probability` (0–1).

## Python usage

```python
from inference import load_model, predict

loaded = load_model()
print(predict("image.jpg", loaded, top_k=3))
```
