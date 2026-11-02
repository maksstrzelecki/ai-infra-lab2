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
python -m pip install torch torchvision fastapi uvicorn python-multipart pillow
uvicorn server:app --host 127.0.0.1 --port 8000
```

Startup loads the model once and downloads approximately 255 MB of weights on
first use into the standard PyTorch cache (`~/.cache/torch/hub/checkpoints`).
Internet access is required for that first download. CPU is the default;
set `MODEL_DEVICE=cuda` or `MODEL_DEVICE=mps` for supported hardware.

Interactive API documentation: <http://127.0.0.1:8000/docs>.

```bash
curl http://127.0.0.1:8000/health
for i in 1 2 3 4 5; do
  curl -X POST 'http://127.0.0.1:8000/predict?top_k=3' -F 'file=@image.jpg' &
done
wait
```

The server collects exactly **5 separate requests** before running one batched
forward pass. Each request receives only its own predictions, using its own
`top_k`. There is no timeout or partial-batch inference: 1–4 requests wait until
the fifth arrives. Send requests concurrently, as in the example above.
One background worker processes batches sequentially; inference runs in a
thread so the event loop can continue accepting requests. Run with one Uvicorn
worker: multiple processes would each have their own queue and model.

`POST /predict` accepts a multipart image upload up to 10 MiB and `top_k`
between 1 and 1000 (default 5). Invalid images return 400, oversized uploads
return 413, and invalid query values return 422. Each prediction contains
`class_id` (zero-based), `class_name`, and `probability` (0–1).

## Python usage

```python
from inference import load_model, predict, predict_batch

loaded = load_model()
print(predict("image.jpg", loaded, top_k=3))
print(predict_batch(["image.jpg", "another.jpg"], loaded, top_k=3))
```

`predict_batch` accepts 1–5 images and returns predictions in input order.
The HTTP queue always waits for exactly 5 requests.

## Test

```bash
python -m pip install httpx
python -m unittest -v test_batching
```

The test uses a small substitute model without downloading pretrained weights.
