from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torchvision.models import EfficientNet_B7_Weights, efficientnet_b7


@dataclass
class LoadedModel:
    model: torch.nn.Module
    preprocess: Callable
    categories: list[str]
    device: torch.device


def optimize_model(model: torch.nn.Module, *, compile_model: bool, quantize: bool):
    if quantize:
        if next(model.parameters()).device.type != "cpu":
            raise ValueError("INT8 quantization requires MODEL_DEVICE=cpu")
        engines = torch.backends.quantized.supported_engines
        if (
            torch.backends.quantized.engine == "none"
            or torch.backends.quantized.engine not in engines
        ):
            engine = next(
                (name for name in ("x86", "fbgemm", "qnnpack") if name in engines), None
            )
            if engine is None:
                raise RuntimeError("No supported INT8 quantization engine")
            torch.backends.quantized.engine = engine
        model.classifier = torch.ao.quantization.quantize_dynamic(
            model.classifier, {torch.nn.Linear}, dtype=torch.qint8
        )
    if compile_model:
        # Keep the quantized classifier outside the compiled graph.
        model.features = torch.compile(model.features, dynamic=False)
    return model


def load_model(
    device: str = "cpu", *, compile_model: bool = True, quantize: bool = True
) -> LoadedModel:
    """Load pretrained weights (downloaded on first use) in evaluation mode."""
    weights = EfficientNet_B7_Weights.DEFAULT
    model = efficientnet_b7(weights=weights).to(device).eval()
    model = optimize_model(model, compile_model=compile_model, quantize=quantize)
    return LoadedModel(
        model, weights.transforms(), weights.meta["categories"], torch.device(device)
    )


@torch.inference_mode()
def predict(
    image: Image.Image | str | Path, loaded: LoadedModel, top_k: int = 5
) -> list[dict]:
    return predict_batch([image], loaded, top_k)[0]


@torch.inference_mode()
def predict_batch(
    images: list[Image.Image | str | Path], loaded: LoadedModel, top_k: int = 5
) -> list[list[dict]]:
    """Run one forward pass for 1–5 images, preserving their order."""
    if not 1 <= len(images) <= 5:
        raise ValueError("Batch must contain between 1 and 5 images")
    if not 1 <= top_k <= len(loaded.categories):
        raise ValueError(f"top_k must be between 1 and {len(loaded.categories)}")
    tensors = []
    for image in images:
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                rgb = source.convert("RGB")
        else:
            rgb = image.convert("RGB")
        tensors.append(loaded.preprocess(rgb))
    batch = torch.stack(tensors).to(loaded.device)
    probabilities = loaded.model(batch).softmax(dim=1)
    scores, indices = probabilities.topk(top_k)
    return [
        [
            {"class_id": index, "class_name": loaded.categories[index], "probability": score}
            for score, index in zip(image_scores, image_indices)
        ]
        for image_scores, image_indices in zip(scores.tolist(), indices.tolist())
    ]
