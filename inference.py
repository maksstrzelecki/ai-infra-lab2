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


def load_model(device: str = "cpu") -> LoadedModel:
    """Load pretrained weights (downloaded on first use) in evaluation mode."""
    weights = EfficientNet_B7_Weights.DEFAULT
    model = efficientnet_b7(weights=weights).to(device).eval()
    return LoadedModel(
        model, weights.transforms(), weights.meta["categories"], torch.device(device)
    )


@torch.inference_mode()
def predict(
    image: Image.Image | str | Path, loaded: LoadedModel, top_k: int = 5
) -> list[dict]:
    """Return top-k class indices, labels, and softmax probabilities."""
    if not 1 <= top_k <= len(loaded.categories):
        raise ValueError(f"top_k must be between 1 and {len(loaded.categories)}")
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            rgb = source.convert("RGB")
    else:
        rgb = image.convert("RGB")
    batch = loaded.preprocess(rgb).unsqueeze(0).to(loaded.device)
    probabilities = loaded.model(batch).softmax(dim=1)[0]
    scores, indices = probabilities.topk(top_k)
    return [
        {"class_id": index, "class_name": loaded.categories[index], "probability": score}
        for score, index in zip(scores.tolist(), indices.tolist())
    ]
