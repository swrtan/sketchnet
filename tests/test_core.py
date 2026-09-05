import numpy as np
import torch

from sketchnet.config import INPUT_SIZE
from sketchnet.model import SketchCNN
from sketchnet.preprocessing import preprocess_strokes


def stroke(points):
    a = np.asarray(points, dtype=float)
    return [a[:, 0].tolist(), a[:, 1].tolist()]


def test_blank_shape_and_range():
    image = preprocess_strokes([])
    assert image.shape == (1, INPUT_SIZE, INPUT_SIZE)
    assert image.dtype == torch.float32
    assert 0 <= float(image.min()) and float(image.max()) <= 1
    assert not image.any()


def test_translation_and_scale_invariance():
    base = stroke([(10, 20), (40, 60), (70, 20)])
    shifted = stroke([(110, -180), (170, -100), (230, -180)])
    scaled = stroke([(20, 40), (80, 120), (140, 40)])
    assert torch.allclose(preprocess_strokes([base]), preprocess_strokes([shifted]), atol=0.03)
    assert torch.allclose(preprocess_strokes([base]), preprocess_strokes([scaled]), atol=0.03)


def test_model_logits_shape():
    logits = SketchCNN()(torch.zeros(2, 1, INPUT_SIZE, INPUT_SIZE))
    assert logits.shape == (2, 10)
