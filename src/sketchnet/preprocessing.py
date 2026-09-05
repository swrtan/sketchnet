"""Canonical white-ink rasterizer; bbox-centered, padded, scale-independent width."""
import numpy as np
import torch
from PIL import Image, ImageDraw

from .config import INPUT_SIZE

PREPROCESSING_VERSION = 'centered-v1'

def preprocess_strokes(strokes, size=INPUT_SIZE):
    if size != INPUT_SIZE:
        raise ValueError(f'Model requires {INPUT_SIZE}px input')
    valid = []
    for stroke in strokes:
        if len(stroke) != 2 or len(stroke[0]) != len(stroke[1]) or not stroke[0]:
            raise ValueError('Stroke must contain equal nonempty x/y arrays')
        points = np.asarray(stroke, dtype=np.float64).T
        if not np.isfinite(points).all():
            raise ValueError('Non-finite stroke coordinates')
        valid.append(points)
    if not valid:
        return torch.zeros((1, size, size), dtype=torch.float32)
    all_points = np.concatenate(valid)
    lo, hi = all_points.min(0), all_points.max(0)
    span = max(float((hi-lo).max()), 1e-8)
    scale = (size-12)/span
    center = (lo+hi)/2
    canvas = Image.new('L', (size*4, size*4), 0)
    draw = ImageDraw.Draw(canvas)
    for stroke in valid:
        points = np.rint(((stroke-center)*scale+size/2)*4).astype(int)
        pts = [tuple(p) for p in points]
        if len(pts)>1:
            draw.line(pts, fill=255, width=8, joint='curve')
        for x,y in (pts[0],pts[-1]):
            draw.ellipse((x-4,y-4,x+4,y+4),fill=255)
    canvas = canvas.resize((size,size),Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(canvas,dtype=np.float32)/255).unsqueeze(0)
