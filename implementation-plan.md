# SketchNet Lab delivery

1. Inspect: empty Windows workspace, Python 3.13, ~17.8 GB free; install isolated CPU runtime.
2. Data: bounded balanced Google Quick Draw simplified vectors, recognized examples; deterministic independent splits. Shared 64px rasterizer for dataset and live browser strokes.
3. Baseline: compact CNN, best validation checkpoint; inspect validation before final test. No test-driven tuning.
4. Product: static HTML/CSS/JS plus FastAPI on loopback. This avoids a frontend build chain while retaining full canvas control. No runtime LLM or external service.
5. Verify: data/preprocessing/model/API tests, measured CPU timings, actual browser drawing smoke test, evaluation artifacts.

## Boundaries
Ten-class closed-set classifier; probabilities are not calibrated certainty. No authentication, database, cloud hosting, or browser training. Activations show measured tensors, not explanations.

## Data source
https://github.com/googlecreativelab/quickdraw-dataset — Google Quick Draw, CC BY 4.0. Simplified vector data is aligned/scaled to 256px; rasterization is controlled locally for train/inference consistency.
