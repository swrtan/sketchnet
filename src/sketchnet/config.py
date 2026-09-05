from pathlib import Path

CLASSES = ('cat', 'dog', 'car', 'tree', 'house', 'fish', 'airplane', 'chair', 'apple', 'star')
INPUT_SIZE = 64
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'
ARTIFACTS_DIR = ROOT / 'artifacts'
CHECKPOINT_PATH = ARTIFACTS_DIR / 'best.pt'
DATA_NPZ = DATA_DIR / 'quickdraw.npz'
