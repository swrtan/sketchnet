"""Validated, versioned local dataset and stratified deterministic splits."""
from pathlib import Path

import numpy as np

from .config import CLASSES, DATA_NPZ, INPUT_SIZE
from .preprocessing import PREPROCESSING_VERSION


def load_dataset(path=DATA_NPZ):
    with np.load(path, allow_pickle=False) as z:
        if z['classes'].tolist() != list(CLASSES) or str(z['preprocessing_version']) != PREPROCESSING_VERSION:
            raise ValueError('Dataset mapping/preprocessing differs; regenerate dataset')
        x,y,ids = z['images'],z['labels'],z['ids']
    if x.shape != (len(y),1,INPUT_SIZE,INPUT_SIZE) or x.dtype != np.float32:
        raise ValueError('Invalid sample shape/dtype')
    if not np.isfinite(x).all() or x.min()<0 or x.max()>1:
        raise ValueError('Invalid normalized samples')
    if len(ids)!=len(y) or len(set(ids.tolist()))!=len(ids):
        raise ValueError('Duplicate or missing sample identifiers')
    if not np.issubdtype(y.dtype,np.integer) or set(y.tolist()) != set(range(len(CLASSES))):
        raise ValueError('Invalid class labels')
    if min(np.bincount(y))<10:
        raise ValueError('At least ten samples per class required')
    return x,y,ids

def save_dataset(records,path=DATA_NPZ):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,images=np.stack([r[0] for r in records]).astype(np.float32),
        labels=np.asarray([r[1] for r in records],dtype=np.int64),ids=np.asarray([r[2] for r in records],dtype=str),
        classes=np.asarray(CLASSES),preprocessing_version=PREPROCESSING_VERSION)
    temporary.replace(path)

def split_indices(labels,seed=42):
    rng=np.random.default_rng(seed)
    out={'train':[],'validation':[],'test':[]}
    for c in range(len(CLASSES)):
        ix=np.flatnonzero(labels==c); rng.shuffle(ix); n=len(ix)
        if n<10: raise ValueError('At least ten samples per class required')
        a=int(n*.7); b=int(n*.85)
        out['train'].extend(ix[:a]);out['validation'].extend(ix[a:b]);out['test'].extend(ix[b:])
    return {k:np.asarray(v,dtype=int) for k,v in out.items()}
