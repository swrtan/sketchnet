import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
from sketchnet.config import ARTIFACTS_DIR, CHECKPOINT_PATH, CLASSES, DATA_NPZ, INPUT_SIZE
from sketchnet.data import load_dataset, split_indices
from sketchnet.model import SketchCNN
from sketchnet.preprocessing import PREPROCESSING_VERSION


def device(name):
    ok_mps=getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()
    if name=='auto': name='cuda' if torch.cuda.is_available() else ('mps' if ok_mps else 'cpu')
    if name=='cuda' and not torch.cuda.is_available(): print('warning: CUDA unavailable; using CPU',flush=True); name='cpu'
    if name=='mps' and not ok_mps: print('warning: MPS unavailable; using CPU',flush=True); name='cpu'
    return torch.device(name)
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def main(a):
    if min(a.epochs,a.batch)<=0 or a.lr<=0: raise ValueError('epochs, batch and lr must be positive')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(4); dev=device(a.device)
    x,y,_=load_dataset(a.data); s=split_indices(y,a.seed); model=SketchCNN(len(CLASSES)).to(dev); opt=torch.optim.Adam(model.parameters(),lr=a.lr); criterion=nn.CrossEntropyLoss(); hist=[]; best=float('inf')
    def loader(k,shuffle=False): return DataLoader(TensorDataset(torch.tensor(x[s[k]]),torch.tensor(y[s[k]],dtype=torch.long)),a.batch,shuffle=shuffle)
    Path(a.checkpoint).parent.mkdir(parents=True,exist_ok=True); Path(a.history).parent.mkdir(parents=True,exist_ok=True)
    for ep in range(1,a.epochs+1):
        model.train(); total=correct=n=0
        for xb,yb in loader('train',True):
            xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(); z=model(xb); loss=criterion(z,yb); loss.backward(); opt.step(); total+=loss.item()*len(yb); correct+=(z.argmax(1)==yb).sum().item(); n+=len(yb)
        model.eval(); vl=vc=vn=0
        with torch.no_grad():
            for xb,yb in loader('validation'):
                z=model(xb.to(dev)); vl+=criterion(z,yb.to(dev)).item()*len(yb); vc+=(z.argmax(1).cpu()==yb).sum().item(); vn+=len(yb)
        row={'epoch':ep,'train_loss':total/n,'train_accuracy':correct/n,'val_loss':vl/vn,'val_accuracy':vc/vn}; hist.append(row); Path(a.history).write_text(json.dumps(hist,indent=2)); print(f"epoch {ep}/{a.epochs} val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.3f}",flush=True)
        if row['val_loss']<best:
            best=row['val_loss']; torch.save({'model_state':model.state_dict(),'classes':list(CLASSES),'input_size':INPUT_SIZE,'seed':a.seed,'dataset_sha256':digest(a.data),'architecture':'SketchCNN','preprocessing_version':PREPROCESSING_VERSION,'best_epoch':ep},a.checkpoint)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default=str(DATA_NPZ)); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch',type=int,default=64); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto'); p.add_argument('--seed',type=int,default=42); p.add_argument('--checkpoint',default=str(CHECKPOINT_PATH)); p.add_argument('--history',default=str(ARTIFACTS_DIR/'history.json')); main(p.parse_args())
