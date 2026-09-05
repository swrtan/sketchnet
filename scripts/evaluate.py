import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from sketchnet.config import ARTIFACTS_DIR, CHECKPOINT_PATH, CLASSES, DATA_NPZ
from sketchnet.data import load_dataset, split_indices
from sketchnet.model import SketchCNN
from sketchnet.preprocessing import PREPROCESSING_VERSION


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def main(a):
    torch.set_num_threads(4); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    x,y,_=load_dataset(a.data); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    actual=digest(a.data)
    if ck.get('preprocessing_version') != PREPROCESSING_VERSION or ck.get('input_size') != 64: raise ValueError('Incompatible preprocessing or input size')
    if ck.get('dataset_sha256') != actual: raise ValueError('checkpoint dataset SHA256 does not match --data')
    if ck.get('seed') != a.seed: raise ValueError(f"checkpoint seed {ck.get('seed')} does not match --seed {a.seed}")
    if tuple(ck.get('classes',())) != tuple(CLASSES): raise ValueError('checkpoint classes are incompatible')
    model=SketchCNN(len(CLASSES)); model.load_state_dict(ck['model_state']); model.eval(); s=split_indices(y,a.seed)[a.split]
    preds=[]; probs=[]
    with torch.no_grad():
        for xb,_ in DataLoader(TensorDataset(torch.tensor(x[s]),torch.tensor(y[s])),a.batch):
            z=model(xb); probs.append(torch.softmax(z,1).numpy()); preds.append(z.argmax(1).numpy())
    probs=np.concatenate(probs); pred=np.concatenate(preds); true=y[s]; cm=np.zeros((len(CLASSES),len(CLASSES)),dtype=int)
    for t,p in zip(true,pred): cm[t,p]+=1
    per={}
    for i,name in enumerate(CLASSES):
        tp=cm[i,i]; support=cm[i].sum(); fp=cm[:,i].sum()-tp
        precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/support if support else 0.; f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        per[name]={'accuracy':float(recall),'precision':precision,'recall':recall,'f1':f1,'support':int(support)}
    metrics={'split':a.split,'seed':a.seed,'dataset_sha256':actual,'accuracy':float((pred==true).mean()),'per_class':per,'confusion_matrix':cm.tolist(),'confidence_histogram':np.histogram(probs.max(1),bins=10,range=(0,1))[0].tolist()}
    (out/f'metrics_{a.split}.json').write_text(json.dumps(metrics,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(cm,cmap='Blues'); fig.colorbar(im,ax=ax); ax.set(xticks=range(10),yticks=range(10),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='True',title=f'Confusion matrix ({a.split})'); plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
    for i in range(10):
        for j in range(10): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout(); fig.savefig(out/f'confusion_{a.split}.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.hist(probs.max(1),bins=10,range=(0,1)); ax.set(xlabel='Confidence',ylabel='Count',title=f'Confidence ({a.split})'); fig.tight_layout(); fig.savefig(out/f'confidence_{a.split}.png',dpi=140); plt.close(fig)
    errors=np.concatenate([np.flatnonzero((pred!=true) & (true==i))[:3] for i in range(len(CLASSES))])[:25]; fig,axes=plt.subplots(5,5,figsize=(10,10));
    for ax in axes.flat: ax.axis('off')
    for ax,k in zip(axes.flat,errors): ax.imshow(x[s[k],0],cmap='gray'); ax.set_title(f'{CLASSES[true[k]]} / {CLASSES[pred[k]]}',fontsize=8); ax.axis('off')
    fig.suptitle('Representative mistakes'); fig.tight_layout(); fig.savefig(out/f'mistakes_{a.split}.png',dpi=140); plt.close(fig)
    hist=Path(a.history)
    if hist.exists():
        rows=json.loads(hist.read_text()); fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].plot([r['epoch'] for r in rows],[r['train_loss'] for r in rows],label='train'); ax[0].plot([r['epoch'] for r in rows],[r['val_loss'] for r in rows],label='validation'); ax[0].set_title('Loss'); ax[1].plot([r['epoch'] for r in rows],[r['train_accuracy'] for r in rows],label='train'); ax[1].plot([r['epoch'] for r in rows],[r['val_accuracy'] for r in rows],label='validation'); ax[1].set_title('Accuracy'); [q.legend() for q in ax]; fig.tight_layout(); fig.savefig(out/'curves.png',dpi=140); plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--split',choices=['validation','test'],default='validation'); p.add_argument('--data',default=str(DATA_NPZ)); p.add_argument('--checkpoint',default=str(CHECKPOINT_PATH)); p.add_argument('--out',default=str(ARTIFACTS_DIR)); p.add_argument('--history',default=str(ARTIFACTS_DIR/'history.json')); p.add_argument('--batch',type=int,default=128); p.add_argument('--seed',type=int,default=42); main(p.parse_args())
