"""Download only a bounded prefix of recognized Quick Draw vectors per class."""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sketchnet.config import CLASSES, DATA_DIR, DATA_NPZ
from sketchnet.data import save_dataset
from sketchnet.preprocessing import preprocess_strokes


def download(sample_count=1500,out=DATA_NPZ):
    if sample_count<10: raise ValueError('Use at least ten samples/class')
    cache=DATA_DIR/'vectors';cache.mkdir(parents=True,exist_ok=True)
    records=[]; seen=set()
    for label,name in enumerate(CLASSES):
        path=cache/f'{name}-{sample_count}.ndjson'
        if not path.exists():
            temporary=path.with_suffix('.partial')
            url=f'https://storage.googleapis.com/quickdraw_dataset/full/simplified/{name}.ndjson'
            count=0; local_seen=set()
            try:
                with urllib.request.urlopen(url,timeout=120) as response,temporary.open('w',encoding='utf-8') as f:
                    for line in response:
                        item=json.loads(line)
                        key=str(item['key_id'])
                        if not item.get('recognized',False) or key in local_seen: continue
                        local_seen.add(key);f.write(json.dumps({'key_id':key,'drawing':item['drawing']})+'\n');count+=1
                        if count>=sample_count: break
                if count<sample_count: raise ValueError(f'Only {count} samples available for {name}')
                temporary.replace(path)
            except Exception as exc:
                raise RuntimeError(f'Download failed for {name}; completed class caches retained. Retry command. {exc}') from exc
        count=0
        with path.open(encoding='utf-8') as f:
            for line in f:
                item=json.loads(line);key=str(item['key_id'])
                if key in seen: raise ValueError(f'Duplicate dataset ID {key}')
                seen.add(key);records.append((preprocess_strokes(item['drawing']).numpy(),label,key));count+=1
        if count!=sample_count: raise ValueError(f'Cache count mismatch: {path}')
        print(f'{name}: {count} samples',flush=True)
    save_dataset(records,out)
    print(f'Saved {len(records)} balanced samples to {out}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sample-count',type=int,default=1500);p.add_argument('--out',default=str(DATA_NPZ))
    download(**vars(p.parse_args()))
