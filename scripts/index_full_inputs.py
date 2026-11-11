"""Index explicit tensor shards, reject leakage, fit TRAIN-only preprocessing."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import os
import numpy as np
from ml_trial_common import FEATURES, training_weights

SEED=12345


def split_for(content_key,seed=SEED):
    key=content_key.decode() if isinstance(content_key,bytes) else str(content_key)
    number=int(hashlib.sha256((str(seed)+'|'+key).encode()).hexdigest()[:16],16)/float(2**64)
    return 0 if number<.70 else 1 if number<.85 else 2


def require_unique(values,name):
    if len(np.unique(values))!=len(values):
        raise ValueError('Repeated '+name+'; resolve source duplication before training')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('prepared',type=Path)
    ap.add_argument('output',type=Path)
    ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--lumi-pb',type=float,default=1.,help='Common scale cancels in optimization; physical plots require calibrated exposure')
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    paths=[args.prepared/'manifest.json']
    sources={}
    shards=[]
    cap=None
    for path in paths:
        record=json.loads(path.read_text())
        if cap is None: cap=record['cap']
        if not (cap==record['cap']): raise ValueError("Input invariant failed: cap==record['cap']")
        for row in record['sources']:
            if row['key'] in sources: raise ValueError('Repeated source key')
            sources[row['key']]=row
        for row in record['shards']:
            shards.append(dict(row,directory=str(path.parent.resolve())))
    if not sources or not shards: raise ValueError('Empty prepared inputs')
    if {r['label'] for r in sources.values()} != {0,1}:
        raise ValueError('Both signal and background are required')
    require_unique(np.array([r['source'] for r in sources.values()]),'source identity')
    if not np.isfinite(args.lumi_pb) or args.lumi_pb <= 0:
        raise ValueError('Positive finite luminosity required')
    norm=defaultdict(float)
    conventions={}
    for row in sources.values():
        convention=(row['label'],row['xsec_pb'],row['factor'])
        if row['sample'] in conventions and conventions[row['sample']]!=convention:
            raise ValueError('Inconsistent cross section or factor within '+row['sample'])
        conventions[row['sample']]=convention
        norm[row['sample']]+=row['sum_genweights']
    for value in norm.values():
        if not np.isfinite(value) or value<=0: raise ValueError('Invalid normalization denominator')
    meta=[]; samples=[]; shard_ids=[]; offsets=[]; labels=[]; physics=[]
    for si,shard in enumerate(shards):
        directory=Path(shard['directory'])
        for key,checksum in shard['sha256'].items():
            if not (hashlib.sha256((directory/(shard['stem']+'_'+key+'.npy')).read_bytes()).hexdigest()==checksum): raise ValueError("Input invariant failed: hashlib.sha256((directory/(shard['stem']+'_'+key+'.npy')).read_bytes()).hexdigest()==checksum")
        a=np.load(directory/(shard['stem']+'_meta.npy'),allow_pickle=False)
        if not (len(a)==shard['rows']): raise ValueError("Input invariant failed: len(a)==shard['rows']")
        source=sources[shard['source_key']]
        meta.append(a); samples.extend([source['sample']]*len(a)); labels.extend([source['label']]*len(a))
        shard_ids.extend([si]*len(a)); offsets.extend(range(len(a)))
        physics.extend(a['genWeight']*args.lumi_pb*source['xsec_pb']*source['factor']/norm[source['sample']])
    meta=np.concatenate(meta)
    for name in ['event_key','content_key','full_key']: require_unique(meta[name],name)
    split=np.array([split_for(k,args.seed) for k in meta['content_key']],dtype=np.int8)
    y=np.array(labels,dtype=np.int64); sample=np.array(samples)
    for name in np.unique(sample):
        if min(np.sum((sample==name)&(split==s)) for s in range(3))<10:
            raise ValueError('Fewer than ten rows per partition for '+name)
    physics=np.asarray(physics,dtype=np.float64)
    if not np.isfinite(physics).all(): raise ValueError('Nonfinite event weights')
    weight=training_weights(y,sample,physics,split==0).astype(np.float32)
    sums=np.zeros(len(FEATURES)); squares=sums.copy(); count=0
    cursor=0
    for shard in shards:
        directory=Path(shard['directory']); n=shard['rows']
        take=split[cursor:cursor+n]==0; cursor+=n
        x=np.load(directory/(shard['stem']+'_x.npy'),mmap_mode='r')[take].astype(np.float64)
        mask=np.load(directory/(shard['stem']+'_mask.npy'),mmap_mode='r')[take,0]
        count+=int(mask.sum())
        for f in range(len(FEATURES)):
            values=x[:,f][mask]; sums[f]+=values.sum(); squares[f]+=(values*values).sum()
    mean=sums/count; std=np.maximum(np.sqrt(np.maximum(squares/count-mean*mean,0)),.001)
    np.savez(args.output/'index.npz',shard=np.array(shard_ids,dtype=np.int32),offset=np.array(offsets,dtype=np.int32),
             label=y,sample=sample,split=split,physics_weight=physics,optimization_weight=weight,**{n:meta[n] for n in meta.dtype.names})
    for shard in shards:
        shard['directory']=os.path.relpath(shard['directory'],args.output.resolve())
    report=dict(cap=cap,features=FEATURES,mean=mean.tolist(),std=std.tolist(),seed=args.seed,shards=shards,
                sources=list(sources.values()),sum_genweights=dict(norm),lumi_pb=args.lumi_pb,
                signal_exclusions=[],
                index_sha256=hashlib.sha256((args.output/'index.npz').read_bytes()).hexdigest(),
                partition='SHA256(seed|full capped-input fingerprint), 70% train, 15% validation, 15% test; exact duplicate inputs rejected',
                constituents=dict(mean=float(meta['nPart'].mean()),median=float(np.median(meta['nPart'])),
                                  p95=float(np.quantile(meta['nPart'],.95)),fraction_above_cap=float(np.mean(meta['nPart']>cap))),
                samples={name:dict(train=int(((sample==name)&(split==0)).sum()),validation=int(((sample==name)&(split==1)).sum()),
                                   test=int(((sample==name)&(split==2)).sum())) for name in np.unique(sample)})
    (args.output/'configuration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['shards','sources']},indent=2))


if __name__=='__main__': main()
