"""Train a full author-reference network on checked memory-mapped PF shards."""
import argparse
from collections import OrderedDict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset,DataLoader
from full_models import architecture,model_for,forward
from training_io import load_index, strict_precision


def progress(path, **record):
    if path is None: return
    record['utc'] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(path)


class ShardDataset(Dataset):
    def __init__(self,index,config,indices):
        self.index,self.config,self.indices=index,config,indices
        self.cache=OrderedDict()
        self.mean=np.asarray(config['mean'],dtype=np.float32)[:,None]
        self.std=np.asarray(config['std'],dtype=np.float32)[:,None]

    def __len__(self): return len(self.indices)

    def __getitem__(self,item):
        row=int(self.indices[item]); si=int(self.index['shard'][row]); offset=int(self.index['offset'][row])
        if si not in self.cache:
            shard=self.config['shards'][si]
            self.cache[si]=[np.load(Path(shard['directory'])/(shard['stem']+'_'+k+'.npy'),mmap_mode='r') for k in ['x','v','mask']]
            if len(self.cache)>32: self.cache.popitem(last=False)
        self.cache.move_to_end(si)
        x,v,mask=[a[offset].copy() for a in self.cache[si]]
        points=x[:2].copy()
        x=((x-self.mean)/self.std)*mask
        return x,v,mask,points,int(self.index['label'][row]),float(self.index['optimization_weight'][row]),row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('indexed',type=Path)
    ap.add_argument('models',type=Path)
    ap.add_argument('output',type=Path)
    ap.add_argument('--model',choices=['ParticleNet'],required=True)
    ap.add_argument('--profile',choices=['full','compact'],default='full')
    ap.add_argument('--progress',type=Path)
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--accumulate',type=int,default=8,help='Microbatches per optimizer step; effective default batch 128')
    args=ap.parse_args()
    if args.epochs<1 or args.batch_size<2 or args.accumulate<1: raise ValueError('Invalid training bounds')
    if not torch.cuda.is_available(): raise RuntimeError('Assigned GPU unavailable; no CPU fallback')
    strict_precision(torch)
    config,index=load_index(args.indexed)
    seed=config['seed']; torch.set_num_threads(2)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    args.output.mkdir(parents=True,exist_ok=False)
    config.update(model=args.model,profile=args.profile,architecture=architecture(args.model,args.profile),batch_size=args.batch_size,epochs=args.epochs,
        accumulation_steps=args.accumulate,effective_batch_size=args.batch_size*args.accumulate,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),
        optimization='Uniform shuffled rows; positive weighted CE; W relative HT cross sections retained; equal signal mass totals and equal class totals fitted on training only',
        checkpoint_selection='Lowest validation weighted CE; test scores computed once after checkpoint selection',
        model_sha256=hashlib.sha256((args.models/(args.model+'.py')).read_bytes()).hexdigest(),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    (args.output/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    def loader(split,shuffle=False):
        return DataLoader(ShardDataset(index,config,np.flatnonzero(index['split']==split)),
                          batch_size=args.batch_size,shuffle=shuffle,num_workers=0,drop_last=False,
                          generator=torch.Generator().manual_seed(seed))
    model=model_for(args.model,args.models,args.profile).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    trainloader=loader(0,True); validloader=loader(1)
    best=float('inf'); history=[]; started=time.time()
    progress(args.progress,stage='training_start',model=args.model,profile=args.profile,
             train_rows=len(trainloader.dataset),validation_rows=len(validloader.dataset),
             epochs=args.epochs,parameters=sum(p.numel() for p in model.parameters()),gpu=config['gpu'])
    for epoch in range(args.epochs):
        record=dict(epoch=epoch+1)
        for training,batches in [(True,trainloader),(False,validloader)]:
            model.train(training); loss_sum=0.; weight_sum=0.
            phase_start=time.time(); phase_rows=0
            with torch.set_grad_enabled(training):
                for batch_index,batch in enumerate(batches):
                    x,v,mask,points,y,w,row=[a.cuda() for a in batch]
                    if training and batch_index%args.accumulate==0:
                        optimizer.zero_grad(set_to_none=True)
                        group_rows=min(args.batch_size*args.accumulate,len(batches.dataset)-batch_index*args.batch_size)
                    logits=forward(args.model,model,x,v,mask,points)
                    losses=torch.nn.functional.cross_entropy(logits,y,reduction='none')
                    loss=(losses*w).sum()/group_rows if training else (losses*w).mean()
                    if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                    if training:
                        loss.backward()
                        if (batch_index+1)%args.accumulate==0 or batch_index+1==len(batches):
                            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                            optimizer.step()
                    loss_sum+=float((losses.detach()*w).sum()); weight_sum+=float(w.sum())
                    phase_rows+=len(y)
                    if (batch_index+1)%200==0 or batch_index+1==len(batches):
                        elapsed=time.time()-phase_start
                        progress(args.progress,stage='training' if training else 'validation',
                                 model=args.model,profile=args.profile,epoch=epoch+1,epochs=args.epochs,
                                 rows=phase_rows,total_rows=len(batches.dataset),phase_seconds=elapsed,
                                 rows_per_second=phase_rows/max(elapsed,1.e-9),
                                 elapsed_seconds=time.time()-started)
            record['train_loss' if training else 'validation_loss']=loss_sum/weight_sum
        history.append(record)
        with (args.output/'history.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(record)); writer.writeheader(); writer.writerows(history)
        if record['validation_loss']<best:
            best=record['validation_loss']
            torch.save(dict(state_dict=model.state_dict(),configuration=config,architecture=args.model,
                            best_epoch=epoch+1),args.output/'best.pt')
        print(json.dumps(record),flush=True)
        progress(args.progress,stage='epoch_complete',model=args.model,profile=args.profile,
                 epoch=epoch+1,epochs=args.epochs,elapsed_seconds=time.time()-started,
                 remaining_training_seconds=(time.time()-started)/(epoch+1)*(args.epochs-epoch-1),
                 history=history)
    checkpoint=torch.load(args.output/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    scores=[]; selected=[]
    with torch.no_grad():
        for batch in loader(2):
            x,v,mask,points,y,w,row=[a.cuda() for a in batch]
            scores.extend(torch.softmax(forward(args.model,model,x,v,mask,points),dim=1)[:,1].cpu().numpy())
            selected.extend(row.cpu().numpy())
    selected=np.asarray(selected); scores=np.asarray(scores)
    if not (np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1))): raise ValueError('Input invariant failed: np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1))')
    np.savez_compressed(args.output/'test_scores.npz',score=scores,**{k:index[k][selected] for k in
                        ['label','sample','split','physics_weight','optimization_weight','event_key','content_key','full_key','nPart']})
    summary=dict(model=args.model,profile=args.profile,best_epoch=checkpoint['best_epoch'],seconds=time.time()-started,
                 parameters=sum(p.numel() for p in model.parameters()),test_rows=len(scores),
                 auc=roc_auc_score(index['label'][selected],scores,sample_weight=index['optimization_weight'][selected]),
                 auc_note='Positive optimization weighting; not a physical signed-weight or full-background sensitivity result')
    per_mass=[]
    test_sample=index['sample'][selected]; test_label=index['label'][selected]
    for sample in sorted(set(test_sample[test_label==1])):
        take=(test_label==0)|(test_sample==sample)
        per_mass.append(dict(sample=sample,signal_test_rows=int((test_sample==sample).sum()),
                             background_test_rows=int((test_label==0).sum()),
                             auc=roc_auc_score(test_label[take],scores[take],
                                               sample_weight=index['optimization_weight'][selected][take])))
    with (args.output/'per_mass_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(per_mass[0])); writer.writeheader(); writer.writerows(per_mass)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    progress(args.progress,stage='test_complete',**summary)


if __name__=='__main__': main()
