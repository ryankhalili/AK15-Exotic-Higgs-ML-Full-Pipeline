"""Full ParT balanced training with update-scaled warmup and validation gates."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Sampler
from warmup import schedule, learning_rate, choose_validation

from training_io import load_index, strict_precision


class BalancedBatches(Sampler):
    def __init__(self, labels, weights, batch_size, seed):
        labels, weights = np.asarray(labels), np.asarray(weights)
        if (batch_size < 2 or batch_size % 2 or labels.ndim != 1
                or weights.shape != labels.shape or not np.isin(labels, [0,1]).all()
                or not np.isfinite(weights).all() or np.any(weights<=0)
                or not all(np.any(labels==i) for i in (0,1))):
            raise ValueError('Batch, labels or weights')
        self.groups=[np.flatnonzero(labels==i) for i in [0,1]]
        self.probabilities=[weights[g].astype(float)/weights[g].sum(dtype=float) for g in self.groups]
        self.size=batch_size;self.seed=seed;self.epoch=0
        self.batches=(len(labels)+batch_size-1)//batch_size
    def __len__(self): return self.batches
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);self.epoch+=1
        for _ in range(self.batches):
            batch=np.concatenate([rng.choice(g,self.size//2,p=p) for g,p in zip(self.groups,self.probabilities)])
            rng.shuffle(batch);yield batch.tolist()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('indexed',type=Path);ap.add_argument('models',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--model',default='ParticleTransformer',choices=['ParticleTransformer'])
    ap.add_argument('--profile',default='full',choices=['full'])
    ap.add_argument('--epochs',type=int,default=6);ap.add_argument('--batch-size',type=int,default=16,choices=[16])
    ap.add_argument('--progress',type=Path);ap.add_argument('--learning-rate',type=float,default=.001)
    ap.add_argument('--warmup-epochs',type=float,default=1.)
    args=ap.parse_args()
    schedule(2, args.batch_size, 8, args.epochs, args.warmup_epochs, args.learning_rate)
    here = Path(__file__).resolve().parent
    import train_full_models as common
    import full_models as models
    if any(Path(m.__file__).resolve().parent != here for m in (common, models)):
        raise ValueError('Unexpected frozen helper import')
    ShardDataset, progress = common.ShardDataset, common.progress
    model_for, architecture, forward = models.model_for, models.architecture, models.forward
    if not torch.cuda.is_available():raise RuntimeError('Assigned GPU unavailable')
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '1':
        raise ValueError('TF32 force-enable override')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    strict_precision(torch)
    config,index=load_index(args.indexed)
    if (not np.isin(index['split'], [0,1,2]).all()
            or not np.isin(index['label'], [0,1]).all()
            or not np.isfinite(index['optimization_weight']).all()
            or np.any(index['optimization_weight']<=0)):
        raise ValueError('Invalid persisted membership/weights')
    train=np.flatnonzero(index['split']==0);seed=config['seed'];accumulate=8
    warmup = schedule(len(train), args.batch_size, accumulate, args.epochs,
                      args.warmup_epochs, args.learning_rate)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    sampler=BalancedBatches(index['label'][train],index['optimization_weight'][train],args.batch_size,seed)
    training=DataLoader(ShardDataset(index,config,train),batch_sampler=sampler,num_workers=0)
    def loader(split):return DataLoader(ShardDataset(index,config,np.flatnonzero(index['split']==split)),batch_size=args.batch_size,num_workers=0)
    config.update(model=args.model,profile=args.profile,architecture=architecture(args.model,args.profile),
                  batch_size=args.batch_size,accumulation_steps=accumulate,effective_batch_size=args.batch_size*accumulate,
                  epochs=args.epochs,optimizer=dict(name='AdamW',lr=args.learning_rate,weight_decay=.0001),
                  lr_schedule=warmup, helper_sha256={name:hashlib.sha256((here/name).read_bytes()).hexdigest()
                      for name in ('train_full_models.py','full_models.py','training_io.py')},
                  trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  schedule_code_sha256=hashlib.sha256((here/'warmup.py').read_bytes()).hexdigest(),
                  numeric_policy=dict(matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                                      cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                                      float32_matmul_precision=torch.get_float32_matmul_precision()),
                  optimization='Eight signal and eight W per microbatch; within-class sampling proportional to positive train-derived optimization weights. Unit CE on sampled rows; no second weight multiplication.',
                  sampler_replacement=True,epoch_definition='ceil(original train rows / microbatch size) draws',
                  checkpoint_selection='Lowest weighted validation CE; test scored only after checkpoint selection',
                  model_sha256=hashlib.sha256((args.models/(args.model+'.py')).read_bytes()).hexdigest(),
                  torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    model=model_for(args.model,args.models,args.profile).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.0001)
    history=[];best=float('inf');started=time.time();updates=0
    progress(args.progress,stage='training_start',model=args.model,profile=args.profile,
             train_rows=len(train),lr_schedule=warmup,epochs=args.epochs)
    for epoch in range(args.epochs):
        model.train();total=0.;rows=0;clipped=0;max_norm=0.
        first_lr=learning_rate(warmup,updates+1)
        for bi,batch in enumerate(training):
            if bi%accumulate==0:
                optimizer.zero_grad(set_to_none=True);group=min(accumulate,len(training)-bi)
                for pg in optimizer.param_groups:
                    pg['lr']=learning_rate(warmup,updates+1)
            x,v,mask,points,y,w,row=[a.cuda() for a in batch]
            loss=torch.nn.functional.cross_entropy(forward(args.model,model,x,v,mask,points),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            (loss/group).backward();total+=float(loss.detach())*len(y);rows+=len(y)
            if (bi+1)%accumulate==0 or bi+1==len(training):
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True))
                max_norm=max(max_norm,norm);clipped+=int(norm>5.)
                optimizer.step();updates+=1
            if (bi+1)%200==0:
                progress(args.progress,stage='training',epoch=epoch+1,epochs=args.epochs,rows=rows,
                         total_rows=len(training)*args.batch_size,elapsed_seconds=time.time()-started,
                         optimizer_updates=updates,learning_rate=optimizer.param_groups[0]['lr'])
        model.eval();weighted=0.;denominator=0.;scores=[];labels=[];weights=[]
        with torch.no_grad():
            for batch in loader(1):
                x,v,mask,points,y,w,row=[a.cuda() for a in batch]
                logits=forward(args.model,model,x,v,mask,points)
                weighted+=float((torch.nn.functional.cross_entropy(logits,y,reduction='none')*w).sum());denominator+=float(w.sum())
                scores.extend(torch.softmax(logits,1)[:,1].cpu().numpy());labels.extend(y.cpu().numpy());weights.extend(w.cpu().numpy())
        if not math.isfinite(weighted) or not math.isfinite(denominator) or denominator<=0:
            raise ValueError('Invalid weighted validation loss; test untouched')
        record=dict(epoch=epoch+1,train_loss=total/rows,validation_loss=weighted/denominator,
                    optimizer_updates=updates,first_lr=first_lr,last_lr=optimizer.param_groups[0]['lr'],
                    clipped_updates=clipped,max_preclip_gradient_norm=max_norm,
                    validation_auc=roc_auc_score(labels,scores,sample_weight=weights),validation_score_std=float(np.std(scores)))
        history.append(record)
        with (args.output/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(record));writer.writeheader();writer.writerows(history)
        print(json.dumps(record),flush=True)
        progress(args.progress,stage='epoch_complete',epoch=epoch+1,epochs=args.epochs,history=history,elapsed_seconds=time.time()-started)
        if (not all(math.isfinite(record[k]) for k in ('train_loss','validation_loss','validation_auc','validation_score_std'))
                or record['validation_score_std']<1e-4 or record['validation_auc']<=.55):
            raise RuntimeError('Learning gate failed on validation; test set remains untouched')
        if record['validation_loss']<best:
            best=record['validation_loss'];torch.save(dict(state_dict=model.state_dict(),configuration=config,architecture=args.model,best_epoch=epoch+1),args.output/'best.pt')
    chosen=choose_validation(history,args.epochs)
    if updates!=warmup['total_updates']:
        raise ValueError('Incomplete optimizer schedule')
    selection=dict(status='VALIDATION_ACCEPTED',selection_metric='minimum weighted validation CE',
                   best_epoch=chosen['epoch'],validation_loss=chosen['validation_loss'],
                   validation_auc=chosen['validation_auc'],validation_score_std=chosen['validation_score_std'],
                   checkpoint_sha256=hashlib.sha256((args.output/'best.pt').read_bytes()).hexdigest(),
                   index_sha256=config['index_sha256'],test_used_for_selection=False,optimizer_updates=updates)
    (args.output/'validation_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    checkpoint=torch.load(args.output/'best.pt',map_location='cpu',weights_only=False)
    if checkpoint['best_epoch']!=chosen['epoch']:
        raise ValueError('Checkpoint/validation selection mismatch')
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    scores=[];selected=[]
    with torch.no_grad():
        for batch in loader(2):
            x,v,mask,points,y,w,row=[a.cuda() for a in batch]
            scores.extend(torch.softmax(forward(args.model,model,x,v,mask,points),1)[:,1].cpu().numpy());selected.extend(row.cpu().numpy())
    selected=np.asarray(selected);scores=np.asarray(scores)
    if not np.isfinite(scores).all():raise ValueError('Nonfinite scores')
    np.savez_compressed(args.output/'test_scores.npz',score=scores,**{k:index[k][selected] for k in ['label','sample','split','physics_weight','optimization_weight','event_key','content_key','full_key','nPart']})
    summary=dict(model=args.model,profile=args.profile,best_epoch=checkpoint['best_epoch'],seconds=time.time()-started,
                 parameters=sum(p.numel() for p in model.parameters()),test_rows=len(scores),
                 auc=roc_auc_score(index['label'][selected],scores,sample_weight=index['optimization_weight'][selected]),
                 auc_note='Positive optimization-weighted test AUC, not a physical yield or limit')
    per_mass=[]
    test_sample=index['sample'][selected];test_label=index['label'][selected]
    for sample in sorted(set(test_sample[test_label==1])):
        take=(test_label==0)|(test_sample==sample)
        per_mass.append(dict(sample=sample,signal_test_rows=int((test_sample==sample).sum()),
                             background_test_rows=int((test_label==0).sum()),
                             auc=roc_auc_score(test_label[take],scores[take],
                                               sample_weight=index['optimization_weight'][selected][take])))
    with (args.output/'per_mass_metrics.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(per_mass[0]));writer.writeheader();writer.writerows(per_mass)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    progress(args.progress,stage='test_complete',**summary)


if __name__=='__main__':main()
