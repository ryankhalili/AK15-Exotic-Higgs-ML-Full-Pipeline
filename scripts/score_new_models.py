"""Score trusted local training bundles into a new, fully checked ROOT copy."""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from full_models import architecture, model_for, forward
from ml_trial_common import FEATURES, tensors
from training_io import strict_precision

MODEL_NAMES = ('ParticleNet', 'ParticleTransformer')
REPLAY_ATOL = 1.e-5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(arrays):
    # This is the byte contract used by prepare_full_inputs.py, including shapes.
    digest = hashlib.sha256()
    for array in arrays:
        value = np.asarray(array).copy()
        value[value == 0] = 0
        digest.update(str(value.shape).encode() + value.dtype.str.encode() + value.tobytes())
    return digest.hexdigest()


def validate_result(directory, models, name):
    import torch
    from sklearn.metrics import roc_auc_score
    config = json.loads((directory/'results/configuration.json').read_text())
    summary = json.loads((directory/'results/summary.json').read_text())
    with (directory/'results/history.csv').open() as handle:
        history = list(csv.DictReader(handle))
    if len(history) != config['epochs'] or [int(r['epoch']) for r in history] != list(range(1, len(history)+1)):
        raise ValueError('Incomplete epoch history: '+name)
    losses = np.array([[float(r['train_loss']), float(r['validation_loss'])] for r in history])
    if not np.isfinite(losses).all():
        raise ValueError('Nonfinite training history')
    if summary['model'] != name or config['model'] != name:
        raise ValueError('Wrong model result')
    profile = config.get('profile', 'full')
    # JSON changes tuples to lists, so compare their serialized representation.
    if json.dumps(config['architecture'], sort_keys=True) != json.dumps(architecture(name, profile), sort_keys=True):
        raise ValueError('Architecture does not match recorded profile')
    if config['cap'] != 256 or config['features'] != FEATURES:
        raise ValueError('Unexpected 256-constituent feature contract')
    if sha256(models/(name+'.py')) != config['model_sha256']:
        raise ValueError('Pinned model code changed')
    index_path = directory/'indexed/index.npz'
    if sha256(index_path) != config['index_sha256']:
        raise ValueError('Index hash mismatch')
    with np.load(index_path, allow_pickle=False) as source:
        index = {k: source[k] for k in source.files}
    take = index['split'] == 2
    with np.load(directory/'results/test_scores.npz', allow_pickle=False) as scores:
        for key in scores.files:
            if key != 'score':
                np.testing.assert_array_equal(scores[key], index[key][take])
        predicted = scores['score']
        if not np.isfinite(predicted).all() or np.any((predicted < 0) | (predicted > 1)):
            raise ValueError('Invalid test score')
        auc = roc_auc_score(scores['label'], predicted, sample_weight=scores['optimization_weight'])
        if len(predicted) != summary['test_rows'] or not np.isclose(auc, summary['auc'], rtol=1.e-12, atol=1.e-12):
            raise ValueError('Test summary disagrees with scores')
    path = directory/'results/best.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    # Training preserves tuples in torch.save; configuration.json stores lists.
    # Compare every field after the same JSON representation, retaining exact values.
    if (json.dumps(checkpoint['configuration'], sort_keys=True) != json.dumps(config, sort_keys=True)
            or checkpoint['architecture'] != name):
        raise ValueError('Checkpoint provenance mismatch')
    if checkpoint['best_epoch'] != int(np.argmin(losses[:, 1]))+1 or summary['best_epoch'] != checkpoint['best_epoch']:
        raise ValueError('Checkpoint was not selected on minimum validation loss')
    return config, checkpoint, index, summary


def text_key(key):
    return key.decode() if isinstance(key, bytes) else str(key)


class NewFrozenModels:
    def __init__(self, trained, models, device='cuda'):
        if not trained or not set(trained) <= set(MODEL_NAMES) or any(path is None for path in trained.values()):
            raise ValueError('Supply ParticleNet, ParticleTransformer, or both; no unknown or missing model')
        selected = tuple(name for name in MODEL_NAMES if name in trained)
        import torch
        strict_precision(torch)
        self.torch, self.device = torch, torch.device(device)
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested GPU unavailable')
        torch.set_num_threads(2)
        self.models, self.configs, self.hashes, self.test_scores = {}, {}, {}, {}
        reference, reference_config = None, None
        for name in selected:
            config, checkpoint, index, summary = validate_result(trained[name], models, name)
            if reference is not None:
                for key in ['event_key', 'content_key', 'split', 'label', 'physics_weight', 'optimization_weight']:
                    np.testing.assert_array_equal(index[key], reference[key])
                for key in ['mean', 'std', 'cap', 'features', 'seed']:
                    if config[key] != reference_config[key]:
                        raise ValueError('Models have different input/split contracts')
                if config.get('profile', 'full') != reference_config.get('profile', 'full'):
                    raise ValueError('Do not mix full and compact campaigns')
            reference, reference_config = index, config
            model = model_for(name, models, config.get('profile', 'full')).to(self.device)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            if sum(p.numel() for p in model.parameters()) != summary['parameters']:
                raise ValueError('Parameter count mismatch')
            self.models[name], self.configs[name] = model.eval(), config
            self.hashes[name] = sha256(trained[name]/'results/best.pt')
            with np.load(trained[name]/'results/test_scores.npz', allow_pickle=False) as saved:
                self.test_scores[name] = {text_key(key): float(value)
                    for key, value in zip(saved['content_key'], saved['score'])}
        self.cap = config['cap']
        self.mean = np.asarray(config['mean'], dtype='float32')[None, :, None]
        self.std = np.asarray(config['std'], dtype='float32')[None, :, None]
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError('Invalid saved normalization')
        self.sources = {text_key(k): int(v) for k, v in zip(reference['event_key'], reference['split'])}
        self.contents = {text_key(k): int(v) for k, v in zip(reference['content_key'], reference['split'])}
        self.expected_content = {text_key(key): text_key(content)
            for key, content in zip(reference['event_key'], reference['content_key'])}
        if len(self.sources) != len(reference['event_key']) or len(self.contents) != len(reference['content_key']):
            raise ValueError('Duplicate identities or input fingerprints in training index')
        self.replay = {name: dict(comparisons=0, max_abs_difference=0.) for name in selected}
        self.metadata = dict(checkpoint_sha256=self.hashes, cap=self.cap,
            profile=config.get('profile', 'full'), features=FEATURES, scored_models=list(selected),
            score_definition='softmax(logits)[:,1]; uncalibrated discriminant',
            input_split_codes={'-2':'MC membership unverified', '-1':'data', '0':'train', '1':'validation', '2':'test'},
            saved_test_replay=dict(absolute_tolerance=REPLAY_ATOL, relative_tolerance=0., models=self.replay),
            training='Binary signal/background training in region A only; see training configuration',
            weights='Preserved as ROOT values; not model features. Physics yields require signed weights and calibrated exposure.',
            control_regions='Scores in B/C/D do not establish QCD closure or D-to-A score-shape transfer')

    def __call__(self, batch):
        import awkward as ak
        parts = ['pt','eta','phi','energy','px','py','pz','pdgId','charge','fromPV']
        suffix = {'energy':'E','px':'PX','py':'PY','pz':'PZ'}
        rows, membership, content_keys = [], [], []
        for i in range(len(batch['jet_pt'])):
            p = {k: ak.to_numpy(batch['Part_'+suffix.get(k,k)][i]) for k in parts}
            if len({len(v) for v in p.values()}) != 1 or len(p['pt']) != int(batch['nPart'][i]):
                raise ValueError('PF constituent count mismatch')
            row = tensors(p, *[float(batch[k][i]) for k in ['jet_pt','jet_eta','jet_phi','jet_mass']], cap=self.cap)
            rows.append(row)
            content = fingerprint(row[:3])
            content_keys.append(content)
            if bool(batch['isData'][i]):
                membership.append(-1)
            else:
                key = str(batch['event_key'][i])
                a = self.sources.get(key)
                b = self.contents.get(content)
                if key in self.expected_content and content != self.expected_content[key]:
                    raise ValueError('Known event identity has changed input fingerprint')
                if a is not None and b is not None and a != b:
                    raise ValueError('Event and content membership disagree')
                membership.append(a if a is not None else b if b is not None else -2)
        x, v, mask = [np.stack([r[j] for r in rows]) for j in range(3)]
        points = x[:, :2].copy()
        x = ((x-self.mean)/self.std)*mask
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite standardized input')
        arrays = [self.torch.from_numpy(a).to(self.device) for a in [x,v,mask,points]]
        result = {}
        with self.torch.inference_mode():
            for name, model in self.models.items():
                predicted = self.torch.softmax(forward(name, model, *arrays), dim=1)[:,1].cpu().numpy()
                for i, content in enumerate(content_keys):
                    if membership[i] == 2:
                        if content not in self.test_scores[name]:
                            raise ValueError('Test member has no saved reference score: '+name)
                        delta = abs(float(predicted[i]) - self.test_scores[name][content])
                        if not np.isfinite(delta) or delta > REPLAY_ATOL:
                            raise ValueError('Strict saved-test score replay failed: '+name)
                        self.replay[name]['comparisons'] += 1
                        self.replay[name]['max_abs_difference'] = max(self.replay[name]['max_abs_difference'], delta)
                result[name+'_score'] = predicted
        result.update(NN_valid=np.ones(len(rows),dtype=bool),
                      NN_trainingRegion=ak.to_numpy(batch['abcdRegion'])==1,
                      NN_inputSplit=np.asarray(membership,dtype='int8'))
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--particle-net', type=Path)
    ap.add_argument('--particle-transformer', type=Path)
    ap.add_argument('--models', type=Path, required=True)
    ap.add_argument('--writer-directory', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args()
    trained = {name: path for name, path in [('ParticleNet', args.particle_net),
                                            ('ParticleTransformer', args.particle_transformer)] if path is not None}
    if not trained:
        ap.error('Supply --particle-net, --particle-transformer, or both')
    sys.path.append(str(args.writer_directory.resolve()))
    writer = importlib.import_module('score_root_copy')
    if Path(writer.__file__).resolve().parent != args.writer_directory.resolve():
        raise ValueError('Unexpected ROOT writer import')
    import uproot
    from make_selected_pf_ntuple import validate_selection
    with uproot.open(args.input) as source:
        tree = source['Events']
        scalar = [k for k in tree.keys() if not k.startswith('Part_') and k not in ['source_id','event_key']]
        spec = json.loads(str(source['ProductionSpec']))
        validate_selection(tree.arrays(scalar,library='np'),
                           source['AnalysisMetadata'].arrays(library='np'),spec['is_data'])
    predictor = NewFrozenModels(trained,args.models,args.device)
    predictor.metadata.update(adapter_sha256=sha256(__file__),writer_sha256=sha256(writer.__file__))
    audit = writer.augment(args.input,args.output,predictor,predictor.metadata,args.batch_size,
                           score_branches=tuple(name+'_score' for name in predictor.models))
    print(json.dumps({k:v for k,v in audit.items() if k!='metadata'},indent=2))


if __name__ == '__main__': main()
