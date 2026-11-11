"""ROOT copier for one or both explicitly supplied models.

Every original object, branch and row is retained; unsupported objects fail closed.
The score schema is explicit per call, never changed through module globals.
"""
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

KNOWN_SCORES = ('ParticleNet_score', 'ParticleTransformer_score')
COMMON_BRANCHES = dict(NN_valid='bool', NN_trainingRegion='bool', NN_inputSplit='int8')
PF = dict(pt='pt', eta='eta', phi='phi', mass='mass', energy='E',
          px='PX', py='PY', pz='PZ', pdgId='pdgId', charge='charge', fromPV='fromPV')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def score_schema(score_branches):
    if isinstance(score_branches, (str, bytes)):
        raise ValueError('Supply an explicit sequence of known score branches')
    selected = tuple(score_branches)
    if (not selected or len(set(selected)) != len(selected)
            or not set(selected) <= set(KNOWN_SCORES)):
        raise ValueError('Unknown, empty or duplicate score schema')
    return dict({name: 'float32' for name in KNOWN_SCORES if name in selected}, **COMMON_BRANCHES)


def validate_extras(extra, batch, schema):
    n = len(batch['event'])
    if set(extra) != set(schema) or any(np.asarray(value).shape != (n,) for value in extra.values()):
        raise ValueError('Prediction length/schema mismatch')
    for name in schema:
        values = np.asarray(extra[name])
        if name in KNOWN_SCORES:
            if (values.dtype.kind != 'f' or not np.isfinite(values).all()
                    or np.any((values < 0) | (values > 1))):
                raise ValueError('Invalid score: '+name)
        elif name in ['NN_valid', 'NN_trainingRegion']:
            if values.dtype.kind != 'b':
                raise ValueError('Expected boolean prediction metadata: '+name)
        elif values.dtype.kind not in 'iu' or not np.isin(values, [-2, -1, 0, 1, 2]).all():
            raise ValueError('Invalid input-split codes')
    if not np.array_equal(np.asarray(extra['NN_trainingRegion']), ak.to_numpy(batch['abcdRegion']) == 1):
        raise ValueError('Training-region metadata disagrees with original ABCD region')
    is_data = ak.to_numpy(batch['isData']).astype(bool)
    split = np.asarray(extra['NN_inputSplit'])
    if np.any(split[is_data] != -1) or np.any(split[~is_data] == -1):
        raise ValueError('Data/MC split metadata disagrees with original event identity')


def compare_original(input_path, output_path, step, schema):
    checked = 0
    with uproot.open(input_path) as src, uproot.open(output_path) as dst:
        if set(dst.classnames(cycle=False)) != set(src.classnames(cycle=False)) | {'NNMetadata'}:
            raise ValueError('Changed original ROOT object set')
        for name, cls in src.classnames(cycle=False).items():
            if dst[name].classname != cls:
                raise ValueError('Changed ROOT class: '+name)
            if cls == 'TTree':
                original, copied = src[name], dst[name]
                if original.num_entries != copied.num_entries:
                    raise ValueError('Changed entry count: '+name)
                expected = set(original.keys()) | (set(schema) if name == 'Events' else set())
                if set(copied.keys()) != expected:
                    raise ValueError('Changed original branch set: '+name)
                if any(original[key].typename != copied[key].typename for key in original.keys()):
                    raise ValueError('Changed original branch type: '+name)
                for start in range(0, original.num_entries, step):
                    old = original.arrays(entry_start=start, entry_stop=start+step, library='ak', how=dict)
                    new = copied.arrays(list(old), entry_start=start, entry_stop=start+step, library='ak', how=dict)
                    for key in old:
                        if not ak.almost_equal(old[key], new[key], rtol=0, atol=0):
                            raise ValueError('Original values changed: '+name+'/'+key)
                checked += len(original.keys())
            elif cls == 'TObjString':
                if str(src[name]) != str(dst[name]):
                    raise ValueError('Metadata string changed: '+name)
            elif cls.startswith('TH1'):
                for attr in ['values', 'variances']:
                    np.testing.assert_array_equal(getattr(src[name], attr)(flow=True), getattr(dst[name], attr)(flow=True))
                np.testing.assert_array_equal(src[name].axis().edges(), dst[name].axis().edges())
    return checked


def augment(input_path, output_path, predictor, metadata, step=128, *, score_branches):
    schema = score_schema(score_branches)
    scores = tuple(name for name in schema if name in KNOWN_SCORES)
    models = [name[:-len('_score')] for name in scores]
    if 'checkpoint_sha256' in metadata and set(metadata['checkpoint_sha256']) != set(models):
        raise ValueError('Checkpoint metadata disagrees with supplied score schema')
    if 'scored_models' in metadata and set(metadata['scored_models']) != set(models):
        raise ValueError('Declared models disagree with supplied score schema')
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if step <= 0 or input_path == output_path or output_path.exists():
        raise ValueError('Use a positive batch size and a NEW, distinct output path')
    audit_path = output_path.with_suffix('.audit.json')
    if audit_path.exists():
        raise ValueError('Existing audit would be overwritten')
    digest = sha256(input_path)
    score_digest = {name: hashlib.sha256() for name in scores}
    with uproot.open(input_path) as src:
        classes = src.classnames(cycle=False)
        if classes.get('Events') != 'TTree' or 'NNMetadata' in classes:
            raise ValueError('Expected unscored selected Events TTree')
        if len(src.keys()) != len(classes):
            raise ValueError('Multiple ROOT cycles need a separate preservation strategy')
        for name, cls in classes.items():
            if cls != 'TTree' and cls != 'TObjString' and not cls.startswith('TH1'):
                raise ValueError('Unsupported ROOT object; refusing to discard: '+name+' '+cls)
        tree = src['Events']
        if (set(KNOWN_SCORES) | set(COMMON_BRANCHES)) & set(tree.keys()):
            raise ValueError('Input already has score branches')
        required = {'run', 'luminosityBlock', 'event', 'inputEntry', 'source_id', 'event_key', 'nPart',
                    'isData', 'abcdRegion', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_mass'} | {'Part_'+v for v in PF.values()}
        if not required <= set(tree.keys()):
            raise ValueError('Not a full-PF selected ntuple; missing '+str(sorted(required-set(tree.keys()))))
        total = tree.num_entries
        with uproot.create(output_path) as dst:
            dst.copy_from(src, filter_classname=lambda cls: cls != 'TTree')
            # uproot 5.3.12 may create an unused empty-name directory during a
            # top-level raw copy. It is not an input object and contains no data.
            if '' not in classes and '' in dst.keys(cycle=False):
                if dst[''].keys(cycle=False):
                    raise ValueError('Unexpected nonempty directory created by ROOT copier')
                del dst['']
            for name, cls in classes.items():
                if cls != 'TTree':
                    continue
                original = src[name]
                empty = original.arrays(entry_stop=0, library='ak', how=dict)
                branches = {key: value.type.content for key, value in empty.items()}
                if name == 'Events':
                    branches.update(schema)
                dst.mktree(name, branches, title=original.title)
                for batch in original.iterate(step_size=step, library='ak', how=dict):
                    if name == 'Events':
                        extra = predictor(batch)
                        validate_extras(extra, batch, schema)
                        # Write exactly the declared types and hash those same values.
                        extra = {key: np.asarray(value, dtype=schema[key]) for key, value in extra.items()}
                        for key in scores:
                            score_digest[key].update(extra[key].astype('<f4').tobytes())
                        batch.update(extra)
                    dst[name].extend(batch)
            metadata = dict(metadata, input_sha256=digest, input_path=str(input_path), entries=total,
                scored_models=models, score_branches=list(scores),
                output_layout='Only supplied model scores are embedded in Events; absent models have no branch',
                branch_preservation='All original selected-ntuple rows, branches, normalization and luminosity metadata')
            dst['NNMetadata'] = json.dumps(metadata, sort_keys=True)
    checked = compare_original(input_path, output_path, step, schema)
    with uproot.open(output_path) as out:
        for key in scores:
            checksum = hashlib.sha256()
            for batch in out['Events'].iterate([key], step_size=step, library='np'):
                checksum.update(batch[key].astype('<f4').tobytes())
            if checksum.hexdigest() != score_digest[key].hexdigest():
                raise ValueError('Score readback mismatch')
    if sha256(input_path) != digest:
        raise ValueError('Input changed during processing')
    audit = dict(status=0, input_sha256=digest, output_sha256=sha256(output_path),
                 entries=total, original_branches_verified=checked,
                 score_sha256={key: value.hexdigest() for key, value in score_digest.items()}, metadata=metadata)
    with audit_path.open('x') as target:
        json.dump(audit, target, indent=2)
    return audit
