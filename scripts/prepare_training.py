"""Prepare bounded PF tensor shards from an explicit list of selected MC ntuples."""
import argparse
import hashlib
import json
from pathlib import Path
import awkward as ak
import numpy as np
import uproot
from ml_trial_common import tensors
from score_new_models import fingerprint

META = np.dtype([('event_key', 'S64'), ('content_key', 'S64'), ('full_key', 'S64'),
                 ('genWeight', '<f8'), ('nPart', '<i4')])
PARTS = ['pt', 'eta', 'phi', 'energy', 'px', 'py', 'pz', 'pdgId', 'charge', 'fromPV']
SUFFIX = {'energy': 'E', 'px': 'PX', 'py': 'PY', 'pz': 'PZ'}


def prepare(manifest_path, output, shard_size=1000):
    manifest_path, output = Path(manifest_path).resolve(), Path(output)
    specs = json.loads(manifest_path.read_text())
    if not isinstance(specs, list) or not specs or shard_size < 1:
        raise ValueError('Nonempty source list and positive shard size required')
    if len({s['key'] for s in specs}) != len(specs):
        raise ValueError('Repeated source key')
    paths = [(manifest_path.parent / s['file']).resolve() for s in specs]
    if len(set(paths)) != len(paths):
        raise ValueError('Repeated input path')
    for spec, path in zip(specs, paths):
        if spec['label'] not in (0, 1) or not path.is_file():
            raise ValueError('Each source needs label 0/1 and an existing local ROOT file')
        if not all(np.isfinite(spec[k]) and spec[k] > 0 for k in ('xsec_pb', 'factor')):
            raise ValueError('Positive explicit normalization constants required')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(cap=256, sources=[], shards=[], source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    for spec, path in zip(specs, paths):
        with uproot.open(path) as root:
            norm = root['normalization'].values()
            if len(norm) < 2 or not np.isfinite(norm[:2]).all() or norm[0] <= 0:
                raise ValueError('Missing preselection input count/signed sumGenWeights')
            provenance = json.loads(str(root['ProductionSpec']))
            if provenance['is_data'] or provenance['sample'] != spec['sample']:
                raise ValueError('Data or sample-label mismatch in training inputs')
            source_id = provenance['source']
            buffers = [[], [], [], []]
            selected = 0

            def flush():
                if not buffers[0]:
                    return
                stem = 'shard_%06d' % len(manifest['shards'])
                arrays = [np.stack(b) for b in buffers[:3]] + [np.array(buffers[3], dtype=META)]
                hashes = {}
                for key, array in zip(('x', 'v', 'mask', 'meta'), arrays):
                    dest = output / (stem + '_' + key + '.npy')
                    np.save(dest, array, allow_pickle=False)
                    hashes[key] = hashlib.sha256(dest.read_bytes()).hexdigest()
                manifest['shards'].append(dict(stem=stem, rows=len(arrays[0]), source_key=spec['key'],
                                                sample=spec['sample'], sha256=hashes))
                for buffer in buffers:
                    buffer.clear()

            for batch in root['Events'].iterate(step_size=200, library='ak'):
                if ak.any(batch.isData) or not ak.all((batch.abcdRegion >= 1) & (batch.abcdRegion <= 4)):
                    raise ValueError('Invalid MC/ABCD membership')
                for row in batch[batch.abcdRegion == 1]:
                    parts = {k: ak.to_numpy(row['Part_' + SUFFIX.get(k, k)]) for k in PARTS}
                    if len({len(v) for v in parts.values()}) != 1 or len(parts['pt']) != int(row.nPart):
                        raise ValueError('PF count mismatch')
                    if not all(np.isfinite(v).all() for v in parts.values()):
                        raise ValueError('Nonfinite full-PF input')
                    jet = [float(row[k]) for k in ('jet_pt', 'jet_eta', 'jet_phi', 'jet_mass')]
                    if not np.isfinite(jet).all() or jet[0] <= 0 or not np.isfinite(row.genWeight):
                        raise ValueError('Invalid jet or generator weight')
                    x, v, mask, count = tensors(parts, *jet, cap=256)
                    key = str(row.event_key)
                    if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
                        raise ValueError('Expected SHA256 event key')
                    order = np.argsort(-parts['pt'], kind='stable')
                    full = fingerprint([np.asarray(jet, dtype='<f4')] +
                                       [np.asarray(parts[k][order], dtype='<f4') for k in sorted(parts)])
                    for buffer, value in zip(buffers[:3], (x, v, mask)):
                        buffer.append(value)
                    buffers[3].append((key.encode(), fingerprint((x, v, mask)).encode(), full.encode(), float(row.genWeight), count))
                    selected += 1
                    if len(buffers[0]) >= shard_size:
                        flush()
            flush()
            manifest['sources'].append(dict(key=spec['key'], source=source_id, sample=spec['sample'],
                label=spec['label'], input_events=int(norm[0]), sum_genweights=float(norm[1]),
                xsec_pb=spec['xsec_pb'], factor=spec['factor'], selected_A=selected))
    with (output / 'manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--shard-size', type=int, default=1000)
    args = parser.parse_args()
    result = prepare(args.manifest, args.output, args.shard_size)
    print('Prepared %d sources in %d shards' % (len(result['sources']), len(result['shards'])))


if __name__ == '__main__':
    main()
