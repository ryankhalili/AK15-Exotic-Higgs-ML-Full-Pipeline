"""Build signed score TH1 templates and statistical ABCD shapes from scored samples."""
import argparse
import hashlib
import json
from pathlib import Path
import awkward as ak
import numpy as np
import uproot
from abcd_statistics import predict, eigen_variations
from template_io import th1, qcd_spec
from build_card import PROCESSES, require, sha

BACKGROUND = PROCESSES[2:]


def validate_config(config):
    edges = np.asarray(config['edges'], dtype=float)
    require(edges.ndim == 1 and 2 <= len(edges) <= 1001 and np.isfinite(edges).all()
            and np.all(np.diff(edges) > 0) and edges[0] == 0 and edges[-1] == 1,
            'Score edges must increase from 0 to 1')
    require(config['score_branch'] in ('ParticleNet_score', 'ParticleTransformer_score'), 'Unknown score branch')
    lumi = config['luminosity_pb']
    require(np.isfinite(lumi) and lumi > 0, 'Positive explicit luminosity required')
    samples = config['samples']
    require(isinstance(samples, list) and samples, 'Nonempty samples required')
    require(len({s['name'] for s in samples}) == len(samples), 'Repeated sample name')
    require({s['process'] for s in samples} == {'data', 'signal', *BACKGROUND},
            'Supply data, one signal hypothesis, and all five non-QCD process groups')
    require(sum(s['process'] == 'signal' for s in samples) == 1, 'One signal hypothesis per build')
    for sample in samples:
        require(isinstance(sample['files'], list) and bool(sample['files']), 'Empty file list')
        selection = sample['region_a_selection']
        require(selection in ('all', 'test'), 'Explicit all/test region-A selection required')
        if sample['process'] == 'data':
            require(selection == 'all', 'Data must use all unique events')
            require(not any(k in sample for k in ('xsec_pb', 'factor', 'sum_genweights', 'test_fraction')),
                    'Do not specify MC weights or sampling factors for data')
        else:
            require(all(np.isfinite(sample[k]) and sample[k] > 0 for k in ('xsec_pb', 'factor', 'sum_genweights')),
                    'Positive finite MC normalization constants required')
        if selection == 'test':
            require(np.isfinite(sample['test_fraction']) and 0 < sample['test_fraction'] < 1,
                    'Supply the frozen test fraction')
    return edges


def collect(config, base):
    edges = validate_config(config)
    shape = (4, len(edges) - 1)
    groups = {p: [np.zeros(shape), np.zeros(shape)] for p in ('data', 'signal', *BACKGROUND)}
    data_seen, mc_seen, files_seen, receipts = {}, set(), set(), []
    duplicates = 0
    for sample in config['samples']:
        is_data = sample['process'] == 'data'
        coefficient = 1. if is_data else config['luminosity_pb'] * sample['xsec_pb'] * sample['factor'] / sample['sum_genweights']
        norm_sum = 0.
        for filename in sample['files']:
            path = (Path(base) / filename).resolve()
            require(path not in files_seen, 'Repeated input file')
            files_seen.add(path)
            digest = sha(path)
            with uproot.open(path) as root:
                spec = json.loads(str(root['ProductionSpec']))
                require(bool(spec['is_data']) == is_data and spec['sample'] == sample['name'], 'Input provenance differs from sample manifest')
                if not is_data:
                    norm = root['normalization'].values()
                    require(len(norm) >= 2 and np.isfinite(norm[:2]).all(), 'Invalid preselection normalization')
                    norm_sum += float(norm[1])
                fields = ['run', 'luminosityBlock', 'event', 'event_key', 'isData', 'genWeight',
                          'abcdRegion', 'NN_valid', 'NN_inputSplit', config['score_branch']]
                for batch in root['Events'].iterate(fields, library='ak', step_size=50000):
                    score = ak.to_numpy(batch[config['score_branch']])
                    region = ak.to_numpy(batch.abcdRegion)
                    split = ak.to_numpy(batch.NN_inputSplit)
                    weight = ak.to_numpy(batch.genWeight).astype(float)
                    require(np.isfinite(score).all() and np.all((score >= 0) & (score <= 1)), 'Invalid score')
                    require(np.isfinite(weight).all() and np.isin(region, [1, 2, 3, 4]).all(), 'Invalid weight/region')
                    require(bool(ak.all(batch.NN_valid)) and bool(ak.all(batch.isData == is_data)), 'Invalid scoring/data flag')
                    require(np.isin(split, [-2, -1, 0, 1, 2]).all(), 'Invalid split code')
                    require(np.all(split == -1) if is_data else np.all(split != -1), 'Data/MC split mismatch')
                    keep = np.ones(len(score), dtype=bool)
                    if is_data:
                        require(np.all(weight == 1), 'Data must have unit generator weights')
                        for i, row in enumerate(batch):
                            identity = tuple(int(row[k]) for k in ('run', 'luminosityBlock', 'event'))
                            signature = (int(region[i]), float(score[i]))
                            if identity in data_seen:
                                require(data_seen[identity] == signature, 'Duplicate data has different region/score; review overlap upstream')
                                keep[i] = False
                                duplicates += 1
                            else:
                                data_seen[identity] = signature
                    else:
                        for key in ak.to_list(batch.event_key):
                            require(key not in mc_seen, 'Repeated MC event key')
                            mc_seen.add(key)
                        a = region == 1
                        if sample['region_a_selection'] == 'test':
                            require(np.isin(split[a], [0, 1, 2]).all(), 'Unverified training membership in region A')
                            keep &= ~a | (split == 2)
                            weight[a] /= sample['test_fraction']
                        else:
                            require(np.all(split[a] == -2), 'Training-member MC cannot use all rows for region-A templates')
                        weight *= coefficient
                    for r in range(1, 5):
                        take = keep & (region == r)
                        groups[sample['process']][0][r-1] += np.histogram(score[take], edges, weights=weight[take])[0]
                        groups[sample['process']][1][r-1] += np.histogram(score[take], edges, weights=weight[take]**2)[0]
                receipts.append(dict(sample=sample['name'], file=filename, sha256=digest, rows=root['Events'].num_entries))
            require(sha(path) == digest, 'Input changed during read')
        if not is_data:
            require(np.isclose(norm_sum, sample['sum_genweights'], rtol=1e-11, atol=0), 'Processed signed generator denominator differs from inputs')
    return edges, groups, dict(files=receipts, data_duplicates_removed=duplicates)


def build(config_path, output):
    config_path, output = Path(config_path), Path(output)
    require(not output.exists(), 'Choose a new output directory')
    config = json.loads(config_path.read_text())
    edges, groups, receipt = collect(config, config_path.resolve().parent)
    residual = groups['data'][0] - sum(groups[p][0] for p in BACKGROUND)
    variance = groups['data'][1] + sum(groups[p][1] for p in BACKGROUND)
    qcd = predict(residual, variance)
    variations = eigen_variations(qcd['values'], qcd['covariance'])
    require(bool(variations), 'No nonzero QCD uncertainty modes')
    for p in ('signal', *BACKGROUND):
        require(np.all(groups[p][0][0] >= 0) and groups[p][0][0].sum() > 0,
                'Negative or empty MC template: review common binning, never clip ' + p)
    output.mkdir(parents=True, exist_ok=False)
    with uproot.create(output / 'histograms.root') as root:
        for process, name in [('data', 'data_obs')] + [(p, p) for p in ('signal', *BACKGROUND)]:
            root['pass/' + name] = th1(name, groups[process][0][0], groups[process][1][0], edges)
        root['pass/QCD'] = th1('QCD', qcd['values'], np.diag(qcd['covariance']), edges)
        root['statistical_covariance/QCD'] = (qcd['covariance'], edges, edges)
        for nuisance, pair in variations.items():
            for direction, values in zip(('Up', 'Down'), pair):
                name = 'QCD_' + nuisance + direction
                root['pass/' + name] = th1(name, values, np.zeros_like(values), edges)
    with uproot.open(output / 'histograms.root') as root:
        for process, name in [('data', 'data_obs')] + [(p, p) for p in ('signal', *BACKGROUND)]:
            np.testing.assert_allclose(root['pass/' + name].values(), groups[process][0][0])
            np.testing.assert_allclose(root['pass/' + name].variances(), groups[process][1][0])
        np.testing.assert_allclose(root['statistical_covariance/QCD'].values(), qcd['covariance'])
    (output / 'qcd_shapes.json').write_text(json.dumps(qcd_spec(variations), indent=2) + '\n')
    receipt.update(luminosity_pb=config['luminosity_pb'], score_branch=config['score_branch'],
                   qcd_yield=qcd['total'], qcd_variance=qcd['total_variance'],
                   config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                   groups={p: dict(sumw=y.tolist(), sumw2=v.tolist()) for p, (y, v) in groups.items()},
                   scope='Conditional ABCD statistical model; not a validated transfer or branching-fraction result')
    (output / 'histogram_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = build(args.config, args.output)
    print(json.dumps({k: result[k] for k in ('qcd_yield', 'qcd_variance', 'data_duplicates_removed')}, indent=2))


if __name__ == '__main__':
    main()
