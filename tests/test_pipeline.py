import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import awkward as ak
import numpy as np
import pytest
import uproot
from abcd_statistics import predict, eigen_variations
from build_card import PROCESSES, finalize_card, qcd_policy
from expected_limits import commands
from full_models import architecture
from index_full_inputs import split_for, require_unique
from limit_checks import validate_quantiles
from ml_trial_common import training_weights
from prepare_training import prepare
from score_histograms import build, collect, validate_config
from training_io import load_index
from train_transformer import BalancedBatches
from warmup import schedule, learning_rate, choose_validation

ROOT = Path(__file__).resolve().parents[1]


def root_sample(path, sample, *, data=False, n=160, pf=False, split=-2, score_shift=0.):
    indices = np.arange(n)
    region = indices % 4 + 1
    values = dict(run=np.ones(n, dtype='uint32'), luminosityBlock=np.ones(n, dtype='uint32'),
        event=indices.astype('uint64'), event_key=ak.Array([hashlib.sha256((sample+str(i)).encode()).hexdigest() for i in indices]),
        isData=np.full(n, data), genWeight=np.where(indices % 13 == 0, -1., 1.) if not data else np.ones(n),
        abcdRegion=region.astype('int32'), NN_valid=np.ones(n, dtype=bool),
        NN_inputSplit=np.full(n, -1 if data else split, dtype='int8'),
        ParticleNet_score=((indices // 4 % 2) * .5 + .25 + score_shift).astype('float32'))
    if pf:
        # Only A rows enter training. Every row has distinct full/capped content.
        values.update(jet_pt=300.+indices*.1, jet_eta=indices*.0001,
                      jet_phi=indices*.0002, jet_mass=40.+indices*.01,
                      nPart=np.full(n, 20, dtype='int32'))
        for name in ('pt','eta','phi','E','PX','PY','PZ','pdgId','charge','fromPV'):
            base = np.arange(1, 21, dtype=float)
            rows = [base * (1 + i*.0001 + (0.1 if sample == 'signal' else 0)) for i in indices]
            values['Part_'+name] = ak.Array(rows)
    schema = {k: ('string' if k == 'event_key' else 'var * float64' if k.startswith('Part_') else v.dtype) for k, v in values.items()}
    with uproot.create(path) as root:
        root.mktree('Events', schema)
        root['Events'].extend(values)
        root['ProductionSpec'] = json.dumps(dict(sample=sample, is_data=data, source=sample))
        root['normalization'] = (np.array([100000., 100000.]), np.arange(3, dtype=float))


def histogram_config(tmp_path):
    samples = []
    for process in ('data', 'signal', 'WJets', 'DYJets', 'TT', 'ST', 'VV'):
        filename = process + '.root'
        root_sample(tmp_path / filename, process, data=process == 'data', n=1600 if process == 'data' else 160)
        row = dict(name=process, process=process, files=[filename], region_a_selection='all')
        if process != 'data':
            row.update(xsec_pb=1., factor=1., sum_genweights=100000.)
        samples.append(row)
    return dict(luminosity_pb=1000., score_branch='ParticleNet_score', edges=[0., .5, 1.], samples=samples)


def test_signed_histograms_and_abcd_readback(tmp_path):
    config = histogram_config(tmp_path)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    report = build(path, tmp_path / 'output')
    edges, groups, receipt = collect(config, tmp_path)
    with uproot.open(tmp_path / 'output/histograms.root') as root:
        np.testing.assert_allclose(root['pass/WJets'].values(), groups['WJets'][0][0])
        np.testing.assert_allclose(root['pass/WJets'].variances(), groups['WJets'][1][0])
        assert root['pass/WJets'].values().sum() < .01 * 40
        assert root['pass/WJets'].variances().sum() == pytest.approx(.0001 * 40)
        assert root['pass/data_obs'].values().sum() == 400
        assert root['pass/QCD'].values().sum() == pytest.approx(report['qcd_yield'])
        assert root['statistical_covariance/QCD'].values().sum() == pytest.approx(report['qcd_variance'])
        assert np.all(root['pass/QCD'].values(flow=True)[[0,-1]] == 0)
    policy = qcd_policy(json.loads((tmp_path / 'output/qcd_shapes.json').read_text()))
    assert len(policy['nuisances']) == 2
    with pytest.raises(ValueError, match='new output'):
        build(path, tmp_path / 'output')


def test_overlaps_denominators_and_membership_fail_closed(tmp_path):
    config = histogram_config(tmp_path)
    duplicate = tmp_path / 'data_copy.root'
    root_sample(duplicate, 'data', data=True, n=1600)
    config['samples'][0]['files'].append(duplicate.name)
    _, groups, receipt = collect(config, tmp_path)
    assert receipt['data_duplicates_removed'] == 1600
    assert groups['data'][0].sum() == 1600
    duplicate.unlink()
    root_sample(duplicate, 'data', data=True, n=1600, score_shift=.01)
    with pytest.raises(ValueError, match='different region/score'):
        collect(config, tmp_path)
    config['samples'][0]['files'].pop()
    config['samples'][1]['sum_genweights'] = 99999.
    with pytest.raises(ValueError, match='denominator'):
        collect(config, tmp_path)
    config['samples'][1]['sum_genweights'] = 100000.
    signal = tmp_path / 'signal.root'
    signal.unlink()
    root_sample(signal, 'signal', split=0)
    with pytest.raises(ValueError, match='Training-member'):
        collect(config, tmp_path)
    config['samples'][1].update(region_a_selection='test', test_fraction=.15)
    _, groups, _ = collect(config, tmp_path)
    assert groups['signal'][0][0].sum() == 0
    assert groups['signal'][0][1:].sum() > 0


def test_score_endpoints_and_unit_data_weights(tmp_path):
    config = histogram_config(tmp_path)
    bad = copy.deepcopy(config)
    bad['samples'][0]['factor'] = 2
    with pytest.raises(ValueError, match='data'):
        validate_config(bad)
    bad = copy.deepcopy(config)
    bad['edges'] = [0., .5, .5, 1.]
    with pytest.raises(ValueError, match='edges'):
        validate_config(bad)
    config['samples'][1]['files'].append('signal.root')
    with pytest.raises(ValueError, match='Repeated input'):
        collect(config, tmp_path)


def test_prepare_index_and_train_only_preprocessing(tmp_path):
    specs = []
    for name, label in (('signal', 1), ('background', 0)):
        root_sample(tmp_path / (name+'.root'), name, n=1200, pf=True)
        specs.append(dict(key=name, sample=name, file=name+'.root', label=label, xsec_pb=1., factor=1.))
    manifest = tmp_path / 'training.json'
    manifest.write_text(json.dumps(specs))
    prepared, indexed = tmp_path / 'prepared', tmp_path / 'indexed'
    result = prepare(manifest, prepared, 70)
    assert sum(s['rows'] for s in result['shards']) == 600
    subprocess.run([sys.executable, '-B', str(ROOT/'scripts/index_full_inputs.py'), str(prepared), str(indexed)], check=True, capture_output=True)
    config, index = load_index(indexed)
    assert config['cap'] == 256
    assert len(index['label']) == 600
    train_x = []
    for i, shard in enumerate(config['shards']):
        x = np.load(Path(shard['directory']) / (shard['stem']+'_x.npy'))
        mask = np.load(Path(shard['directory']) / (shard['stem']+'_mask.npy'))
        for row in np.flatnonzero((index['shard'] == i) & (index['split'] == 0)):
            offset = index['offset'][row]
            train_x.append(x[offset][:, mask[offset, 0]])
    np.testing.assert_allclose(config['mean'], np.concatenate(train_x, axis=1).mean(axis=1, dtype=float), atol=1e-8)
    persisted = json.loads((indexed/'configuration.json').read_text())
    assert all(not Path(s['directory']).is_absolute() for s in persisted['shards'])
    with (indexed/'index.npz').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        load_index(indexed)


def test_split_and_weight_contract():
    keys = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(1000)]
    assert [split_for(k) for k in keys][::-1] == [split_for(k) for k in keys[::-1]]
    with pytest.raises(ValueError, match='Repeated'):
        require_unique(np.array(['same', 'same']), 'content')
    labels = np.array([0, 0, 1, 1, 1, 1])
    samples = np.array(['b','b','s1','s1','s2','s2'])
    physics = np.array([2., -1., 4., -3., 9., -1.])
    weights = training_weights(labels, samples, physics, np.ones(6, dtype=bool))
    assert np.all(weights > 0)
    assert weights[labels == 0].sum() == pytest.approx(weights[labels == 1].sum())
    assert weights[samples == 's1'].sum() == pytest.approx(weights[samples == 's2'].sum())
    np.testing.assert_array_equal(physics, [2,-1,4,-3,9,-1])


def test_abcd_covariance_and_failures():
    residual = np.array([[100.,120.],[800.,900.],[600.,700.],[400.,500.]])
    variance = residual.copy()
    q = predict(residual, variance)
    variations = eigen_variations(q['values'], q['covariance'])
    reconstructed = sum(np.outer((up-down)/2, (up-down)/2) for up, down in variations.values())
    np.testing.assert_allclose(reconstructed, q['covariance'])
    for invalid in (np.full((4,2), -1.), np.full((4,2), np.nan)):
        with pytest.raises(ValueError):
            predict(invalid, variance)


def test_balanced_sampler_and_schedule():
    sampler = BalancedBatches(np.array([0,0,1,1]), np.array([1.,2.,3.,4.]), 16, 5)
    for batch in sampler:
        assert sum(i < 2 for i in batch) == 8
    config = schedule(100, 16, 8, 6, 1, .001)
    assert learning_rate(config, config['total_updates']) == pytest.approx(.001)
    with pytest.raises(ValueError):
        choose_validation([], 6)
    assert architecture('ParticleTransformer')['num_layers'] == 8
    assert architecture('ParticleNet')['conv_params'][0][0] == 16


def test_card_mode_and_expected_only_quantiles():
    names = list(PROCESSES)
    text = 'bin pass\nobservation -1\nbin ' + ' '.join(['pass']*7) + '\nprocess ' + ' '.join(names) + '\nprocess 0 1 2 3 4 5 6\nrate ' + ' '.join(['-1']*7)
    rates = {name: 1.23456789012345 for name in names + ['data_obs']}
    card = finalize_card(text, rates, qcd_policy(fixed=True))
    assert 'pass autoMCStats 0 1 2' in card
    assert '1.23456789012345' in card
    cmd = commands(100.)[1]
    assert cmd[cmd.index('--run')+1] == 'blind'
    rows = [dict(quantileExpected=q, limit=i+1.) for i, q in enumerate((.025,.16,.5,.84,.975))]
    assert validate_quantiles(rows, 100)[2]['limit'] == 3
    with pytest.raises(ValueError):
        validate_quantiles(rows + [dict(quantileExpected=-1., limit=2.)], 100)
    with pytest.raises(ValueError):
        validate_quantiles(rows, 4)


@pytest.mark.parametrize('script', ['prepare_training.py','index_full_inputs.py','train_full_models.py',
    'train_transformer.py','make_selected_pf_ntuple.py','score_new_models.py','score_histograms.py','build_card.py','expected_limits.py'])
def test_cli_help(script):
    run = subprocess.run([sys.executable, '-B', str(ROOT/'scripts'/script), '--help'], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert 'usage:' in run.stdout
