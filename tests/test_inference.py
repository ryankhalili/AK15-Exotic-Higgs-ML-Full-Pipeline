"""Optional adapter roundtrip against user-supplied author architecture source."""
import json
import copy
import os
from pathlib import Path
import awkward as ak
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from full_models import architecture, model_for, forward
from particle_features import FEATURES, tensors
from score_new_models import NewFrozenModels, fingerprint, sha256, validate_result
from score_root_copy import augment
from test_score_root_copy import source
import uproot


def test_identity_and_replay_guards_without_external_models(tmp_path):
    input_path = tmp_path/'input.root'
    source(input_path, n=2)
    with uproot.open(input_path) as root:
        batch = root['Events'].arrays(library='ak', how=dict)
    parts = {key: ak.to_numpy(batch['Part_'+suffix][1]) for key, suffix in
             dict(pt='pt',eta='eta',phi='phi',energy='E',px='PX',py='PY',pz='PZ',pdgId='pdgId',charge='charge',fromPV='fromPV').items()}
    content = fingerprint(tensors(parts,1.,1.,1.,1.)[:3])

    class Constant(torch.nn.Module):
        def forward(self, points, features, mask):
            return torch.zeros((len(features),2))

    predictor = NewFrozenModels.__new__(NewFrozenModels)
    predictor.torch, predictor.device = torch, torch.device('cpu')
    predictor.cap = 256
    predictor.mean = np.zeros((1,12,1),dtype='float32')
    predictor.std = np.ones((1,12,1),dtype='float32')
    predictor.models = {'ParticleNet':Constant()}
    predictor.sources, predictor.contents = {'1':2}, {content:2}
    predictor.expected_content = {'1':content}
    predictor.test_scores = {'ParticleNet':{content:.5}}
    predictor.replay = {'ParticleNet':dict(comparisons=0,max_abs_difference=0.)}
    np.testing.assert_array_equal(predictor(batch)['NN_inputSplit'],[-1,2])
    assert predictor.replay['ParticleNet']['comparisons'] == 1
    changed = dict(batch, jet_pt=ak.Array([1.,2.]))
    with pytest.raises(ValueError, match='changed input fingerprint'):
        predictor(changed)
    predictor.test_scores['ParticleNet'][content] = .6
    with pytest.raises(ValueError, match='Strict saved-test score replay'):
        augment(input_path,tmp_path/'failed.root',predictor,{},score_branches=('ParticleNet_score',))
    assert not (tmp_path/'failed.audit.json').exists()


@pytest.mark.skipif(not os.environ.get('AK15_MODEL_SOURCE_DIR'), reason='Author architecture source not bundled; set AK15_MODEL_SOURCE_DIR')
@pytest.mark.parametrize('profile', ['compact', 'full'])
def test_frozen_model_roundtrip_and_tamper_guard(tmp_path, profile):
    models = Path(os.environ['AK15_MODEL_SOURCE_DIR'])
    torch.set_num_threads(2)
    torch.manual_seed(17)
    batch = dict(jet_pt=np.array([300., 320.]), jet_eta=np.array([0., .1]),
                 jet_phi=np.array([0., .2]), jet_mass=np.array([40., 45.]),
                 nPart=np.array([20, 22]), isData=np.array([False, True]),
                 event_key=ak.Array(['a', 'b']), abcdRegion=np.array([1, 2]))
    rows = []
    for i, n in enumerate(batch['nPart']):
        pt = np.arange(1, n+1, dtype=float)
        particles = dict(pt=pt, eta=pt*.001, phi=pt*.002, energy=pt*1.1,
                         px=pt, py=pt*.01, pz=pt*.1, pdgId=np.full(n,211),
                         charge=np.ones(n), fromPV=np.full(n,3))
        for name, values in particles.items():
            suffix = {'energy':'E','px':'PX','py':'PY','pz':'PZ'}.get(name,name)
            batch.setdefault('Part_'+suffix, []).append(values)
        rows.append(tensors(particles, *[batch[k][i] for k in ('jet_pt','jet_eta','jet_phi','jet_mass')]))
    x, v, mask = [np.stack([row[j] for row in rows]) for j in range(3)]
    index = dict(event_key=np.array(['a','b'],dtype='S64'),
                 content_key=np.array([fingerprint(row[:3]) for row in rows],dtype='S64'),
                 split=np.array([2,2]), label=np.array([0,1]),
                 physics_weight=np.array([1.,-1.]), optimization_weight=np.ones(2))
    trained, scores = {}, {}
    for name in ('ParticleNet','ParticleTransformer'):
        directory = tmp_path / name
        (directory/'indexed').mkdir(parents=True)
        (directory/'results').mkdir()
        np.savez(directory/'indexed/index.npz', **index)
        config = dict(model=name, profile=profile, epochs=1, cap=256, features=FEATURES,
                      seed=17, mean=[0.]*12, std=[1.]*12, architecture=architecture(name,profile),
                      model_sha256=sha256(models/(name+'.py')), index_sha256=sha256(directory/'indexed/index.npz'))
        model = model_for(name,models,profile).eval()
        with torch.no_grad():
            score = torch.softmax(forward(name,model,*[torch.from_numpy(a) for a in (x,v,mask,x[:,:2])]),1)[:,1].numpy()
        (directory/'results/configuration.json').write_text(json.dumps(config))
        (directory/'results/history.csv').write_text('epoch,train_loss,validation_loss\n1,0.7,0.6\n')
        summary = dict(model=name, best_epoch=1, test_rows=2, parameters=sum(p.numel() for p in model.parameters()),
                       auc=roc_auc_score(index['label'],score,sample_weight=index['optimization_weight']))
        (directory/'results/summary.json').write_text(json.dumps(summary))
        np.savez(directory/'results/test_scores.npz',score=score,**index)
        torch.save(dict(configuration=config,architecture=name,best_epoch=1,state_dict=model.state_dict()),directory/'results/best.pt')
        trained[name], scores[name] = directory, score
    for selected in (('ParticleNet',), ('ParticleTransformer',), ('ParticleNet','ParticleTransformer')):
        predictor = NewFrozenModels({n:trained[n] for n in selected}, models, 'cpu')
        result = predictor(batch)
        for name in selected:
            np.testing.assert_allclose(result[name+'_score'],scores[name],rtol=1e-6,atol=1e-7)
        np.testing.assert_array_equal(result['NN_inputSplit'],[2,-1])
        assert all(predictor.replay[name]['comparisons'] == 1 for name in selected)
    changed = copy.deepcopy(batch)
    changed['jet_pt'][0] += 10.
    with pytest.raises(ValueError, match='changed input fingerprint'):
        predictor(changed)
    # A different source identity matching a known capped input still gets its test membership.
    changed = copy.deepcopy(batch)
    changed['event_key'] = ak.Array(['different-source','b'])
    np.testing.assert_array_equal(predictor(changed)['NN_inputSplit'], [2,-1])
    # Altering the stored score while retaining its ordering/AUC must fail on actual inference.
    path = trained['ParticleNet']/'results/test_scores.npz'
    with np.load(path) as saved:
        values = {k:saved[k] for k in saved.files}
    values['score'] = values['score']*.99
    np.savez(path, **values)
    replay_predictor = NewFrozenModels({'ParticleNet':trained['ParticleNet']}, models, 'cpu')
    with pytest.raises(ValueError, match='Strict saved-test score replay'):
        replay_predictor(batch)
    config_path = trained['ParticleNet']/'results/configuration.json'
    bad = json.loads(config_path.read_text())
    bad['mean'][0] = 100.
    config_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='Checkpoint provenance'):
        validate_result(trained['ParticleNet'],models,'ParticleNet')
