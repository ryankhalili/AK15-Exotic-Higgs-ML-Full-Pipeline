import importlib.util
from pathlib import Path
import numpy as np
import pytest
from make_selected_pf_ntuple import file_hash, validate_source, validate_sample, validate_selection


def selected():
    values = dict(inputEntry=0, isData=0, selectedMuonTightId=1, selectedMuonPt=60.,
                  selectedMuonEta=0., selectedMuonIso=.1, jet_pt=300., jet_eta=0.,
                  jet_phi=0., jet_mass=40., eventHT=300., met_pt=50.,
                  muonJetDeltaPhi=2., metJetDeltaPhi=2., muonMetTransverseMass=70.,
                  abcdRegion=1, genWeight=1., hasGenWeight=1)
    return {key: np.asarray([value]) for key, value in values.items()}


def test_selected_nonfinite_and_negative_entries():
    metadata = dict(abcdMode=[1], metBranchPrefix=['MET'], muonEtaMax=[2.4])
    validate_selection(selected(), metadata, False)
    for name in ('genWeight', 'jet_phi', 'muonMetTransverseMass'):
        for value in (float('nan'), float('inf')):
            row = selected()
            row[name] = np.array([value])
            with pytest.raises(ValueError, match='Nonfinite'):
                validate_selection(row, metadata, False)
    row = selected()
    row['inputEntry'][0] = -1
    with pytest.raises(ValueError, match='Negative source'):
        validate_selection(row, metadata, False)


@pytest.mark.parametrize('field,value,message', [
    ('isData', 1, 'Data flag'),
    ('selectedMuonTightId', 0, 'tight ID'),
    ('selectedMuonPt', 55., 'Muon pT'),
    ('selectedMuonEta', -2.4, 'Muon abs'),
    ('selectedMuonIso', .5, 'isolation'),
    ('jet_pt', 200., 'AK15 pT'),
    ('jet_eta', 3., 'AK15 abs'),
    ('eventHT', 200., 'HT'),
    ('met_pt', 30., 'PF MET'),
    ('muonJetDeltaPhi', -1.5, 'Muon-jet'),
    ('metJetDeltaPhi', -1., 'MET-jet'),
    ('abcdRegion', 2, 'ABCD region'),
    ('hasGenWeight', 0, 'generator weights'),
])
def test_selection_boundaries(field, value, message):
    metadata = dict(abcdMode=[1], metBranchPrefix=['MET'], muonEtaMax=[2.4])
    row = selected()
    row[field] = np.asarray([value])
    with pytest.raises(ValueError, match=message):
        validate_selection(row, metadata, False)


def test_parent_checksums_and_sample_binding(tmp_path):
    nano, compact = tmp_path/'nano.bin', tmp_path/'compact.bin'
    nano.write_bytes(b'parent')
    compact.write_bytes(b'reduction')
    spec = dict(key='example', sample='signal', is_data=False, nano_sha256=file_hash(nano),
                compact_sha256=file_hash(compact), source='sha256:'+file_hash(nano))
    validate_source(spec, nano, compact)
    validate_sample(spec, {'sampleLabel':['signal']}, {})
    validate_sample(spec, {}, {'sample':['signal']})
    with pytest.raises(ValueError, match='sample'):
        validate_sample(spec, {'sampleLabel':['other']}, {})
    with pytest.raises(ValueError, match='provenance'):
        validate_sample(spec, {}, {'sample':[]})
    with pytest.raises(ValueError, match='Source identity'):
        validate_source(dict(spec, source='invented'), nano, compact)
    nano.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        validate_source(spec, nano, compact)


def test_publication_set_and_synthetic_credentials(tmp_path):
    location = Path(__file__).resolve().parents[1]/'tools/audit_sources.py'
    module_spec = importlib.util.spec_from_file_location('audit', location)
    audit = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(audit)
    source = tmp_path/'example.py'
    source.write_text('token = '+repr('github_'+'pat_'+'a'*60))
    assert any(row['reason'] == 'fine_grained_token' for row in audit.scan(tmp_path)[1])
    source.write_text('url = '+repr('https'+'://'+'user'+':password'+'@example.invalid'))
    assert any(row['reason'] == 'url_credentials' for row in audit.scan(tmp_path)[1])
    source.write_text('value = 1\n')
    (tmp_path/'.work').mkdir()
    (tmp_path/'.work/private.py').write_text('value = 2\n')
    assert not audit.scan(tmp_path)[1]
    assert any(row['reason'] == 'excluded artifact in publication set'
               for row in audit.scan(tmp_path, exact_paths=[Path('.work/private.py')])[1])
