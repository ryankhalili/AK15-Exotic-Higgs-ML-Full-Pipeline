import argparse
import json
import importlib.util
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest
from score_histograms import build
from test_pipeline import histogram_config
from plot_outputs import stack, limits


def test_stack_preserves_integral_and_display_only_signal(tmp_path):
    config = histogram_config(tmp_path)
    config_path = tmp_path/'inputs.json'
    config_path.write_text(json.dumps(config))
    build(config_path, tmp_path/'templates')
    args = argparse.Namespace(templates=tmp_path/'templates/histograms.root', lumi_pb=1000.,
                              signal_scale=1000., signal_label='Signal', log=True, ymin=.01,
                              label='ParticleNet')
    figure, values = stack(args)
    assert values['data'] == 400
    assert values['signal_display_scale'] == 1000
    before = values['prediction']
    plt.close(figure)
    args.signal_scale = 1
    figure, values = stack(args)
    assert values['prediction'] == before
    assert values['undefined_ratio_bins'] == []
    plt.close(figure)
    args.lumi_pb = 2000
    with pytest.raises(ValueError, match='Luminosity'):
        stack(args)


def test_expected_plot_rejects_observed_and_unsorted_bands(tmp_path):
    result = dict(status='EXPECTED_LIMIT_COMPUTED', observed_limit_run=False,
                  quantiles=[dict(quantileExpected=q, limit=i+1.) for i,q in enumerate((.025,.16,.5,.84,.975))])
    (tmp_path/'result.json').write_text(json.dumps(result))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([dict(label='Example', file='result.json')]))
    args = argparse.Namespace(manifest=manifest, label='Expected limits')
    figure, values = limits(args)
    np.testing.assert_array_equal(values['expected_quantiles'], [[1,2,3,4,5]])
    plt.close(figure)
    result['observed_limit_run'] = True
    (tmp_path/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='expected'):
        limits(args)


def test_model_fetch_adaptation_and_existing_destination(tmp_path):
    path = Path(__file__).resolve().parents[1]/'tools/fetch_models.py'
    spec = importlib.util.spec_from_file_location('fetch_models', path)
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    with pytest.raises(FileExistsError):
        fetch.fetch(tmp_path)
    text = b'from weaver.utils.logger import _logger\nx = 1\n'
    transformed = fetch.adapt('ParticleTransformer.py', text)
    assert transformed.endswith(b'x = 1\n')
    assert b'logging.getLogger(__name__)' in transformed
    assert fetch.adapt('ParticleNet.py', text) == text
    with pytest.raises(ValueError):
        fetch.adapt('ParticleTransformer.py', b'changed source')
