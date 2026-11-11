"""Integrity checks shared by the two training entry points."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def strict_precision(torch):
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '1':
        raise ValueError('TF32 force-enable override is incompatible with this score contract')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_index(directory):
    directory = Path(directory).resolve()
    config = json.loads((directory / 'configuration.json').read_text())
    path = directory / 'index.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != config['index_sha256']:
        raise ValueError('Index hash mismatch')
    with np.load(path, allow_pickle=False) as data:
        index = {key: data[key] for key in data.files}
    if (config['cap'] != 256 or not np.isfinite(config['mean']).all()
            or not np.isfinite(config['std']).all() or np.any(np.array(config['std']) <= 0)
            or not np.isin(index['split'], [0, 1, 2]).all()
            or not np.isin(index['label'], [0, 1]).all()
            or not np.isfinite(index['optimization_weight']).all()
            or np.any(index['optimization_weight'] <= 0)):
        raise ValueError('Invalid indexed training contract')
    for split in (0, 1, 2):
        if set(index['label'][index['split'] == split]) != {0, 1}:
            raise ValueError('Each partition requires both classes')
    for shard in config['shards']:
        location = (directory / shard['directory']).resolve()
        for key, expected in shard['sha256'].items():
            path = location / (shard['stem'] + '_' + key + '.npy')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Shard hash mismatch: ' + path.name)
        shard['directory'] = str(location)
    return config, index
