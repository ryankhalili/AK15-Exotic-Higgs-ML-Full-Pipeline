"""Fetch checksum-pinned author models and licenses into a new local directory."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = '34e2ee20ca7d1b94446f1bf18b361c5865c4b6d4'
BASE = 'https://raw.githubusercontent.com/hqucms/weaver-core/' + COMMIT + '/'
FILES = {
    'ParticleNet.py': (BASE + 'weaver/nn/model/ParticleNet.py', '73bcef0afc4250d4d653bb6b362040bebddce5effa8679e4420864395a5d1c5d'),
    'ParticleTransformer.py': (BASE + 'weaver/nn/model/ParticleTransformer.py', 'eeb0b42027be451604d8ad0be7b5a31cb581329903f566a5d1071b08fef93502'),
    'LICENSE.weaver.txt': (BASE + 'LICENSE', '363378725d9f4bcbd9ecc5c052baa10fc72b671d60224c61ca3e95d879020164'),
    'LICENSE.dgcnn.txt': ('https://raw.githubusercontent.com/WangYueFt/dgcnn/f765b469a67730658ba554e97dc11723a7bab628/LICENSE', '288c5357e9620f022174625a153eb2423d5c14a8d9838bb4a9ef3deadf10549d'),
}


def adapt(name, content):
    if name == 'ParticleTransformer.py':
        old = b'from weaver.utils.logger import _logger'
        if content.count(old) != 1:
            raise ValueError('Unexpected logger import')
        return content.replace(old, b'import logging\n_logger = logging.getLogger(__name__)')
    return content


def fetch(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Choose a new architecture directory')
    downloads, receipt = {}, {}
    for name, (url, digest) in FILES.items():
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read(250001)
        if len(content) > 250000 or hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('Upstream source checksum mismatch: ' + name)
        downloads[name] = adapt(name, content)
        receipt[name] = dict(url=url, upstream_sha256=digest,
                             installed_sha256=hashlib.sha256(downloads[name]).hexdigest())
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in downloads.items():
        with (destination / name).open('xb') as stream:
            stream.write(content)
    with (destination / 'upstream.json').open('x') as stream:
        json.dump(dict(commit=COMMIT, files=receipt,
            modification='ParticleTransformer logger import uses the Python standard library; no numerical code changes.'), stream, indent=2)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    fetch(args.destination)
    print('Installed two pinned models and their license notices in ' + str(args.destination))


if __name__ == '__main__':
    main()
