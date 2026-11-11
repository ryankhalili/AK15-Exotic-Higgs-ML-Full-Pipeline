"""Bind one Nano/compact pair to an explicit sample before attaching PF inputs."""
import argparse
import json
from pathlib import Path
import uproot
from make_selected_pf_ntuple import file_hash, validate_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nano', type=Path, required=True)
    parser.add_argument('--compact', type=Path, required=True)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--is-data', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new spec path')
    digest = file_hash(args.nano)
    spec = dict(key=args.sample + ':' + digest, sample=args.sample, is_data=bool(args.is_data),
                source='sha256:' + digest, nano_sha256=digest, compact_sha256=file_hash(args.compact))
    with uproot.open(args.compact) as root:
        metadata = root['AnalysisMetadata'].arrays(library='np')
        labels = root['ABCDEvents'].arrays(['sample'], library='np')
        validate_sample(spec, metadata, labels)
    with args.output.open('x') as stream:
        json.dump(spec, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
