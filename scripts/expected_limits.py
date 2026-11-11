"""Build a checked TH1 card/workspace and run expected-only asymptotic limits."""
import argparse
import json
import math
from pathlib import Path
import subprocess
from build_card import main as build_card
from limit_checks import require, validate_quantiles, validate_workspace_response


def commands(rmax):
    require(math.isfinite(rmax) and rmax > 0, 'Positive finite rMax required')
    return [
        ['text2workspace.py', 'datacard.txt', '-m', '125', '-o', 'workspace.root'],
        ['combine', '-M', 'AsymptoticLimits', 'workspace.root', '-m', '125', '--run', 'blind',
         '--rMin', '0', '--rMax', str(rmax), '-n', '.ExpectedOnly'],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--templates', type=Path, required=True)
    parser.add_argument('--qcd-shapes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rmax', type=float, required=True)
    args = parser.parse_args()
    actions = commands(args.rmax)
    require(not args.output.exists(), 'Choose a new output directory')
    import ROOT
    ROOT.gROOT.SetBatch(True)
    build_card(['--templates', str(args.templates), '--qcd-shapes', str(args.qcd_shapes),
                '--hist-mode', '2', '--output', str(args.output)])
    for index, command in enumerate(actions):
        (args.output / ('command_%d.json' % index)).write_text(json.dumps(command) + '\n')
        with (args.output / ('command_%d.log' % index)).open('x') as log:
            subprocess.run(command, cwd=args.output, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=5400)
        if index == 0:
            validate_workspace_response(ROOT, args.output)
    output = args.output / 'higgsCombine.ExpectedOnly.AsymptoticLimits.mH125.root'
    source = ROOT.TFile.Open(str(output), 'READ')
    require(source and not source.IsZombie(), 'Unreadable limit output')
    try:
        tree = source.Get('limit')
        require(tree and tree.InheritsFrom('TTree'), 'Missing limit tree')
        rows = [dict(quantileExpected=float(row.quantileExpected), limit=float(row.limit)) for row in tree]
    finally:
        source.Close()
    quantiles = validate_quantiles(rows, args.rmax)
    result = dict(status='EXPECTED_LIMIT_COMPUTED', quantiles=quantiles, median_r95=quantiles[2]['limit'],
                  observed_limit_run=False, physics_interpretation_ready=False,
                  scope='Signal strength relative to the supplied signal template, conditional on the stated statistical model.')
    (args.output / 'limit_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
