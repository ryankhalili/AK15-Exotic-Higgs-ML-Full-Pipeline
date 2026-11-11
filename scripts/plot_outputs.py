"""Plot saved score TH1s or explicitly listed expected-limit summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import uproot
from build_card import require

BACKGROUNDS = [('QCD', 'QCD', '#d8449d'), ('WJets', 'W + jets', '#edcf47'),
               ('DYJets', 'DY + jets', '#478dca'), ('TT', 'Top pair', '#6c5ba1'),
               ('ST', 'Single top', '#52a47d'), ('VV', 'WW / WZ / ZZ', '#ce8948')]


def load_histograms(path):
    rows = {}
    with uproot.open(path) as root:
        for name in ['data_obs', 'signal'] + [name for name, _, _ in BACKGROUNDS]:
            hist = root['pass/' + name]
            y, edges = hist.to_numpy()
            variance = hist.variances()
            require(variance is not None and np.isfinite(y).all() and np.isfinite(variance).all()
                    and np.all(y >= 0) and np.all(variance >= 0), 'Invalid TH1: ' + name)
            require(np.all(hist.values(flow=True)[[0,-1]] == 0), 'Unfolded flow bins')
            if rows:
                np.testing.assert_array_equal(edges, rows['data_obs'][2])
            rows[name] = (y, variance, edges)
    return rows


def stack(args):
    rows = load_histograms(args.templates)
    require(np.isfinite(args.lumi_pb) and args.lumi_pb > 0, 'Explicit positive luminosity required')
    require(np.isfinite(args.signal_scale) and args.signal_scale > 0, 'Invalid display scale')
    receipt_path = args.templates.parent / 'histogram_receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        require(np.isclose(receipt['luminosity_pb'], args.lumi_pb, rtol=1e-12, atol=0), 'Luminosity label differs from template receipt')
    data, _, edges = rows['data_obs']
    signal = rows['signal'][0]
    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    prediction = sum(rows[name][0] for name, _, _ in BACKGROUNDS)
    require(prediction.sum() > 0, 'Empty background prediction')
    variance = sum(rows[name][1] for name, _, _ in BACKGROUNDS)
    error = np.sqrt(variance)
    figure, (upper, lower) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True,
        height_ratios=(3.4, 1), constrained_layout=True)
    bottom = np.zeros_like(data)
    for name, label, color in BACKGROUNDS:
        values = rows[name][0]
        upper.bar(edges[:-1], values, width=widths, bottom=bottom, align='edge', color=color, label=label)
        bottom += values
    upper.fill_between(edges, np.r_[prediction-error, (prediction-error)[-1]],
                       np.r_[prediction+error, (prediction+error)[-1]], step='post',
                       facecolor='none', edgecolor='#555555', hatch='///', label='Statistical uncertainty')
    upper.stairs(signal * args.signal_scale, edges, color='#a60000', lw=2,
                 label=args.signal_label + ' x ' + format(args.signal_scale, 'g'))
    upper.errorbar(centers, data, yerr=np.sqrt(data), color='black', fmt='o', ms=3.5, label='Data')
    upper.set_title(args.label, loc='left')
    upper.set_title(format(args.lumi_pb/1000., '.3f') + r' fb$^{-1}$', loc='right')
    upper.set_ylabel('Events / bin')
    if args.log:
        require(args.ymin > 0, 'Log minimum must be positive')
        upper.set_yscale('log')
        upper.set_ylim(args.ymin, max(np.max(data), np.max(signal*args.signal_scale), np.max(prediction+error))*12)
    else:
        upper.set_ylim(0, max(np.max(data), np.max(signal*args.signal_scale), np.max(prediction+error))*1.4)
    upper.legend(frameon=False, ncol=2, fontsize=9)
    valid = prediction > 0
    ratio = np.divide(data, prediction, out=np.full_like(data, np.nan), where=valid)
    ratio_error = np.divide(np.sqrt(data), prediction, out=np.zeros_like(data), where=valid)
    relative_error = np.divide(error, prediction, out=np.zeros_like(data), where=valid)
    lower.fill_between(edges, np.r_[1-relative_error, 1-relative_error[-1]],
                       np.r_[1+relative_error, 1+relative_error[-1]], step='post', color='#dddddd')
    lower.errorbar(centers[valid], ratio[valid], yerr=ratio_error[valid], color='black', fmt='o', ms=3.5)
    lower.axhline(1, color='black', lw=1)
    lower.set_ylim(0, max(2., float(np.max((ratio+ratio_error)[valid]))*1.1) if valid.any() else 2.)
    lower.set_xlabel(args.label + ' signal score')
    lower.set_ylabel('Data / pred.')
    lower.set_xlim(edges[0], edges[-1])
    return figure, dict(data=float(data.sum()), prediction=float(prediction.sum()),
                        data_prediction=float(data.sum()/prediction.sum()),
                        signal_display_scale=args.signal_scale, undefined_ratio_bins=np.flatnonzero(~valid).tolist())


def limits(args):
    records = json.loads(args.manifest.read_text())
    require(isinstance(records, list) and records, 'Provide explicit labeled limit summaries')
    require(len({row['label'] for row in records}) == len(records), 'Repeated plot label')
    quantiles = []
    for row in records:
        result = json.loads((args.manifest.parent / row['file']).read_text())
        require(result['status'] == 'EXPECTED_LIMIT_COMPUTED' and result['observed_limit_run'] is False,
                'Only successful expected limits are supported')
        values = sorted(result['quantiles'], key=lambda q: q['quantileExpected'])
        require(len(values) == 5 and np.allclose([v['quantileExpected'] for v in values],
                [.025,.16,.5,.84,.975], atol=.001, rtol=0), 'Invalid expected quantiles')
        numbers = np.array([v['limit'] for v in values])
        require(np.isfinite(numbers).all() and np.all(numbers > 0) and np.all(np.diff(numbers) >= 0), 'Invalid limits')
        quantiles.append(numbers)
    figure, ax = plt.subplots(figsize=(8, max(3.5, len(records)*.5+1.5)), constrained_layout=True)
    for i, (lo2, lo1, median, hi1, hi2) in enumerate(quantiles):
        ax.plot([lo2,hi2], [i,i], color='#edcf36', lw=16, solid_capstyle='butt')
        ax.plot([lo1,hi1], [i,i], color='#68b95e', lw=16, solid_capstyle='butt')
        ax.plot(median, i, '|', color='black', ms=21, markeredgewidth=2)
    ax.set_yticks(range(len(records)), [row['label'] for row in records])
    ax.set_ylim(len(records)-.4, -.6)
    ax.set_xlim(left=0)
    ax.set_xlabel('Expected 95% CL upper limit on signal strength r')
    ax.set_title(args.label, loc='left')
    ax.grid(axis='x', alpha=.2)
    ax.legend(handles=[Patch(color='#68b95e', label='68% expected band'),
                       Patch(color='#edcf36', label='95% expected band')], frameon=False)
    return figure, dict(labels=[row['label'] for row in records], expected_quantiles=np.asarray(quantiles).tolist(), expected_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('stack')
    p.add_argument('--templates', type=Path, required=True)
    p.add_argument('--lumi-pb', type=float, required=True)
    p.add_argument('--signal-label', required=True)
    p.add_argument('--signal-scale', type=float, default=1.)
    p.add_argument('--log', action='store_true')
    p.add_argument('--ymin', type=float, default=100.)
    p = commands.add_parser('limits')
    p.add_argument('--manifest', type=Path, required=True)
    for p in commands.choices.values():
        p.add_argument('--label', required=True)
        p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Choose a new plot directory')
    figure, receipt = stack(args) if args.command == 'stack' else limits(args)
    args.output.mkdir(parents=True, exist_ok=False)
    for extension in ('png','pdf'):
        figure.savefig(args.output/(args.command+'.'+extension), dpi=180)
    plt.close(figure)
    with (args.output/'values.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
