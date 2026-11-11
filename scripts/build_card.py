"""Build one ordinary TH1 pass-category card with CombineHarvester; never fit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re


PROCESSES = ('signal', 'QCD', 'WJets', 'DYJets', 'TT', 'ST', 'VV')
MC = tuple(p for p in PROCESSES if p != 'QCD')
AUTO_MC_STATS = 'pass autoMCStats 0 1 2'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def qcd_policy(document=None, fixed=False):
    require((document is not None) != fixed, 'Choose supplied QCD shapes OR fixed-QCD diagnostic')
    if fixed:
        return dict(mode='fixed_diagnostic', nuisances=[],
                    coverage_note='QCD fixed: its statistical and systematic uncertainties are omitted.',
                    unmodeled_correlations=['All QCD uncertainty correlations are omitted.'])
    require(isinstance(document, dict), 'QCD specification must be an object')
    require(set(document) == {'nuisances', 'coverage_note', 'unmodeled_correlations'},
            'QCD specification requires exactly nuisances, coverage_note, unmodeled_correlations')
    names = document['nuisances']
    require(isinstance(names, list) and 0 < len(names) <= 256, 'Supply 1..256 QCD shape nuisance names')
    require(all(isinstance(n, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', n)
                and n not in ('r', 'autoMCStats') and not n.startswith('prop_bin') for n in names),
            'Invalid or reserved nuisance name')
    require(len(set(names)) == len(names), 'Repeated QCD nuisance name')
    require(isinstance(document['coverage_note'], str) and document['coverage_note'].strip(),
            'Explain exactly which QCD uncertainties the supplied shapes represent')
    gaps = document['unmodeled_correlations']
    require(isinstance(gaps, list) and all(isinstance(x, str) and x.strip() for x in gaps),
            'unmodeled_correlations must be an explicit list of descriptions')
    return dict(document, mode='supplied_shapes')


def histogram_record(hist, name, reference_edges=None, need_sumw2=False, allow_zero=False):
    require(bool(hist) and hist.InheritsFrom('TH1') and hist.GetDimension() == 1,
            'Expected one-dimensional TH1: ' + name)
    n = hist.GetNbinsX()
    require(0 < n <= 10000, 'Invalid histogram bin count: ' + name)
    if need_sumw2:
        require(hist.GetSumw2N() == n + 2, 'MC requires explicitly stored Sumw2: ' + name)
    axis = hist.GetXaxis()
    edges = [float(axis.GetBinLowEdge(i)) for i in range(1, n + 2)]
    require(all(math.isfinite(v) for v in edges) and all(a < b for a, b in zip(edges, edges[1:])),
            'Invalid bin edges: ' + name)
    require(reference_edges is None or edges == reference_edges, 'Different binning: ' + name)
    values = [float(hist.GetBinContent(i)) for i in range(n + 2)]
    errors = [float(hist.GetBinError(i)) for i in range(n + 2)]
    require(all(math.isfinite(v) and v >= 0 for v in values + errors),
            'Nonfinite or negative bin/error: ' + name)
    require(all(v == 0 for v in (values[0], values[-1], errors[0], errors[-1])),
            'Nonempty underflow/overflow; fold explicitly upstream: ' + name)
    rate = float(hist.Integral(1, n))
    require(math.isfinite(rate) and math.isclose(rate, math.fsum(values[1:-1]), rel_tol=1e-12),
            'Integral inconsistent with visible bins: ' + name)
    require(rate >= 0 if allow_zero else rate > 0, 'Nonpositive template integral: ' + name)
    return dict(integral=rate, edges=edges, values=values[1:-1], errors=errors[1:-1])


def load_templates(root, path, policy):
    source = root.TFile.Open(str(path), 'READ')
    require(bool(source) and not source.IsZombie(), 'Cannot open TH1 input')
    histograms, records = {}, {}
    try:
        keys = ['data_obs'] + list(PROCESSES)
        keys += ['QCD_' + n + direction for n in policy['nuisances'] for direction in ('Up', 'Down')]
        edges = None
        for name in keys:
            obj = source.Get('pass/' + name)
            rec = histogram_record(obj, name, edges, name in MC, name == 'data_obs')
            if edges is None:
                edges = rec['edges']
            records[name] = rec
            histograms[name] = obj.Clone(name)
            histograms[name].SetDirectory(0)
        for nuisance in policy['nuisances']:
            require(any(records['QCD_' + nuisance + d]['values'] != records['QCD']['values']
                        for d in ('Up', 'Down')), 'Both variations are nominal: ' + nuisance)
        directory = source.GetDirectory('pass')
        present = {str(k.GetName()) for k in directory.GetListOfKeys()}
        unused = sorted(present - set(keys))
        require(not any(n.startswith('QCD_') and n.endswith(('Up', 'Down')) for n in unused),
                'Unlisted QCD variation templates; explicitly select or remove them in a new input')
        covariance_present = bool(source.Get('statistical_covariance/QCD'))
    finally:
        source.Close()
    # autoMCStats sees only MC errors; explicit QCD shapes are the sole QCD uncertainties.
    for name, hist in histograms.items():
        if name == 'QCD' or name.startswith('QCD_'):
            for i in range(hist.GetNbinsX() + 2):
                hist.SetBinError(i, 0.)
    return histograms, records, covariance_present, unused


def populate_harvester(ch, input_path, rates, policy):
    cb = ch.CombineHarvester()
    categories = [(0, 'pass')]
    cb.AddObservations(['*'], ['AK15'], ['2018'], ['score'], categories)
    cb.AddProcesses(['*'], ['AK15'], ['2018'], ['score'], list(PROCESSES[1:]), categories, False)
    cb.AddProcesses(['125'], ['AK15'], ['2018'], ['score'], ['signal'], categories, True)
    for name in policy['nuisances']:
        cb.cp().process(['QCD']).AddSyst(cb, name, 'shape', ch.SystMap()(1.0))
    cb.ExtractShapes(str(input_path), '$BIN/$PROCESS', '$BIN/$PROCESS_$SYSTEMATIC')
    cb.ForEachProc(lambda proc: proc.set_rate(rates[proc.process()]))
    cb.ForEachObs(lambda obs: obs.set_rate(rates['data_obs']))
    return cb


def finalize_card(text, rates, policy, hist_mode=2):
    require(hist_mode in (1, 2), 'Unsupported histogram normalization mode')
    lines = text.splitlines()
    rows = [line.split() for line in lines if line.strip() and not line.lstrip().startswith('#')]
    require(not any('autoMCStats' in row for row in rows), 'CH already emitted autoMCStats')
    process_rows = [r[1:] for r in rows if r[0] == 'process']
    named = [r for r in process_rows if set(r) == set(PROCESSES)]
    require(len(named) == 1 and len(named[0]) == len(PROCESSES), 'Unexpected CH process layout')
    order = named[0]
    numeric = [r for r in process_rows if all(re.fullmatch(r'-?\d+', x) for x in r)]
    require(len(numeric) == 1 and len(numeric[0]) == len(order), 'Missing CH process identifiers')
    require(all((int(i) <= 0) == (p == 'signal') for p, i in zip(order, numeric[0])),
            'Incorrect signal/background identifiers')
    require(all(set(r[1:]) == {'pass'} for r in rows if r[0] == 'bin'), 'Unexpected category')
    for nuisance in policy['nuisances']:
        matches = [r for r in rows if r[0] == nuisance]
        require(len(matches) == 1 and matches[0][1] == 'shape' and len(matches[0]) == 2 + len(order),
                'Missing or unexpected QCD shape row: ' + nuisance)
        for process, token in zip(order, matches[0][2:]):
            require(token == '-' if process != 'QCD' else float(token) == 1.,
                    'QCD shape applied to an unintended process: ' + nuisance)
    require(sum(r[0] == 'rate' for r in rows) == 1 and sum(r[0] == 'observation' for r in rows) == 1,
            'Unexpected CH rates/observation layout')
    # Preserve full integral precision even if CH's text writer rounds its rate column.
    for i, line in enumerate(lines):
        fields = line.split()
        if fields and fields[0] == 'rate':
            lines[i] = 'rate ' + ' '.join(format(rates[p], '.17g') for p in order)
        if fields and fields[0] == 'observation':
            lines[i] = 'observation ' + format(rates['data_obs'], '.17g')
    lines += ['# QCD TH1 errors are zero; only explicitly supplied QCD shapes are modeled.',
              '# Cross-process/control-region correlations are NOT inferred.',
              '# Fixed-QCD diagnostic only.' if policy['mode'] == 'fixed_diagnostic'
              else '# Supplied QCD shapes only; covariance completeness is NOT certified.',
              'pass autoMCStats 0 1 %d' % hist_mode]
    return '\n'.join(lines) + '\n'


def check_written_shapes(root, path, records):
    source = root.TFile.Open(str(path), 'READ')
    require(bool(source) and not source.IsZombie(), 'Cannot inspect CH output shapes')
    try:
        for name, original in records.items():
            # CH writes mass-labeled signal shapes, as referenced by its shapes line.
            written_name = 'signal125' if name == 'signal' else name
            rec = histogram_record(source.Get('pass/' + written_name), name, original['edges'],
                                   name in MC, name == 'data_obs')
            for key in ('integral', 'values', 'errors'):
                expected = [0.] * len(original['errors']) if key == 'errors' and (
                    name == 'QCD' or name.startswith('QCD_')) else original[key]
                got = rec[key]
                pairs = zip(got, expected) if isinstance(got, list) else [(got, expected)]
                require(all(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-12) for a, b in pairs),
                        'CH output changed ' + key + ': ' + name)
    finally:
        source.Close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--templates', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--hist-mode', type=int, choices=(2,), default=2)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--qcd-shapes', type=Path)
    group.add_argument('--fixed-qcd-diagnostic', action='store_true')
    args = parser.parse_args(argv)
    source, output = args.templates.resolve(), args.output.resolve()
    require(source.is_file() and source.stat().st_size <= 256 * 1024**2, 'Missing or >256MiB template input')
    require(not output.exists(), 'Output exists; preserve it and choose a new directory')
    document = None
    if args.qcd_shapes:
        require(args.qcd_shapes.stat().st_size <= 65536, 'QCD specification exceeds 64KiB')
        document = json.loads(args.qcd_shapes.read_text())
    policy = qcd_policy(document, args.fixed_qcd_diagnostic)
    input_hash = sha(source)
    # ROOT/CH are deliberately absent from import-time and ROOT-free unit tests.
    import ROOT
    import CombineHarvester.CombineTools.ch as ch
    ROOT.gROOT.SetBatch(True)
    histograms, records, covariance_present, unused = load_templates(ROOT, source, policy)
    require(sha(source) == input_hash, 'Template input changed during reading')
    output.mkdir(parents=True, exist_ok=False)
    intermediate = output / 'fit_inputs.root'
    handle = ROOT.TFile.Open(str(intermediate), 'CREATE')
    require(bool(handle) and not handle.IsZombie(), 'Cannot create fit input copy')
    try:
        handle.mkdir('pass').cd()
        for name, hist in histograms.items():
            require(hist.Write(name) > 0, 'Cannot write histogram: ' + name)
    finally:
        handle.Close()
    rates = {name: records[name]['integral'] for name in ('data_obs',) + PROCESSES}
    cb = populate_harvester(ch, intermediate, rates, policy)
    previous = Path.cwd()
    try:
        os.chdir(output)
        cb.WriteDatacard('datacard.txt', 'shapes.root')
    finally:
        os.chdir(previous)
    card = output / 'datacard.txt'
    card.write_text(finalize_card(card.read_text(), rates, policy, args.hist_mode))
    check_written_shapes(ROOT, output / 'shapes.root', records)
    report = dict(status='CARD_BUILT_NOT_FIT', input=str(source), input_sha256=input_hash,
                  builder_sha256=sha(Path(__file__)), category='pass', rates=rates,
                  qcd_policy=policy, qcd_bin_errors_in_fit=0,
                  qcd_covariance_object_present=covariance_present,
                  qcd_covariance_object_consumed=False, qcd_covariance_completeness_verified=False,
                  auto_mc_stats='pass autoMCStats 0 1 %d' % args.hist_mode,
                  histogram_normalization_mode=args.hist_mode, mc_stat_processes=list(MC),
                  unused_pass_objects=unused, histograms=records,
                  root_version=ROOT.gROOT.GetVersion(), combineharvester_module=str(ch.__file__),
                  inference_run=False, observed_limit_run=False, physics_clearance=False,
                  limitations=['No luminosity, calibration, detector/theory or other nuisances invented.',
                               'No QCD/MC, control-region, or cross-category correlations inferred.',
                               'No signal cross-section/branching-ratio interpretation supplied.',
                               'Do not combine these classifiers as independent event categories.'],
                  artifacts={p: sha(output / p) for p in ('datacard.txt', 'shapes.root', 'fit_inputs.root')})
    with (output / 'build_report.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=report['status'], output=str(output), rates=rates), indent=2))


if __name__ == '__main__':
    main()
