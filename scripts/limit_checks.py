"""Reusable numerical helpers extracted from validated source."""
import json
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)

def validate_quantiles(rows, rmax):
    targets = (.025, .16, .5, .84, .975)
    require(len(rows) == 5, 'Expected exactly five expected quantiles, no observed result')
    ordered = sorted(rows, key=lambda row: row['quantileExpected'])
    for row, target in zip(ordered, targets):
        require(abs(row['quantileExpected'] - target) < .001, 'Unexpected confidence quantile')
        require(math.isfinite(row['limit']) and 0 < row['limit'] < .9*rmax,
                'Nonpositive/nonfinite limit or search-range boundary')
    require(all(a['limit'] <= b['limit'] for a, b in zip(ordered, ordered[1:])),
            'Expected quantiles are not ordered')
    return ordered

def validate_workspace_response(ROOT, result):
    report = json.loads((result/'build_report.json').read_text())
    handle = ROOT.TFile.Open(str(result/'workspace.root'), 'READ')
    require(handle and not handle.IsZombie(), 'Unreadable generated workspace')
    checks = []
    try:
        w = handle.Get('w')
        x = w.var('CMS_th1x')
        morph = w.function('shapeBkg_pass_QCD_morph')
        norm = w.function('n_exp_final_binpass_proc_QCD') or w.arg('n_exp_binpass_proc_QCD')
        require(x and morph and norm, 'Missing generated QCD function or normalization')
        names = report['qcd_policy']['nuisances']
        for name in names:
            require(w.var(name), 'Missing QCD nuisance: '+name)
            w.var(name).setVal(0.)
        original_x = x.getVal()
        try:
            for name, shift, key in [('central', 0., 'QCD')] + [
                    (name, shift, 'QCD_'+name+direction) for name in names
                    for shift, direction in [(1., 'Up'), (-1., 'Down')]]:
                if name != 'central':
                    w.var(name).setVal(shift)
                expected = report['histograms'][key]['values']
                values = []
                for i in range(len(expected)):
                    x.setVal(i+.5)
                    values.append(float(morph.getVal()*norm.getVal()))
                checks.append(dict(name=name, shift=shift, expected=expected, actual=values,
                    expected_integral=math.fsum(expected), actual_integral=math.fsum(values),
                    max_abs_delta=max(abs(a-b) for a,b in zip(values,expected))))
                if name != 'central':
                    w.var(name).setVal(0.)
        finally:
            x.setVal(original_x)
    finally:
        handle.Close()
    with (result/'workspace_response.json').open('x') as out:
        json.dump(dict(checks=checks, relative_tolerance=2e-6, absolute_tolerance=1e-5), out, indent=2)
    for check in checks:
        require(all(math.isclose(a,b,rel_tol=2e-6,abs_tol=1e-5)
                    for a,b in zip(check['actual'],check['expected'])),
                'Workspace changed QCD bin response: '+check['name']+' '+str(check['shift']))
    return checks
