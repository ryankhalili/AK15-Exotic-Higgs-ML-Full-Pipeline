"""Reusable numerical helpers extracted from validated source."""
import numpy as np


def qcd_spec(names):
    return dict(nuisances=list(names),
        coverage_note='First-order joint statistical covariance of subtracted B and C totals and D score bins. Data counting errors and signed MC sumw2 included. MC A errors use autoMCStats separately.',
        unmodeled_correlations=['Residual isolation/transverse-mass and score-shape transfer correlations are not calibrated.',
                               'Efficiency and detector/theory systematic correlations are not included.',
                               'Signal contamination of control regions is not included in this one-category model.'])

def th1(name, values, variance, edges):
    from uproot.writing.identify import to_TAxis, to_TH1x
    values, variance = np.asarray(values, dtype=float), np.asarray(variance, dtype=float)
    centers = .5 * (edges[1:] + edges[:-1])
    return to_TH1x(fName=name, fTitle=name, data=np.r_[0., values, 0.],
        fEntries=0., fTsumw=float(values.sum()), fTsumw2=float(variance.sum()),
        fTsumwx=float(np.dot(values, centers)), fTsumwx2=float(np.dot(values, centers**2)),
        fSumw2=np.r_[0., variance, 0.],
        fXaxis=to_TAxis(fName='xaxis', fTitle='Classifier score', fNbins=len(values),
                       fXmin=float(edges[0]), fXmax=float(edges[-1]), fXbins=edges))
