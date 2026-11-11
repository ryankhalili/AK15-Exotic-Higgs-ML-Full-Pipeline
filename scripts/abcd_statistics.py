"""ABCD score-shape estimate and first-order statistical covariance."""
import numpy as np


def predict(residual, variance):
    residual = np.asarray(residual, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if residual.ndim != 2 or residual.shape[0] != 4 or variance.shape != residual.shape:
        raise ValueError('Expected four regions with common score bins')
    if not np.isfinite(residual).all() or not np.isfinite(variance).all() or np.any(variance < 0):
        raise ValueError('Invalid yields or variances')
    B, C, D = residual[1:].sum(axis=1)
    if min(B, C, D) <= 0 or np.any(residual[3] < 0):
        raise ValueError('Nonpositive control yield or negative D score bin; review binning')
    di = residual[3]
    q = B * C * di / D**2
    j = np.column_stack((C * di / D**2, B * di / D**2,
                         B * C / D**2 * np.eye(len(di)) -
                         2 * B * C / D**3 * di[:, None] * np.ones((1, len(di)))))
    v = np.r_[variance[1].sum(), variance[2].sum(), variance[3]]
    covariance = (j * v) @ j.T
    total = B * C / D
    total_variance = ((C / D)**2 * v[0] + (B / D)**2 * v[1]
                      + (B * C / D**2)**2 * variance[3].sum())
    np.testing.assert_allclose(q.sum(), total, rtol=1e-12)
    np.testing.assert_allclose(covariance.sum(), total_variance, rtol=1e-11)
    return dict(values=q, covariance=covariance, jacobian=j,
                B=float(B), C=float(C), D=float(D), total=float(total),
                total_variance=float(total_variance))


def eigen_variations(values, covariance):
    values, covariance = np.asarray(values), np.asarray(covariance)
    if np.any(values <= 0):
        raise ValueError('QCD variations require positive bins; use reviewed coarser binning')
    w, vectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(w))), 1.)
    if np.min(w) < -1e-10 * scale:
        raise ValueError('QCD covariance is not positive semidefinite')
    variations = {}
    reconstructed = np.zeros_like(covariance)
    for i in np.flatnonzero(w > scale * 1e-12):
        shift = vectors[:, i] * np.sqrt(w[i])
        if np.any(values - np.abs(shift) <= 0):
            raise ValueError('QCD statistical variation crosses zero; review common binning')
        variations['QCDStat%02d' % (len(variations) + 1)] = (values + shift, values - shift)
        reconstructed += np.outer(shift, shift)
    np.testing.assert_allclose(reconstructed, covariance, rtol=1e-9, atol=scale * 1e-10)
    return variations
