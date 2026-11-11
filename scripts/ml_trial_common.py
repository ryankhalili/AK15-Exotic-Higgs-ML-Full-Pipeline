"""PF features and separate optimization/physics weight contracts."""
import hashlib
import numpy as np

FEATURES = ["deta", "dphi", "log_pt", "log_pt_fraction", "log_energy_fraction",
            "charge", "fromPV", "is_charged_hadron", "is_neutral_hadron",
            "is_photon", "is_electron", "is_muon"]
CAP = 256


def event_key(sample, source, entry):
    return hashlib.sha256((sample + "|" + source + "|" + str(int(entry))).encode()).hexdigest()


def split_key(key):
    value = int(key[:16], 16) % 10
    return 0 if value < 8 else value - 7


def require_unique_inputs(x, v, mask):
    """Reject repeated physical inputs even when source-derived keys differ."""
    if not len(x) == len(v) == len(mask):
        raise ValueError("Inconsistent input lengths")
    seen = {}
    for i in range(len(x)):
        digest = hashlib.sha256()
        for array, dtype in [(x, "<f4"), (v, "<f4"), (mask, "bool")]:
            row = np.asarray(array[i], dtype=dtype).copy()
            row[row == 0] = 0
            digest.update(row.tobytes())
        key = digest.digest()
        if key in seen:
            raise ValueError("Identical network inputs at rows %d and %d; "
                             "resolve provenance and split leakage before training" % (seen[key], i))
        seen[key] = i


def tensors(p, jet_pt, jet_eta, jet_phi, jet_mass, cap=CAP):
    """Identical reconstructed-PF features for old flat and ordinary Nano inputs."""
    if not isinstance(cap, int) or cap < 1:
        raise ValueError("Constituent cap must be a positive integer")
    order = np.argsort(-np.asarray(p["pt"]), kind="stable")[:cap]
    if not len(order):
        raise ValueError("Selected jet has no PF constituents")
    q = {k: np.asarray(v)[order] for k, v in p.items()}
    if not all(np.isfinite(v).all() for v in q.values()):
        raise ValueError("Nonfinite constituent")
    jet_energy = np.sqrt((jet_pt * np.cosh(jet_eta)) ** 2 + jet_mass ** 2)
    deta = q["eta"] - jet_eta
    dphi = np.arctan2(np.sin(q["phi"] - jet_phi), np.cos(q["phi"] - jet_phi))
    pid = abs(q["pdgId"])
    feat = np.stack([deta, dphi, np.log(np.maximum(q["pt"], 1.e-6)),
                     np.log(np.maximum(q["pt"] / jet_pt, 1.e-8)),
                     np.log(np.maximum(q["energy"] / jet_energy, 1.e-8)),
                     q["charge"], q["fromPV"] / 3., pid == 211, pid == 130,
                     pid == 22, pid == 11, pid == 13]).astype("float32")
    vec = np.stack([q[k] for k in ["px", "py", "pz", "energy"]]).astype("float32")
    x = np.zeros((len(FEATURES), cap), dtype="float32")
    v = np.zeros((4, cap), dtype="float32")
    mask = np.zeros((1, cap), dtype=bool)
    x[:, :len(order)], v[:, :len(order)], mask[:, :len(order)] = feat, vec, True
    return x, v, mask, len(p["pt"])


def physics_weights(genweight, denominator, xsec, factor, luminosity_pb):
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Invalid processed sumGenWeights")
    if not all(np.isfinite(v) and v > 0 for v in (xsec, factor, luminosity_pb)):
        raise ValueError("Positive finite normalization constants required")
    return np.asarray(genweight, dtype="float64") * luminosity_pb * xsec * factor / denominator


def training_weights(labels, samples, physics, train):
    """Positive optimization weights; signed physics weights remain untouched."""
    w = abs(np.asarray(physics, dtype="float64"))
    signal_samples = np.unique(samples[labels == 1])
    for sample in signal_samples:
        take = (samples == sample) & (labels == 1)
        norm = w[take & train].sum()
        if norm <= 0:
            raise ValueError("Signal mass missing from training")
        w[take] /= norm * len(signal_samples)
    background_norm = w[(labels == 0) & train].sum()
    if background_norm <= 0:
        raise ValueError("No weighted background in training")
    w[labels == 0] /= background_norm
    return w * train.sum() / 2.
