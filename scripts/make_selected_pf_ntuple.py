"""Attach all selected-jet PF constituents to audited PF-MET ABCD events."""
import argparse
import hashlib
import json
from pathlib import Path
import awkward as ak
import numpy as np
import uproot

PF = ["pt", "eta", "phi", "mass", "energy", "px", "py", "pz", "pdgId", "charge", "fromPV"]
IDS = ["run", "luminosityBlock", "event"]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_source(spec, nano, compact):
    if type(spec.get('is_data')) is not bool or not isinstance(spec.get('sample'), str) or not spec['sample'].strip():
        raise ValueError('Explicit data flag and nonempty sample required')
    if not isinstance(spec.get('key'), str) or not spec['key'].strip():
        raise ValueError('Nonempty source key required')
    for name, path in [('nano', nano), ('compact', compact)]:
        expected = spec.get(name + '_sha256', '')
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected) or file_hash(path) != expected:
            raise ValueError('Input checksum mismatch: ' + name)
    if spec.get('source') != 'sha256:' + spec['nano_sha256']:
        raise ValueError('Source identity must be bound to the parent Nano checksum')


def validate_sample(spec, metadata, arrays):
    if 'sampleLabel' in metadata:
        names = [str(value) for value in metadata['sampleLabel']]
    elif 'sample' in arrays and len(arrays['sample']):
        names = [str(value) for value in arrays['sample']]
    else:
        raise ValueError('Compact sample provenance absent; regenerate compact metadata')
    if not names or set(names) != {spec['sample']}:
        raise ValueError('Compact sample does not match the source specification')


def validate_selection(a, metadata, is_data):
    if not np.all(a['inputEntry'] >= 0):
        raise ValueError('Negative source entry')
    for name in ('genWeight', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_mass',
                 'selectedMuonPt', 'selectedMuonEta', 'selectedMuonIso',
                 'eventHT', 'met_pt', 'muonJetDeltaPhi', 'metJetDeltaPhi',
                 'muonMetTransverseMass'):
        if not np.isfinite(a[name]).all():
            raise ValueError('Nonfinite selected field: ' + name)
    if not (int(metadata["abcdMode"][0]) == 1): raise ValueError('Input invariant failed: int(metadata["abcdMode"][0]) == 1')
    if not (str(metadata["metBranchPrefix"][0]) == "MET"): raise ValueError('Input invariant failed: str(metadata["metBranchPrefix"][0]) == "MET"')
    if not (np.isclose(metadata["muonEtaMax"][0], 2.4)): raise ValueError('Input invariant failed: np.isclose(metadata["muonEtaMax"][0], 2.4)')
    if not (np.all(a["isData"] == int(is_data))): raise ValueError('Input invariant failed: np.all(a["isData"] == int(is_data))')
    if not (np.all(a["selectedMuonTightId"] == 1)): raise ValueError('Input invariant failed: np.all(a["selectedMuonTightId"] == 1)')
    if not (np.all(a["selectedMuonPt"] > 55)): raise ValueError('Input invariant failed: np.all(a["selectedMuonPt"] > 55)')
    if not (np.all(abs(a["selectedMuonEta"]) < 2.4)): raise ValueError('Input invariant failed: np.all(abs(a["selectedMuonEta"]) < 2.4)')
    if not (np.all(a["selectedMuonIso"] < .5)): raise ValueError('Input invariant failed: np.all(a["selectedMuonIso"] < .5)')
    if not (np.all(a["jet_pt"] > 200) and np.all(abs(a["jet_eta"]) < 3)): raise ValueError('Input invariant failed: np.all(a["jet_pt"] > 200) and np.all(abs(a["jet_eta"]) < 3)')
    if not (np.all(a["eventHT"] > 200) and np.all(a["met_pt"] > 30)): raise ValueError('Input invariant failed: np.all(a["eventHT"] > 200) and np.all(a["met_pt"] > 30)')
    if not (np.all(abs(a["muonJetDeltaPhi"]) > 1.5)): raise ValueError('Input invariant failed: np.all(abs(a["muonJetDeltaPhi"]) > 1.5)')
    if not (np.all(abs(a["metJetDeltaPhi"]) > 1)): raise ValueError('Input invariant failed: np.all(abs(a["metJetDeltaPhi"]) > 1)')
    region = np.where(a["selectedMuonIso"] < .15, 1, 3) + (a["muonMetTransverseMass"] <= 50)
    if not (np.array_equal(region, a["abcdRegion"])): raise ValueError('Input invariant failed: np.array_equal(region, a["abcdRegion"])')
    if not (len(set(a["inputEntry"].tolist())) == len(region)): raise ValueError('Input invariant failed: len(set(a["inputEntry"].tolist())) == len(region)')
    if is_data:
        if not (np.all(a["genWeight"] == 1)): raise ValueError('Input invariant failed: np.all(a["genWeight"] == 1)')
    else:
        if not (np.all(a["hasGenWeight"] == 1)): raise ValueError('Input invariant failed: np.all(a["hasGenWeight"] == 1)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("nano")
    ap.add_argument("compact")
    ap.add_argument("output")
    ap.add_argument("--audit", type=Path, help="Default: OUTPUT.audit.json")
    args = ap.parse_args()
    audit = args.audit or Path(args.output).with_suffix('.audit.json')
    if Path(args.output).exists() or audit.exists():
        raise FileExistsError('Choose new output and audit paths')
    spec = json.loads(Path(args.spec).read_text())
    validate_source(spec, args.nano, args.compact)
    with uproot.open(args.compact) as compact:
        arrays = compact["ABCDEvents"].arrays(library="np")
        # String object arrays are metadata, not constituent features.
        a = {k: v for k, v in arrays.items() if v.dtype.kind != "O"}
        meta = compact["AnalysisMetadata"].arrays(library="np")
        norm = compact["normalization"].values()
        if len(norm) < 2 or not np.isfinite(norm).all() or norm[0] < 0 or int(norm[0]) != norm[0]:
            raise ValueError('Nonfinite or invalid preselection normalization')
        validate_sample(spec, meta, arrays)
        validate_selection(a, meta, spec["is_data"])
        if "expected_selected" in spec:
            if not (len(a["inputEntry"]) == spec["expected_selected"]): raise ValueError('Input invariant failed: len(a["inputEntry"]) == spec["expected_selected"]')
        if "expected_events" in spec:
            if not (int(norm[0]) == spec["expected_events"]): raise ValueError('Input invariant failed: int(norm[0]) == spec["expected_events"]')
        with uproot.open(args.nano) as nano:
            tree = nano["Events"]
            if not (tree.num_entries == int(norm[0])): raise ValueError('Input invariant failed: tree.num_entries == int(norm[0])')
            required = ["SuperFatJetAK15PFCand_" + k for k in PF]
            names = required + IDS + ["SuperFatJetAK15PFCand_jetIdx", "SuperFatJetAK15_pt", "HLT_Mu50"]
            entries = a["inputEntry"].astype("int64")
            if not np.all(entries < tree.num_entries):
                raise ValueError('Source entry beyond parent Nano')
            schema = {k: v.dtype for k, v in a.items()}
            schema.update(source_id="string", event_key="string", is_training_region="bool")
            for k in PF:
                name = "Part_" + {"energy": "E", "px": "PX", "py": "PY", "pz": "PZ"}.get(k, k)
                schema[name] = "var * " + ("int32" if k in ["pdgId", "charge", "fromPV"] else "float32")
            schema["nPart"] = "int32"
            total_pf = 0
            with uproot.create(args.output) as out:
                out.mktree("Events", schema)
                for start in sorted(set((entries // 500 * 500).tolist())):
                    ix = np.flatnonzero((entries >= start) & (entries < start + 500))
                    raw = tree.arrays(names, entry_start=start, entry_stop=start+500, library="ak")
                    batch = {k: v[ix] for k, v in a.items()}
                    parts = {k: [] for k in PF}
                    keys = []
                    for j in ix:
                        i = int(entries[j] - start)
                        if not (all(int(raw[k][i]) == int(a[k][j]) for k in IDS)): raise ValueError('Input invariant failed: all(int(raw[k][i]) == int(a[k][j]) for k in IDS)')
                        if not (bool(raw["HLT_Mu50"][i])): raise ValueError('Input invariant failed: bool(raw["HLT_Mu50"][i])')
                        jet = int(a["selectedJetIdx"][j])
                        if not (np.isclose(float(raw["SuperFatJetAK15_pt"][i][jet]), a["jet_pt"][j])): raise ValueError('Input invariant failed: np.isclose(float(raw["SuperFatJetAK15_pt"][i][jet]), a["jet_pt"][j])')
                        keep = raw["SuperFatJetAK15PFCand_jetIdx"][i] == jet
                        if not (int(ak.sum(keep)) > 0): raise ValueError('Input invariant failed: int(ak.sum(keep)) > 0')
                        for k in PF:
                            values = ak.to_numpy(raw["SuperFatJetAK15PFCand_"+k][i][keep])
                            if not (np.isfinite(values).all()): raise ValueError('Input invariant failed: np.isfinite(values).all()')
                            parts[k].append(values)
                        identity = "%d:%d:%d" % tuple(int(a[k][j]) for k in IDS)
                        # Data copies share the same key regardless of file identity.
                        if not spec["is_data"]:
                            identity = spec["sample"] + ":" + spec["source"] + ":" + identity
                        keys.append(hashlib.sha256(identity.encode()).hexdigest())
                    for k in PF:
                        name = "Part_" + {"energy": "E", "px": "PX", "py": "PY", "pz": "PZ"}.get(k, k)
                        dtype = np.int32 if k in ["pdgId", "charge", "fromPV"] else np.float32
                        batch[name] = ak.values_astype(ak.Array(parts[k]), dtype)
                    batch["nPart"] = np.array([len(v) for v in parts["pt"]], dtype=np.int32)
                    total_pf += int(batch["nPart"].sum())
                    batch["source_id"] = ak.Array([spec["source"]] * len(ix))
                    batch["event_key"] = ak.Array(keys)
                    batch["is_training_region"] = batch["abcdRegion"] == 1
                    out["Events"].extend(batch)
                out["normalization"] = compact["normalization"]
                meta_schema = {k: "string" if v.dtype.kind in "OUS" else v.dtype for k,v in meta.items()}
                out.mktree("AnalysisMetadata", meta_schema)
                out["AnalysisMetadata"].extend({k:ak.Array(v.tolist()) if meta_schema[k]=="string" else v for k,v in meta.items()})
                out["ProductionSpec"] = json.dumps(spec, sort_keys=True)
                lumis = compact["LuminosityBlocks"].arrays(library="np")
                out.mktree("LuminosityBlocks", {k:v.dtype for k,v in lumis.items()})
                if len(next(iter(lumis.values()))): out["LuminosityBlocks"].extend(lumis)
    with uproot.open(args.output) as check:
        if not (check["Events"].num_entries == len(entries)): raise ValueError('Input invariant failed: check["Events"].num_entries == len(entries)')
        if not (np.array_equal(check["Events"]["inputEntry"].array(library="np"), entries)): raise ValueError('Input invariant failed: np.array_equal(check["Events"]["inputEntry"].array(library="np"), entries)')
        if not (np.array_equal(check["Events"]["abcdRegion"].array(library="np"), a["abcdRegion"])): raise ValueError('Input invariant failed: np.array_equal(check["Events"]["abcdRegion"].array(library="np"), a["abcdRegion"])')
        if not (np.array_equal(check["normalization"].values(), norm)): raise ValueError('Input invariant failed: np.array_equal(check["normalization"].values(), norm)')
    validate_source(spec, args.nano, args.compact)
    report = dict(key=spec["key"], sample=spec["sample"], status=0,
                  input_events=int(norm[0]), sum_genweights=float(norm[1]),
                  selected=len(entries), n_pf=total_pf, output_bytes=Path(args.output).stat().st_size,
                  regions={str(r): int((a["abcdRegion"] == r).sum()) for r in range(1,5)},
                  source=spec["source"], compact_source=spec.get("compact"),
                  nano_sha256=spec['nano_sha256'], compact_sha256=spec['compact_sha256'],
                  output_sha256=file_hash(args.output),
                  note="All PF constituents; raw genWeight, unit data weight; no final training split assigned")
    with audit.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
