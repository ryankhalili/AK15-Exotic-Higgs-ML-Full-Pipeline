# Stage Contracts

## Selection And ROOT Layout

The provided configuration is an analysis-specific UL2018 muon selection, not a
universal NanoAOD converter. Require a tight muon with pT >55 GeV, |eta| <2.4,
HLT_Mu50, global-leading AK15 pT >200 GeV and |eta| <3, |DeltaPhi(muon,jet)| >1.5,
PF MET >30 GeV, |DeltaPhi(MET,jet)| >1 and AK4 HT >200 GeV. HT includes AK4 jets
with pT >30 GeV, |eta| <3 and jetId >=2. Isolation is Muon_pfRelIso04_all.

Regions are encoded as integers: A=1 (iso <0.15, mT >50), B=2 (iso <0.15,
mT <=50), C=3 (0.15 <=iso <0.5, mT >50), D=4 (0.15 <=iso <0.5, mT <=50).
The PF join verifies this exact contract. A different selection needs a separately
reviewed adapter; changing JSON cuts alone does not change its validation.

Selected ntuples contain a TTree named `Events`, with `run`, `luminosityBlock`,
`event`, `inputEntry`, `source_id`, SHA256 `event_key`, `isData`, `genWeight`,
`hasGenWeight`, `abcdRegion`, jet kinematics and selection scalars. PF vectors
are `Part_pt`, `Part_eta`, `Part_phi`, `Part_mass`, `Part_E`, `Part_PX`, `Part_PY`,
`Part_PZ`, `Part_pdgId`, `Part_charge`, `Part_fromPV`, with scalar `nPart`.
All PF constituents are retained; networks take the highest-pT 256 with masks.

Preserve `AnalysisMetadata` and `LuminosityBlocks` TTrees, `ProductionSpec`
TObjString (`sample`, `source`, `is_data`), and the `normalization` histogram.
Its first two visible bins are preselection input events and signed generator
sum, not selected-event normalization. Other original TH1/TObjString/TTree
objects are supported by the score writer; other object classes fail closed.
Multiple ROOT cycles and already-scored inputs are rejected.

## Training Bundles

`prepare_training.py` writes checksummed NPY x/v/mask/meta shards and a manifest.
`index_full_inputs.py` rejects duplicate source identities and event/content/full-PF
fingerprints, combines normalization per sample, assigns a deterministic content
split, and writes `index.npz` plus `configuration.json`. Its shard paths are
relative to the index directory. Moving the prepared/index directory layout as
a unit preserves those paths. Trainers resolve them at runtime and verify hashes.

Inference expects `BUNDLE/indexed/index.npz` and the full `BUNDLE/results/` set:
`configuration.json`, `history.csv`, `summary.json`, `best.pt`, `test_scores.npz`.
The model source digest, feature contract, architecture, index digest, complete
epoch history, checkpoint/JSON configuration and validation selection are checked.
No campaign-specific mass list, training date, predetermined checkpoint digest,
or private path is part of the portable contract.

The twelve features are delta eta/phi, log pT, log pT fraction, log energy
fraction, charge, fromPV/3, and charged-hadron/neutral-hadron/photon/electron/muon
indicators. Mean/std use TRAIN constituents only. Full four-vectors feed ParT.
The raw generator weight is never a model feature.

## Scored Samples And Statistics

Scored input adds `ParticleNet_score` and/or `ParticleTransformer_score`,
`NN_valid`, `NN_trainingRegion`, `NN_inputSplit` and `NNMetadata`. Split codes:
-2=unverified MC membership, -1=data, 0=train, 1=validation, 2=test. A known event
identity must match its stored capped-input fingerprint. Matching content from a
different source still receives the frozen membership. Encountered test rows
must reproduce saved scores within absolute 1e-5. This is a strict numerical gate,
not a claim of bit-for-bit agreement or of full-source replay coverage.

Histogram manifests describe all sources of each sample. MC coefficient:
`luminosity_pb * xsec_pb * factor / sum_genweights`; multiply signed `genWeight`.
The manifest denominator must equal the sum of input preselection denominators.
ROOT arrays are read in bounded batches; duplicate detection retains identity
keys in memory, so memory for this check grows with the number of distinct events.
Data weights are exactly one. Test-only region-A templates additionally divide
by the frozen `test_fraction`; no correction is applied to control regions.
Changing a normalization policy requires rebuilding all A/B/C/D contributions.

With non-QCD-subtracted control totals B,C,D and D-bin residual d_i, the QCD
prediction is `q_i = B*C*d_i / D^2`. The covariance propagates independent B/C
total variances and D-bin variances through the full Jacobian. Statistical
eigenvectors become explicit Up/Down shapes, only when all shifted bins remain
positive. Invalid residuals/negative template bins are errors; they are never
silently clipped. This does not model transfer nonclosure or missing systematics.

TH1 layout is `pass/{data_obs,signal,QCD,WJets,DYJets,TT,ST,VV}`, with explicit
Sumw2. QCD variations are `pass/QCD_<nuisance>Up/Down`; a TH2 covariance is stored
at `statistical_covariance/QCD`. `qcd_shapes.json` names all nuisances and states
coverage limitations. One signal hypothesis, classifier and category per build.
Score edges include exactly 0 and 1; score 1 belongs to the final visible bin.

The Combine builder preserves actual integral rates, rejects negative bins,
and uses ordinary TH1 inputs through genuine CombineHarvester. Its fixed-QCD
option is diagnostic-only; the limit runner requires explicit QCD shapes.
The workspace response check compares central and +/-1 nuisance values against
each supplied QCD bin before invoking Combine. The nominal mass label 125 is a
Combine bookkeeping label, not an inferred physical signal normalization.
