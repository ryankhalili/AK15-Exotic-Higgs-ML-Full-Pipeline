# AK15 Exotic Higgs ML Full Pipeline

Modular source pipeline for MiniAOD -> leading-AK15 NanoAOD -> selected full-PF
ntuples -> ParticleNet/Particle Transformer -> scored ROOT -> score TH1 templates
-> CombineHarvester cards -> expected limits. Each stage has its own entry point;
start at training or at scored samples without running reconstruction.

Repository: `https://github.com/ryankhalili/AK15-Exotic-Higgs-ML-Full-Pipeline`

```bash
git clone https://github.com/ryankhalili/AK15-Exotic-Higgs-ML-Full-Pipeline.git
cd AK15-Exotic-Higgs-ML-Full-Pipeline
```

This is a source-only staging release, not a bundled dataset, trained-model
release, or validated branching-fraction result. See [license status](LICENSE_STATUS.md)
before redistribution. The examples contain illustrative numbers and nonexistent
input filenames, not analysis inputs or recommended physical normalizations.

## Environments

Use separate shells/environments for these stages. Do not install modern Python
packages inside the legacy CMSSW environment.

| Stage | Runtime | Export validation |
| --- | --- | --- |
| MiniAOD -> Nano | CMSSW_10_6_17, slc7_amd64_gcc700, EL7-compatible Linux | Source exported; build/real input smoke still required |
| Nano -> compact | ROOT/ACLiC in the same CMSSW runtime | Source exported; runtime smoke still required |
| PF ntuples, tensors, scoring, histograms | Python 3.10+, NumPy, Awkward, uproot | Synthetic ROOT and CLI tests |
| Training | Separate PyTorch environment and CUDA GPU | Dataset/sampler tests; no new GPU training run |
| Cards/workspaces/limits | ROOT, CombineHarvester, Combine | ROOT-free contracts tested; no local likelihood fit |

The recorded successful standard-TH1 environment used **Combine v11.1.0**,
`pass autoMCStats 0 1 2`, and per-bin workspace response checks. The public
container digest and CVMFS entry point are in
[combine-runtime.json](environments/combine-runtime.json). This is not a claim
that any arbitrary current ROOT/CH installation is equivalent: the CVMFS alias
is independently mutable. Record the actual tool versions and rerun the
workspace checks for your installation. No cluster account or batch system is required.

From a checkout root, in a modern Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/analysis.txt
# Add these only for model training/inference:
python -m pip install -r requirements/training.txt
mkdir -p inputs work external
```

On Windows use `.venv\Scripts\Activate.ps1`; reconstruction/Combine commands
below are Linux commands. Tested local package versions are recorded in
[tested-local.txt](requirements/tested-local.txt), not a universal CUDA lockfile.

## Enter At MiniAOD

Use an existing compatible CMSSW area. From this repository root, copy only the
two source packages into its `src` directory (first ensure those destinations
do not already exist), then compile inside that area:

```bash
cp -R cmssw/MyAnalysis "$CMSSW_BASE/src/"
cp -R cmssw/JMEAnalysis "$CMSSW_BASE/src/"
(cd "$CMSSW_BASE/src" && scram b -j4)
cmsRun cmssw/nano_mc_cfg.py inputFiles=file:inputs/miniaod.root \
  outputFile=work/sample.root maxEvents=100 ak15LeadingOnly=True
```

The configuration writes `work/sample_Nano.root`. For data use
`cmssw/nano_data_cfg.py`; it has the data GlobalTag and sequences. These configs
are UL2018-specific, use CHS AK15 jets with the AK8PFchs correction payload, and
retain all constituents of the global leading AK15 jet. This is not an AK15
calibration validation. Run a bounded input smoke before large processing.

## Enter At Nano Or Ntuples

Still in the ROOT/CMSSW environment:

```bash
bash ntuples/run_physics_analysis_tree.sh work/sample_Nano.root work/compact.root \
  --config examples/selection.json --sample signal_benchmark --is-data 0
```

For data, `--is-data 1 --lumi-mask inputs/certification.json` is mandatory. Supply
your own approved certification mask; none is distributed. The reducer writes
region A to `Events`, all ABCD candidates to `ABCDEvents`, and preselection
normalization/luminosity metadata. The explicit selection is documented in
[contracts](docs/CONTRACTS.md).

In the modern Python environment, join the compact candidates back to their
parent Nano file and retain every selected-jet PF constituent:

```bash
python scripts/make_ntuple_spec.py --nano work/sample_Nano.root --compact work/compact.root \
  --sample signal_benchmark --is-data 0 --output work/source.json
python scripts/make_selected_pf_ntuple.py work/source.json \
  work/sample_Nano.root work/compact.root inputs/signal.root
```

The spec generator checks the compact sample label and binds both inputs by
SHA256. `source` is the Nano content hash, not a changing local filename.
Check that the sample label itself matches your dataset manifest. This join requires one complete
parent Nano per compact file and matching entry counts/identities. Use the same
bounded Nano file for both commands. A successful write produces
`inputs/signal.audit.json`; existing output/audit paths are refused.

## Enter At Training

Supply selected full-PF ROOT files matching [the schema](docs/CONTRACTS.md), and
an explicit manifest like `examples/training.json`. Labels are 0 for background
and 1 for signal. Each signal hypothesis has its own sample name; nominal and
extension sources of one process share its name and normalization convention.
Only region A enters supervised learning. Data are rejected.

Fetch the pinned author implementations and their license notices once:

```bash
python tools/fetch_models.py external/models
```

The fetcher checks upstream SHA256 values and refuses an existing destination.
It pins Weaver commit `34e2ee20ca7d1b94446f1bf18b361c5865c4b6d4`; the only
adaptation replaces its logger import with Python logging. These are architecture
sources, not pretrained weights. See [third-party attribution](THIRD_PARTY.md).
The adapter supplies constructor arguments in `scripts/full_models.py` and checks
source digests when reloading trained bundles. Run the optional model test before training.

```bash
python scripts/prepare_training.py examples/training.json work/prepared
python scripts/index_full_inputs.py work/prepared work/pn/indexed
python scripts/train_full_models.py work/pn/indexed external/models work/pn/results \
  --model ParticleNet --profile full --epochs 12 --batch-size 16 --accumulate 8

mkdir -p work/part
cp -R work/pn/indexed work/part/indexed
python scripts/train_transformer.py work/part/indexed external/models work/part/results \
  --epochs 6 --learning-rate 0.001 --warmup-epochs 1
```

The two sibling `indexed` directories use the same relative shard locations.
Keep `work/prepared` alongside the training bundles. The default split is
content-hash based 70/15/15; exact event, full-PF and capped-input duplicates
are rejected. All standardization and class/mass balancing use TRAIN only.
Each sample needs at least ten rows per partition. Optimization weights are
positive; signed physics weights remain separate.

ParticleNet uses uniformly shuffled, weighted cross entropy. Full ParT uses
class-balanced weighted sampling, unit cross entropy on sampled rows, and
optimizer-update warmup. ParT's validation learning gates fail closed; they
are not a promise that arbitrary samples will train successfully. Validation
loss chooses the checkpoint; test scores are evaluated after selection.

## Score Selected Ntuples

Only load trusted model Python files and locally produced/trusted checkpoints;
checkpoint deserialization and model imports can execute code.

```bash
python scripts/score_new_models.py inputs/signal.root inputs/signal_scored.root \
  --particle-net work/pn --particle-transformer work/part --models external/models \
  --device cuda --batch-size 16
```

Either model option may be omitted. The writer preserves original ROOT fields,
objects, rows, types and signed weights, then adds only the requested score
branches and membership metadata. A known identity with changed capped features
is rejected. Every encountered test member is freshly rescored and compared to
its saved score with absolute tolerance `1e-5`, relative tolerance zero. A failure
aborts; no cached-score substitution or automatic tolerance change is available.
Numerical portability across devices is not guaranteed. Zero test members in a
file means zero replay coverage, not a validated replay. Counts/deltas are saved
in `NNMetadata` and the success audit. Partial failed outputs are not success.

## Enter At Scored Samples

No model source, checkpoint, or training run is needed for this entry point.
Populate `examples/scored.json` with your own local files, luminosity, cross
sections, process factors, processed signed generator denominators, score
branch, and reviewed bin edges. Paths in manifests are relative to the manifest.
Do not use the illustrative numbers as real normalization.

```bash
python scripts/score_histograms.py examples/scored.json work/templates
python scripts/plot_outputs.py stack --templates work/templates/histograms.root \
  --lumi-pb 1000 --label ParticleNet --signal-label 'Signal benchmark' \
  --signal-scale 1000 --log --output work/score_plot
```

This intentionally narrow adapter supports one signal hypothesis and the WJets,
DYJets, TT, ST and VV process groups in one pass category. It recomputes ABCD
subtraction for every invocation. It does not add QCD MC to the stack. Data
overlaps are deduplicated only when region and score agree exactly; conflicting
duplicates fail. Training-member region-A MC requires explicit test selection
and the frozen test fraction. Signed MC and `sumw2` are retained; negative fit
bins or invalid control residuals require reviewed binning, never clipping.
The plotting luminosity must match your template configuration; 1000 above is
only the example value. Scaling signal for display never changes the fit input.

## Cards And Limits

In the separate Combine environment, either build only the card:

```bash
python3 scripts/build_card.py --templates work/templates/histograms.root \
  --qcd-shapes work/templates/qcd_shapes.json --output work/card
```

Or build a new card/workspace, verify QCD responses, and run the expected fit:

```bash
python3 scripts/expected_limits.py --templates work/templates/histograms.root \
  --qcd-shapes work/templates/qcd_shapes.json --output work/limit --rmax 100
```

In the plotting environment, list the successful summary files and labels in
an explicit manifest, such as `examples/limit_grid.json`, then draw the bands:

```bash
python scripts/plot_outputs.py limits --manifest examples/limit_grid.json \
  --label ParticleNet --output work/limit_plot
```

The runner always uses `--run blind`, mode 2, and rejects missing/unordered
expected quantiles, observed entries, and limits near the supplied search bound.
`rmax` must be chosen for the signal normalization; it is not auto-tuned.
QCD TH1 errors are zeroed in fit copies to avoid double-counting the explicit
covariance-derived shape nuisances. MC statistics use `autoMCStats` separately.

**Physics scope:** these are conditional expected signal-strength limits, not
branching fractions. Statistical ABCD covariance does not validate the D-to-A
score-shape transfer. Efficiency, pileup, detector/theory uncertainties, signal
contamination and unmodeled correlations need independent treatment. No physical
signal cross section or cascade branching convention is inferred. PN and ParT
score the same events and must not be combined as independent categories.

## Tests And Release Scope

```bash
python -m pip install -r requirements/test.txt
python -B -m pytest -q -p no:cacheprovider
# Optional CPU roundtrip against supplied architecture source:
AK15_MODEL_SOURCE_DIR=external/models python -B -m pytest tests/test_inference.py -q
python tools/audit_sources.py --inventory
# Before a commit, check exactly the staged source set:
python tools/audit_sources.py --staged
```

Tests generate only temporary synthetic inputs. Local coverage includes tensor
preparation/indexing, train-only preprocessing, balanced sampling, single/dual
score ROOT preservation, signed histograms, overlap/membership rejection, ABCD
covariance, expected-only command/quantile checks, and CLI help. The optional
architecture roundtrip does not establish trained model quality or GPU replay.

Not included: data discovery/download, credential handling, batch submission,
automatic recovery, pretrained weights, production input lists/results, plot
packs, or branching-fraction conversion. Full CMSSW execution, GPU training and
a fresh Combine fit remain integration checks. [FILES.txt](FILES.txt) is the
source inventory; ignored local environments/work products are not release files.
