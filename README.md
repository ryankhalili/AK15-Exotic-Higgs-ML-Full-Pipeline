# AK15 Exotic Higgs ML Full Pipeline

Reconstruct large-radius jets, train ParticleNet and Particle Transformer, add
their scores to event files, and use the score distributions to calculate
expected signal-strength limits.

The steps are:

1. Add AK15 jets and their constituent particles to MiniAOD events.
2. Select events and write smaller ROOT files for the neural networks.
3. Divide simulated events into training, validation and test sets, then train.
4. Apply the trained networks to simulation and collision data.
5. Fill weighted score histograms and estimate the QCD background.
6. Build a statistical model with CombineHarvester and run Combine.

Each stage has its own entry point. Existing selected ntuples can enter at
training; existing scored ntuples can enter at histogram creation.

Repository: `https://github.com/ryankhalili/AK15-Exotic-Higgs-ML-Full-Pipeline`

```bash
set -eo pipefail
git clone https://github.com/ryankhalili/AK15-Exotic-Higgs-ML-Full-Pipeline.git
cd AK15-Exotic-Higgs-ML-Full-Pipeline
export AK15_REPO="$PWD"
mkdir -p inputs work external
```

The repository contains reconstruction, training, inference and statistical
analysis code. Datasets and trained weights are supplied separately. See
[license status](LICENSE_STATUS.md) before redistribution.

| Available input | Start here |
| --- | --- |
| MiniAOD | [Reconstruction](#enter-at-miniaod) |
| NanoAOD with this project's AK15 tables | [Ntuple preparation](#enter-at-nano-or-ntuples) |
| Selected PF ntuples | [Training](#enter-at-training) |
| Trained models and ntuples | [Inference](#score-selected-ntuples) |
| Scored ntuples | [Histograms](#enter-at-scored-samples) |
| Score histograms | [Datacards and limits](#cards-and-limits) |

Commands below use Bash on Linux and run from the repository root unless a
step says otherwise. In a new terminal, return to the checkout and set
`export AK15_REPO="$PWD"` again. Use a user-owned path without spaces for CMSSW.
Run compilation, event processing and GPU training within your site's resource
allocation, not as large interactive workloads on a shared login node.

A **manifest** is a JSON file listing the inputs and the information needed to
interpret them, such as sample names and cross sections. The files in `examples/`
show the format. Their filenames and numbers are illustrative: they are not a
downloadable dataset or a physics configuration ready for use.

Keep inputs in `inputs/`, generated results in `work/`, and downloaded model
source in `external/`. These directories are excluded from Git. Most writers
refuse an existing output. Use a new output path for a repeated calculation.

## Environments

Use separate shells/environments for these stages. Do not install modern Python
packages inside the legacy CMSSW environment.

| Stage | Runtime | Validation in this checkout |
| --- | --- | --- |
| MiniAOD -> Nano | CMSSW_10_6_17, slc7_amd64_gcc700, EL7-compatible Linux | Fresh build and real-input test required |
| Nano -> compact | ROOT/ACLiC in the same CMSSW runtime | Fresh ROOT runtime test required |
| PF ntuples, tensors, scoring, histograms | Python 3.10+, NumPy, Awkward, uproot | Synthetic ROOT and CLI tests |
| Training | Separate PyTorch environment and CUDA GPU | Dataset, sampler and CPU checkpoint tests; fresh GPU training required |
| Cards/workspaces/limits | ROOT, CombineHarvester, Combine | ROOT-free contracts tested; no local likelihood fit |

The standard-TH1 workflow targets **Combine v11.1.0**,
`pass autoMCStats 0 1 2`, and per-bin workspace response checks. The public
container digest and CVMFS entry point are in
[combine-runtime.json](environments/combine-runtime.json). The CVMFS alias can
change independently of that image. Record tool versions and run the workspace
checks in your installation. Batch submission is managed separately.

### Python For Analysis And Models

Open a fresh terminal outside CMSSW. Python 3.10 or newer is required; Python
3.12 is used in CI. A virtual environment keeps this project's packages separate
from system software.

```bash
set -eo pipefail
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/analysis.txt
python -c "import numpy, awkward, uproot, matplotlib; print('Analysis imports OK')"
```

For training or inference, install PyTorch and the additional dependencies:

```bash
python -m pip install -r requirements/training.txt
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

On Windows use `.venv\Scripts\Activate.ps1`; reconstruction/Combine commands
below are Linux commands. Tested local package versions are recorded in
[tested-local.txt](requirements/tested-local.txt), not a universal CUDA lockfile.
The trainers require a CUDA GPU. If the check reports `False`, obtain a GPU
allocation and use the [PyTorch installation instructions](https://pytorch.org/get-started/locally/)
for that machine's driver. CPU-only PyTorch can still run the unit tests.

In later Python terminals, activate the existing environment rather than
creating it again:

```bash
set -eo pipefail
source .venv/bin/activate
```

## Enter At MiniAOD

### 1. Create The Reconstruction Environment

CMSSW is the CMS event-processing software. This reconstruction code targets
`CMSSW_10_6_17` on x86-64 with an EL7-compatible operating system. It is separate
from the Python environment above and the Combine environment below.

Open a fresh Bash terminal at the checkout on a machine with CMS CVMFS and
Apptainer available. Enter the CMS EL7 container:

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
/cvmfs/cms.cern.ch/common/cmssw-el7
```

Inside that container, verify the checkout is visible and create a new CMSSW
area. `set -e` stops the block if a check or build fails.

```bash
set -eo pipefail
test -f cmssw/nano_mc_cfg.py
export AK15_REPO="$PWD"
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc700
mkdir -p "$AK15_REPO/work/cmssw"
cd "$AK15_REPO/work/cmssw"
test ! -e CMSSW_10_6_17
scram project CMSSW CMSSW_10_6_17
cd CMSSW_10_6_17/src
eval "$(scram runtime -sh)"
test ! -e MyAnalysis
test ! -e JMEAnalysis
cp -R "$AK15_REPO/cmssw/MyAnalysis" .
cp -R "$AK15_REPO/cmssw/JMEAnalysis" .
scram b -j4
cd "$AK15_REPO"
cmsRun --help
root-config --version
```

The container must expose both the checkout and input storage. Follow the
[CMS container guide](https://cms-sw.github.io/singularity.html) and your site's
mount policy if a directory is not visible; do not change shared mounts.
No system package installation or administrator privileges are needed here.

For subsequent sessions, enter the same container, then activate the existing
area instead of repeating the creation and copy commands:

```bash
export AK15_REPO="$PWD"
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc700
cd "$AK15_REPO/work/cmssw/CMSSW_10_6_17/src"
eval "$(scram runtime -sh)"
cd "$AK15_REPO"
```

### 2. Run A Small Reconstruction Check

Supply a local UL2018 MC MiniAOD as `inputs/miniaod_mc.root`. The bounded test
below processes 100 events. It tests reconstruction startup, not training
statistics or the full sample.

```bash
test -f inputs/miniaod_mc.root
mkdir -p work/smoke work/logs
test ! -e work/smoke/mc_Nano.root
test ! -e work/logs/nano_smoke.log
cmsRun cmssw/nano_mc_cfg.py inputFiles=file:inputs/miniaod_mc.root \
  outputFile=work/smoke/mc.root maxEvents=100 ak15LeadingOnly=True \
  2>&1 | tee work/logs/nano_smoke.log
root -l -b -q -e 'TFile f("work/smoke/mc_Nano.root"); f.ls();'
```

The configuration appends `_Nano` to the requested output filename. Check the
exit status, log and `Events` tree before continuing. These configs use CHS AK15
jets with the AK8PFchs correction payload and retain all constituents of the
global leading AK15 jet. This choice does not establish a validated AK15
calibration.

### 3. Process Full Input Files

After the small check passes, process one complete source file with no event
limit. Use a different output for every source file:

```bash
mkdir -p work/nano
test ! -e work/nano/signal_001_Nano.root
cmsRun cmssw/nano_mc_cfg.py inputFiles=file:inputs/miniaod_mc.root \
  outputFile=work/nano/signal_001.root maxEvents=-1 ak15LeadingOnly=True
```

For collision data, supply `inputs/miniaod_data.root` and use the data config:

```bash
test -f inputs/miniaod_data.root
test ! -e work/nano/data_001_Nano.root
cmsRun cmssw/nano_data_cfg.py inputFiles=file:inputs/miniaod_data.root \
  outputFile=work/nano/data_001.root maxEvents=-1 ak15LeadingOnly=True
```

Use the MC config for both signal and simulated backgrounds. The data config
selects the data GlobalTag and sequences. Repeat for the intended source list;
keep a one-to-one Nano/compact/PF-ntuple relationship in the steps below.
Remote input authentication and scheduler submission are site-specific and are
not performed by this repository. The commands here do not submit Condor jobs.

## Enter At Nano Or Ntuples

### 4. Apply The Event Selection

This step requires the custom AK15 NanoAOD produced above. Ordinary CMS NanoAOD
does not contain these jet/constituent tables; use the MiniAOD reconstruction
step if they are absent.

Still in the ROOT/CMSSW environment, reduce the full Nano file:

```bash
mkdir -p work/compact
bash ntuples/run_physics_analysis_tree.sh \
  work/nano/signal_001_Nano.root work/compact/signal_001.root \
  --config examples/selection.json --sample signal_benchmark --is-data 0 --max-events -1
```

For data, supply an approved certification JSON. It lists the run and luminosity
sections accepted for physics analysis; it does not itself provide integrated
luminosity.

```bash
test -f inputs/certification.json
bash ntuples/run_physics_analysis_tree.sh \
  work/nano/data_001_Nano.root work/compact/data_001.root \
  --config examples/selection.json --sample data --is-data 1 --max-events -1 \
  --lumi-mask inputs/certification.json
root -l -b -q -e 'TFile f("work/compact/data_001.root"); f.ls();'
```

The reducer writes region A to `Events`, all ABCD candidates to `ABCDEvents`,
and preselection event/weight totals. Data also retain luminosity information.
These regions are event selections, not data-taking periods. See the exact
[selection and branch definitions](docs/CONTRACTS.md#selection-and-root-layout).
Do not truncate this reduction with `--max-events` when making a full-source
PF ntuple; the next step checks the parent event count.

### 5. Attach The Selected Jet's Particles

Open the modern Python terminal and activate `.venv`. The compact file identifies
the selected events and jets. This step looks them up in the corresponding Nano
file and writes all their PF constituent particles:

```bash
source .venv/bin/activate
mkdir -p work/specs
python scripts/make_ntuple_spec.py \
  --nano work/nano/signal_001_Nano.root --compact work/compact/signal_001.root \
  --sample signal_benchmark --is-data 0 --output work/specs/signal_001.json
python scripts/make_selected_pf_ntuple.py work/specs/signal_001.json \
  work/nano/signal_001_Nano.root work/compact/signal_001.root inputs/signal.root

python scripts/make_ntuple_spec.py \
  --nano work/nano/data_001_Nano.root --compact work/compact/data_001.root \
  --sample data --is-data 1 --output work/specs/data_001.json
python scripts/make_selected_pf_ntuple.py work/specs/data_001.json \
  work/nano/data_001_Nano.root work/compact/data_001.root inputs/data.root
python -m json.tool inputs/signal.audit.json
python -m json.tool inputs/data.audit.json
```

The spec generator checks the compact sample label and binds both inputs by
SHA256. `source` is the Nano content hash, not a changing local filename.
Check that the sample label itself matches your dataset manifest. This join requires one complete
parent Nano per compact file and matching entry counts/identities. A successful write produces
`inputs/signal.audit.json`; existing output/audit paths are refused.
Unlike the compact file's `Events`, the selected-PF file's `Events` contains
all four ABCD regions, distinguished by `abcdRegion`. Do not discard B/C/D:
histogram creation needs them for the QCD estimate.

Repeat these commands with distinct filenames for each MC background and signal
source. Files from the same process and extension sample share a sample label;
different signal masses and different W HT bins have different labels. More
than one file per sample is supported in the training and histogram manifests.

## Enter At Training

### 6. Describe The Training Samples

Supply selected full-PF ROOT files matching [the schema](docs/CONTRACTS.md), and
copy the example input list into the ignored input directory:

```bash
test ! -e inputs/training.json
cp examples/training.json inputs/training.json
```

Edit `inputs/training.json` with the actual source files and normalization values.
Paths are relative to that manifest, not to the terminal's current directory.

| Field | Meaning |
| --- | --- |
| `key` | Unique label for this input file |
| `sample` | Must match `ProductionSpec.sample` inside the ROOT file |
| `file` | Selected-PF ROOT file, not a MiniAOD or compact file |
| `label` | 0 for simulated background; 1 for simulated signal |
| `xsec_pb` | Process cross section in picobarns |
| `factor` | Explicit additional normalization, such as a reviewed k factor |

Each signal hypothesis has its own sample name; nominal and
extension sources of one process share its name and normalization convention.
An extension is an additional production of the same simulated process.
Only region A enters supervised learning. Data are rejected.

### 7. Download The Model Code

Download the ParticleNet and Particle Transformer implementations from Weaver:

```bash
python tools/fetch_models.py external/models
```

The download uses a fixed public source version and checks file contents. This
keeps model definitions consistent between training and inference. It downloads
architecture code and license notices, not pretrained weights. Source versions
and the small logging adaptation are documented in [third-party attribution](THIRD_PARTY.md).
An existing download directory is not overwritten.

### 8. Prepare And Split The Events

The first command converts particles into arrays. The second assigns events to
partitions and calculates preprocessing constants from the training set:

```bash
python -m json.tool inputs/training.json
python scripts/prepare_training.py inputs/training.json work/prepared
python scripts/index_full_inputs.py work/prepared work/pn/indexed
python -m json.tool work/pn/indexed/configuration.json
```

The default split is reproducible and content-hash based: 70% training, 15%
validation, 15% testing. It is not a contiguous file segment or every tenth
event. Exact event, full-PF and capped-input duplicates are rejected. Each
sample needs at least ten rows per partition. A 100-event reconstruction test
will generally not supply enough selected events.

The ntuple keeps every constituent. Network arrays use the highest-pT 256,
with masked padding for shorter jets. Feature means, standard deviations,
class balancing and signal-mass balancing are calculated using TRAIN only.
The raw generator weight is not a neural-network input feature.

### 9. Train Both Models

In a GPU allocation, first check CUDA and train ParticleNet:

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU required'; print(torch.cuda.get_device_name(0))"
python scripts/train_full_models.py work/pn/indexed external/models work/pn/results \
  --model ParticleNet --profile full --epochs 12 --batch-size 16 --accumulate 8
python -m json.tool work/pn/results/summary.json
```

Use the same partition for Particle Transformer, with a separate result directory:

```bash
mkdir -p work/part
test ! -e work/part/indexed
cp -R work/pn/indexed work/part/indexed
python scripts/train_transformer.py work/part/indexed external/models work/part/results \
  --epochs 6 --batch-size 16 --learning-rate 0.001 --warmup-epochs 1
python -m json.tool work/part/results/summary.json
```

The two sibling `indexed` directories use the same relative shard locations.
Keep `work/prepared` alongside the training bundles. Batch size 16 means 16
events per GPU forward/backward pass. Both commands accumulate eight such
batches before an optimizer update, normally 128 events per update. The full
model definitions are in `scripts/full_models.py`. Optimization weights are
positive; signed physics weights remain separate.

ParticleNet shuffles the training rows and weights each event's contribution
to the classification loss. Full ParT samples balanced signal/background
batches using those training weights, without multiplying the loss by the same
weights a second time. It starts at a lower learning rate and increases it
during warmup. ParT stops with an error if its validation checks do not show
learning. Validation loss chooses the checkpoint; test scores are evaluated
after that choice.
There is no separate required testing command: each trainer evaluates its
chosen checkpoint on the reserved test set and writes `test_scores.npz`.

Each successful model directory contains `best.pt` (trained parameters),
`configuration.json` (architecture and settings), `history.csv` (epoch results),
`summary.json` (test metrics), and `test_scores.npz` (test predictions). Keep
the entire `indexed/` and `results/` bundle; inference needs both.

## Score Selected Ntuples

### 10. Apply The Models To Event Files

Inference means evaluating a trained model, with no further weight updates.
Only load trusted model Python files and locally produced/trusted checkpoints;
checkpoint deserialization and model imports can execute code.

```bash
python scripts/score_new_models.py inputs/signal.root inputs/signal_scored.root \
  --particle-net work/pn --particle-transformer work/part --models external/models \
  --device cuda --batch-size 16
python -m json.tool inputs/signal_scored.audit.json
```

Repeat for collision data and the simulated backgrounds. The following list
matches the example histogram manifest; substitute your actual filenames:

```bash
set -eo pipefail
for sample in data background dy tt st vv; do
  python scripts/score_new_models.py "inputs/${sample}.root" "inputs/${sample}_scored.root" \
    --particle-net work/pn --particle-transformer work/part --models external/models \
    --device cuda --batch-size 16
done
```

Either model option may be omitted; CPU inference uses `--device cpu`. The writer preserves original ROOT fields,
objects, rows, types and signed weights, then adds only the requested score
branches and membership metadata. A known identity with changed capped features
is rejected. Every encountered test member is freshly rescored and compared to
its saved score with absolute tolerance `1e-5`, relative tolerance zero. A failure
aborts; no cached-score substitution or automatic tolerance change is available.
Numerical portability across devices is not guaranteed. Zero test members in a
file means zero replay coverage, not a validated replay. Counts/deltas are saved
in `NNMetadata` and the success audit. Partial failed outputs are not success.

Inspect the event count, fields and first five predictions without loading ROOT:

```bash
python - <<'PY'
import uproot
with uproot.open('inputs/signal_scored.root') as source:
    tree = source['Events']
    print('Rows:', tree.num_entries)
    print('Objects:', source.keys())
    print(tree.arrays(['event', 'abcdRegion', 'ParticleNet_score',
                       'ParticleTransformer_score', 'NN_inputSplit'],
                      entry_stop=5, library='np'))
PY
```

The scored file is a new copy of the input with additional fields. The original
ntuple is not edited. The score is a classifier output, not a calibrated
probability that a collision contains signal.

## Enter At Scored Samples

### 11. Set The Sample Normalizations

No model source, checkpoint, or training run is needed for this entry point.
Create a histogram manifest and edit its values before running the next step:

```bash
test ! -e inputs/scored.json
cp examples/scored.json inputs/scored.json
```

| Field | What to supply |
| --- | --- |
| `luminosity_pb` | Certified integrated luminosity represented by the collision data, in inverse picobarns |
| `score_branch` | `ParticleNet_score` or `ParticleTransformer_score` |
| `edges` | Common score-bin boundaries, starting at 0 and ending at 1 |
| `samples[].name` | Sample label stored inside each input ROOT file |
| `process` | `data`, `signal`, `WJets`, `DYJets`, `TT`, `ST`, or `VV` |
| `files` | All scored input files for this sample, relative to the manifest |
| `xsec_pb`, `factor` | MC cross section and additional normalization factor |
| `sum_genweights` | Signed preselection generator-weight sum for exactly these processed files |
| `region_a_selection` | `test` for MC used in training; `all` for data or independent MC |
| `test_fraction` | Frozen test fraction, required only for `test` samples |

Give separate entries to processes with different cross sections, even when
they share a plot color. For example, W HT samples have separate entries with
`process: WJets`; ttbar decay channels have separate entries with `process: TT`.
Combine a sample's nominal and extension files under one normalization entry.
Use one signal mass hypothesis per histogram build. No QCD-MC entry is needed:
the fitted QCD prediction comes from collision-data control regions.

The MC weight for each event is:

```text
genWeight * luminosity_pb * xsec_pb * factor / sum_genweights
```

Data contribute one count per event. For training-member MC, only test events
enter region A and their weights additionally divide by `test_fraction`.
B/C/D use all selected events without this division. The denominator is the
preselection signed generator-weight sum, not the number of selected events.
The histogram builder checks it against the input files.

After filling the file lists, print the stored totals to check the denominator:

```bash
python - <<'PY'
import json
from pathlib import Path
import uproot
path = Path('inputs/scored.json')
for sample in json.loads(path.read_text())['samples']:
    if sample['process'] == 'data':
        continue
    total = 0.
    for name in sample['files']:
        with uproot.open(path.parent / name) as source:
            total += float(source['normalization'].values()[1])
    print(sample['name'], 'processed sumGenWeights:', total)
PY
```

The repository does not calculate certified luminosity or infer cross sections.
Supply independently established values. A trigger requirement selects events;
it is not a trigger-efficiency scale factor. Pileup and muon/trigger efficiency
corrections are not applied automatically by this code.

### 12. Create Histograms And The Stack Plot

```bash
python -m json.tool inputs/scored.json
python scripts/score_histograms.py inputs/scored.json work/templates
python -m json.tool work/templates/histogram_receipt.json
LUMI_PB=$(python -c "import json; print(json.load(open('inputs/scored.json'))['luminosity_pb'])")
python scripts/plot_outputs.py stack --templates work/templates/histograms.root \
  --lumi-pb "$LUMI_PB" --label ParticleNet --signal-label 'Signal benchmark' \
  --signal-scale 1000 --log --output work/score_plot
```

Use `--label ParticleTransformer` when that is the manifest's score branch.
The plot contains collision-data points, stacked backgrounds and a separate
signal curve. `--signal-scale 1000` enlarges only the displayed signal; it never
changes the histogram passed to the fit.

The builder produces three files:

| Output | Contents |
| --- | --- |
| `histograms.root` | Binned score distributions for data, signal and backgrounds, with uncertainty variations |
| `qcd_shapes.json` | Names and conventions for the QCD uncertainty variations |
| `histogram_receipt.json` | Input checksums, normalization information and weighted A/B/C/D yields |

Each ROOT histogram is a `TH1`, a one-dimensional set of bins containing event
yields and uncertainties. The directory `pass` contains the region-A templates.
It is a category name, not an additional neural-network score threshold.

For QCD, first subtract weighted non-QCD simulation from collision data in
B/C/D. Denote the remaining totals by B, C and D. The estimate in A is `B*C/D`.
Its score distribution uses the D-region residual fractions: if D-bin i contains
`d_i`, the predicted A-bin yield is `B*C*d_i/D^2`. This assumes a transferable
control-region shape and the relevant ABCD independence; neither is guaranteed
by running the software. See [the statistical contract](docs/CONTRACTS.md#scored-samples-and-statistics).

The builder propagates statistical fluctuations in the shared B/C/D inputs
into correlated QCD-bin uncertainties and writes their Up/Down histograms.
These are analysis-specific calculations in `scripts/abcd_statistics.py`, not
a QCD prediction made by Combine itself.

The histogram builder supports one signal hypothesis and the WJets,
DYJets, TT, ST and VV process groups in one pass category. It recomputes ABCD
subtraction for every invocation. It does not add QCD MC to the stack. Data
overlaps are deduplicated only when region and score agree exactly; conflicting
duplicates fail. Training-member region-A MC requires explicit test selection
and the frozen test fraction. Signed MC and `sumw2` are retained; negative fit
bins or invalid control residuals require reviewed binning, never clipping.
Repeat the build into a new directory for each signal hypothesis and each
classifier. A changed normalization policy also needs a new build, including
the B/C/D subtraction. Do not rescale just the final stack or QCD histogram.

## Cards And Limits

### 13. Set Up Combine And CombineHarvester

CombineHarvester creates the datacard, which lists the processes, yields and
uncertainties. Combine converts that card into a ROOT workspace and calculates
limits. A **workspace** here is a saved statistical model, not a Python virtual
environment or a directory containing source code.

Use a fresh host terminal outside both the reconstruction CMSSW area and the
Python virtual environment. On a Linux site with CVMFS and Apptainer, the
following function runs commands in the public container selected in
`environments/combine-runtime.json`:

```bash
set -eo pipefail
export AK15_REPO="$PWD"
COMBINE_IMAGE=$(python3 -c "import json; print(json.load(open('environments/combine-runtime.json'))['cvmfs_container_image'])")
COMBINE_ENTRYPOINT=$(python3 -c "import json; print(json.load(open('environments/combine-runtime.json'))['cvmfs_runtime_entrypoint'])")
test -d "$COMBINE_IMAGE"
test -f "$COMBINE_ENTRYPOINT"
command -v apptainer
ak15_combine() {
  apptainer exec --cleanenv --no-home --bind /cvmfs --bind "$AK15_REPO" \
    --pwd "$AK15_REPO" "$COMBINE_IMAGE" "$COMBINE_ENTRYPOINT" "$@"
}
ak15_combine python3 -c "import ROOT; import CombineHarvester.CombineTools.ch as ch; print('ROOT:', ROOT.gROOT.GetVersion()); print('CombineHarvester import OK')"
ak15_combine combine --help
ak15_combine text2workspace.py --help
```

Confirm the Combine version is v11.1.0. The public CVMFS entry point can change
independently of the selected image; a successful import is not a full fit test.
If the image or mounts are unavailable, use the official
[Combine installation](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/)
and [CombineHarvester setup](https://github.com/cms-analysis/CombineHarvester#quick-start)
to install a mutually compatible pair in a separate area. Do not install them
inside the reconstruction `CMSSW_10_6_17` area. For an existing installation,
activate it and omit the `ak15_combine` prefix from the commands below.

### 14. Build The Card, Workspace And Expected Limit

Run the complete checked sequence once in a new directory:

```bash
ak15_combine python3 scripts/expected_limits.py --templates work/templates/histograms.root \
  --qcd-shapes work/templates/qcd_shapes.json --output work/limit --rmax 100
```

This one command performs several steps:

1. `build_card.py` checks the TH1 inputs and calls the CombineHarvester Python
   interface to write `datacard.txt` and `shapes.root`. Rates are the actual
   histogram integrals.
2. The standard `text2workspace.py` command builds `workspace.root` from the card.
3. `limit_checks.py` verifies that the workspace reproduces the supplied QCD
   histogram bins at the central and Up/Down uncertainty settings.
4. The standard `combine -M AsymptoticLimits` command computes the expected limit.
5. The runner checks the returned quantiles and writes `limit_summary.json`.

The two underlying Combine commands, executed inside `work/limit`, are shown
below for reference. The runner already executes them; do not repeat them over
the completed output:

```bash
text2workspace.py datacard.txt -m 125 -o workspace.root
combine -M AsymptoticLimits workspace.root -m 125 --run blind \
  --rMin 0 --rMax 100 -n .ExpectedOnly
```

`-m 125` is the model's bookkeeping mass label. `r` multiplies the supplied
signal histogram; `--rMax` sets its search bound. `--run blind` calculates
expected limits without reporting an observed-data limit. The asymptotic
method uses a large-sample approximation for the likelihood test statistic;
this command does not run an ensemble of random simulated experiments.

To stop at a datacard for inspection instead, use this alternative in a new
directory. It does not run a limit calculation:

```bash
ak15_combine python3 scripts/build_card.py --templates work/templates/histograms.root \
  --qcd-shapes work/templates/qcd_shapes.json --output work/card_only
```

### 15. Inspect The Fit Inputs And Result

```bash
sed -n '1,160p' work/limit/datacard.txt
python3 -m json.tool work/limit/build_report.json
python3 -m json.tool work/limit/workspace_response.json
python3 -m json.tool work/limit/limit_summary.json
ak15_combine root -l -b -q -e 'TFile f("work/limit/shapes.root"); f.ls(); f.GetDirectory("pass")->ls();'
ak15_combine root -l -b -q -e 'TFile f("work/limit/workspace.root"); auto w = (RooWorkspace*)f.Get("w"); w->Print();'
```

| File in `work/limit/` | Role |
| --- | --- |
| `fit_inputs.root` | Fit copy of the histograms; QCD bin errors adjusted to avoid counting the same uncertainty twice |
| `shapes.root` | Histograms written by CombineHarvester and referenced by the card |
| `datacard.txt` | Human-readable process, yield and uncertainty specification |
| `workspace.root` | Complete likelihood model produced by `text2workspace.py` |
| `command_0.json`, `command_1.json` | Exact external command arguments |
| `command_0.log`, `command_1.log` | Workspace conversion and Combine output |
| `workspace_response.json` | Numerical comparison of input QCD bins and workspace responses |
| `higgsCombine.ExpectedOnly.AsymptoticLimits.mH125.root` | Combine result tree |
| `limit_summary.json` | Checked expected median and uncertainty-band quantiles |

The central expected result is the median upper limit on `r` at 95% confidence.
The green/yellow-style bands usually shown around it describe expected-limit
variation under the background model; they are not classifier-accuracy errors.
Interpretation as a branching fraction additionally needs the physical signal
production cross section, reference branching fractions and filter conventions.

### 16. Plot Expected Limits

Return to the modern Python terminal. List successful summaries and signal
labels in a plotting manifest. The example points to `work/limit`:

```bash
source .venv/bin/activate
test ! -e inputs/limit_grid.json
cp examples/limit_grid.json inputs/limit_grid.json
python -m json.tool inputs/limit_grid.json
python scripts/plot_outputs.py limits --manifest inputs/limit_grid.json \
  --label ParticleNet --output work/limit_plot
```

Add one entry per independently built signal hypothesis when making a mass
scan. Make a separate manifest/plot for the other classifier.

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

## Tests

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
architecture roundtrips cover both full and compact profiles. They test model
loading and score preservation, not trained performance or GPU reproducibility.
GitHub Actions runs these checks on Linux and Windows with checksum-pinned
architecture sources.

## Scope

Not included: data discovery/download, credential handling, batch submission,
automatic recovery, pretrained weights, production input lists/results, plot
packs, or branching-fraction conversion. Full CMSSW execution, GPU training and
a fresh Combine fit remain integration checks. [FILES.txt](FILES.txt) is the
source inventory; ignored local environments/work products are not release files.

## Common Stopping Points

| Message or symptom | Check before retrying |
| --- | --- |
| `cmsRun` or ROOT unavailable | Enter the reconstruction container and activate the CMSSW area |
| CUDA unavailable | Confirm a GPU allocation and a compatible PyTorch build |
| Too few events per partition | Process more input events; the reconstruction smoke is not a training sample |
| Duplicate event/content | Review the source list and simulation; do not bypass partition checks |
| Existing output | Keep the prior result and choose a new output directory |
| Parent checksum or sample mismatch | Pair the exact Nano and compact file and use consistent sample labels |
| Test score replay differs | Check model source, bundle and numerical runtime; do not substitute saved predictions |
| Generator denominator differs | Sum the preselection weights of exactly the listed input files |
| Negative QCD or fit bins | Review control subtraction and common binning; do not clip negative contributions |
| Limit near `rMax` | Review the signal normalization and rerun in a new directory with an adequate search bound |
