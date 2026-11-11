# Third-Party Attribution

## Included Source

`cmssw/JMEAnalysis/JetToolbox/python/jetToolbox_cff.py` identifies the Jet
Substructure Toolbox and Alejandro Gomez Espinosa, with contributions from
others. Its original attribution header is preserved. The upstream project is
[cms-jet/JetToolbox](https://github.com/cms-jet/JetToolbox). Exact revision and
applicable license of this imported copy remain unverified; no replacement
license has been invented. The adjacent `__init__.py` is a package marker.

The CMSSW configs/custom plugins, ROOT reducer and analysis Python code were
exported as individually selected source files. Their original source bytes are
identified by code digests in `SOURCE_PROVENANCE.json`, without machine paths or
run-directory names. Portability edits are present, so source digests are not
digests of the modified exported files. Their licensing requires maintainer review.

## External Architecture Source

ParticleNet and Particle Transformer are author-reference architectures, not
new architectures claimed by this project. Particle Transformer is described
in [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772).
The [official implementation](https://github.com/jet-universe/particle_transformer)
distributes an [MIT license](https://github.com/jet-universe/particle_transformer/blob/main/LICENSE)
with attribution to Huilin Qu, Congqiao Li and Sitian Qian.

`tools/fetch_models.py` retrieves both architecture files from Weaver commit
`34e2ee20ca7d1b94446f1bf18b361c5865c4b6d4`, checks their SHA256 digests and
downloads that revision's complete MIT license. The ParticleNet lineage also
references Dynamic Graph CNN; its MIT notice is retrieved from commit
`f765b469a67730658ba554e97dc11723a7bab628`. All notices remain beside the files.
The sole downloaded-source adaptation replaces the ParticleTransformer logger
import with Python's standard logging module. Numerical code is unchanged.
The selected revision predates later attention implementation changes; do not
substitute a newer upstream file when loading an existing checkpoint.

## Runtime Dependencies

CMSSW, ROOT, PyTorch, NumPy, Awkward Array, uproot, scikit-learn, pytest,
CombineHarvester and HiggsAnalysis/CombinedLimit are external dependencies,
not redistributed source here. Install them from their official distributions
and retain the licenses supplied by those distributions. This repository's
unresolved license status does not replace or override any dependency license.
