# Third-Party Attribution

## Included Source

`cmssw/JMEAnalysis/JetToolbox/python/jetToolbox_cff.py` identifies the Jet
Substructure Toolbox and Alejandro Gomez Espinosa, with contributions from
others. Its original attribution header is preserved. The upstream project is
[cms-jet/JetToolbox](https://github.com/cms-jet/JetToolbox). Exact revision and
applicable license of this imported copy remain unverified. This particular
copy is included with maintainer authorization and its existing credits;
publication does not assign it a new license.
The adjacent `__init__.py` is a package marker.

The CMSSW configs/custom plugins, ROOT reducer and analysis Python code were
adapted from individually selected source files. Their original source bytes are
identified by code digests in `SOURCE_PROVENANCE.json`, without machine paths or
run-directory names. Portability edits are present, so source digests are not
digests of the modified files. See `LICENSE_STATUS.md` for reuse conditions.

## External Architecture Source

ParticleNet and Particle Transformer are author-reference architectures, not
new architectures claimed by this project. ParticleNet is described by Huilin
Qu and Loukas Gouskos in [ParticleNet: Jet Tagging via Particle Clouds](https://arxiv.org/abs/1902.08570).
Particle Transformer is described
in [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772).
The [official implementation](https://github.com/jet-universe/particle_transformer)
distributes an [MIT license](https://github.com/jet-universe/particle_transformer/blob/main/LICENSE)
with attribution to Huilin Qu, Congqiao Li and Sitian Qian.

`tools/fetch_models.py` retrieves both architecture files from
[this Weaver revision](https://github.com/hqucms/weaver-core/tree/34e2ee20ca7d1b94446f1bf18b361c5865c4b6d4), checks their SHA256 digests and
downloads that revision's complete MIT license. The ParticleNet lineage also
references Dynamic Graph CNN; its MIT notice is retrieved from commit
`f765b469a67730658ba554e97dc11723a7bab628`. All notices remain beside the files.
The sole downloaded-source adaptation replaces the ParticleTransformer logger
import with Python's standard logging module. Numerical code is unchanged.
The selected revision predates later attention implementation changes; do not
substitute a newer upstream file when loading an existing checkpoint.

A Git commit identifies a version of the public dependency. A SHA256 checksum
identifies the contents of a downloaded file. These checks prevent an upstream
code update or an altered download from silently changing the model definition.
The download receipt, `external/models/upstream.json`, records both versions
and checksums locally.

## Runtime Dependencies

CMSSW, ROOT, PyTorch, NumPy, Awkward Array, uproot, scikit-learn, pytest,
CombineHarvester and HiggsAnalysis/CombinedLimit are external dependencies,
not redistributed source here. Install them from their official distributions
and retain the licenses supplied by those distributions. This repository's
license status does not replace or override any dependency license.
