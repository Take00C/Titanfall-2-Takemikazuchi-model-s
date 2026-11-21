# research/

Scripts written during the feasibility study (2026-11-14). They are **not yet wired
into `build.py`**; the Titanfall 2 side of the pipeline (stage 3 onward) will be built
from them after review.

| file | what it does | status |
|---|---|---|
| `tf2/tf2lzham_py.py` | Pure-Python LZHAM decompressor with Titanfall 2's parameters (dictionary 2^20). A port made by reading the C++ in [pg9182/tf2lzham](https://github.com/pg9182/tf2lzham) (MIT, © 2022 pg9182), which wraps Rich Geldreich's LZHAM alpha (MIT). | Verified: 26 models and 1,019 scripts extracted from the retail VPKs matched the CRC32 stored in the VPK directory. Slow (pure Python) but needs no binary. |
| `tf2/vpk_extract.py` | Reads a Respawn `englishclient_*_dir.vpk` index and extracts matching entries, read-only. `python vpk_extract.py <dir.vpk> <outdir> <regex> ...` | Verified as above. |
| `tf2/mdlinfo.py` | Structural dump of a studio v53 `.mdl` to JSON: bones, attachments, hitbox sets, bodygroups, materials, skin families, IK chains, include models, sequences, keyvalues. `python mdlinfo.py <outdir> <file.mdl> ...` | Used to produce the stock Ronin facts in the plan. Own implementation, not cross-checked against another parser yet. |
| `tf2/rpak7_list.py` | Lists the assets in an uncompressed RPak v7 (the format RePak writes for Titanfall 2). | Parsed three local paks exactly to end of file. |
| `tf2/rpak7_guids.py` | Port of the RTech asset-GUID hash; shows that an `.mdl` material name `X` resolves to the rpak asset `material/X_skn.rpak`. | Matched every material in Northstar's own custom models. |

The structure dumps of the stock Ronin models that these produced are under
`build/ronin/mdlinfo/` (regenerable, not committed).
