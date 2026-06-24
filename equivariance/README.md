# Equivariance test

Checks that a trained jet tagger's output logit is invariant under the
symmetry the network actually has. The residual symmetry is the
subgroup of `SO+(1,3)` that

1. stabilizes the spurions
2. leaves the requested `tagging_features` invariant.

The script derives this residual subgroup from the saved config and samples
transformations from it. Inputs go through the exact same preprocessing as training/eval 
(`embed_tagging_data` with `cfg.data`), so spurions, mass regulator, jet boosting and 
feature standardization are applied identically before and after the transformation.

```bash
python equivariance/test_equivariance.py --run_dir runs/<exp>/<run>
```

For the typical settings (standard top-tagging config: beam `lightlike` + time reference,
`tagging_features=all`), the residual symmetry is rotations about the beam axis
(`xyrotation`).

Key options:

| flag | meaning |
|------|---------|
| `--run_dir` | run directory with `config.yaml` and `models/model_run<idx>.pt` |
| `--group` | `auto` (default, derived from config) or force `lorentz`/`rotation`/`boost`/`xyrotation`/`ztransform` for debugging |
| `--n_jets` / `--n_transforms` | number of jets / random transforms |
| `--per_jet` | independent group element per jet (default: one global element) |
| `--std_eta` | boost rapidity spread for boost-containing groups |
| `--data` | optional override path to the `.npz` dataset |

## Output

The script summarizes how much the output changes under the transformations and
writes `equivariance_<group>.json` into the run directory. For every
`(transform, jet)` pair it computes, between the transformed and original
output `y`:

- **logit absolute error** `|y(g·x) − y(x)|` (logit units)
- **logit relative error** `|y(g·x) − y(x)| / |y(x)|` (dimensionless)
- **probability absolute error** `|σ(y(g·x)) − σ(y(x))|` (bounded by 1)
- **probability relative error** `|σ(y(g·x)) − σ(y(x))| / |σ(y(x))|` (dimensionless)

Each is reported at three aggregation levels (each with `mean` and `std`):

- `per_transform` — averaged over jets, one entry per transform (std across jets:
  how uniformly a given transform affects different jets)
- `per_jet` — averaged over transforms, one entry per jet (std across transforms:
  how sensitive a given jet is to different group elements)
- `overall` — averaged over everything

`output_reference` gives the mean/std of the untransformed logits, so the
absolute error can be judged against the natural scale of the output. For an
exactly invariant model all errors are ~0.
