import argparse
import json
import os

import torch
from hydra.utils import instantiate
from lloca.utils.rand_transforms import (
    rand_boost,
    rand_lorentz,
    rand_rotation,
    rand_xyrotation,
    rand_ztransform,
)
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from experiments.tagging.dataset import TopTaggingDataset
from experiments.tagging.embedding import embed_tagging_data

# Continuous generators of SO+(1,3): rotations R and boosts B around/along x,y,z.
ALL_GENERATORS = {"Rx", "Ry", "Rz", "Bx", "By", "Bz"}

GROUPS = {
    "lorentz": rand_lorentz,  # full SO+(1,3): rotation * boost
    "rotation": rand_rotation,  # spatial SO(3)
    "boost": rand_boost,  # pure boosts
    "xyrotation": rand_xyrotation,  # rotations about z (in the xy-plane)
    "ztransform": rand_ztransform,  # z-boost + z-rotation
}
BOOST_GROUPS = {"lorentz", "boost", "ztransform"}

# Maps a set of generators to the matching lloca sampler.
GENERATORS_TO_GROUP = {
    frozenset(ALL_GENERATORS): "lorentz",
    frozenset({"Rx", "Ry", "Rz"}): "rotation",
    frozenset({"Bx", "By", "Bz"}): "boost",
    frozenset({"Rz", "Bz"}): "ztransform",
    frozenset({"Rz"}): "xyrotation",
}


def residual_generators(cfg_data):
    """Return the set of continuous generators left unbroken by spurions + features."""
    gens = set(ALL_GENERATORS)

    if cfg_data.add_time_reference:
        # time reference (1,0,0,0) is preserved only by spatial rotations
        gens -= {"Bx", "By", "Bz"}
    beam = cfg_data.beam_reference
    if beam in ("lightlike", "timelike", "spacelike"):
        # a fixed beam vector along z: only rotations about z survive, and the
        # z-boost rescales it -> broken as well
        gens -= {"Rx", "Ry", "Bx", "By", "Bz"}
    elif beam == "all":
        # three independent beam directions leave no continuous rotation
        gens -= {"Rx", "Ry", "Rz", "Bx", "By", "Bz"}

    # --- tagging-feature symmetry (features recomputed in the lab frame) ---
    tf = cfg_data.tagging_features
    if tf == "all":
        gens &= {"Rz"}  # all features are SO(2)-invariant only
    elif tf == "zinvariant":
        gens &= {"Rz", "Bz"}
    elif tf == "so3invariant":
        gens &= {"Rx", "Ry", "Rz"}
    # tf is None -> pure four-momenta, no extra restriction

    return gens


def resolve_group(cfg_data):
    """Pick the sampler name matching the residual symmetry of this config."""
    gens = residual_generators(cfg_data)
    name = GENERATORS_TO_GROUP.get(frozenset(gens))
    return gens, name


def load_model(run_dir, model_idx, device):
    """Instantiate the model from the saved config and load its weights."""
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    model = instantiate(cfg.model)
    ckpt_path = os.path.join(run_dir, "models", f"model_run{model_idx}.pt")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, cfg


def load_jets(cfg, n_jets, data_path, device):
    """Load ``n_jets`` jets from the test split and batch them with torch_geometric."""
    if data_path is None:
        data_path = os.path.join(cfg.data.data_dir, f"toptagging_{cfg.data.dataset}.npz")
    dataset = TopTaggingDataset()
    dataset.load_data(
        data_path,
        "test",
        network_float64=cfg.use_float64,
        momentum_float64=cfg.data.momentum_float64,
    )
    dataset.data_list = dataset.data_list[:n_jets]
    loader = DataLoader(dataset=dataset, batch_size=n_jets)
    return next(iter(loader)).to(device)


def run_model(model, fourmomenta, scalars, ptr, cfg, momentum_dtype, dtype):
    """Run the model through the standard preprocessing (embed_tagging_data)."""
    embedding = embed_tagging_data(
        fourmomenta.to(momentum_dtype),
        scalars.to(dtype),
        ptr.clone(),  # embed_tagging_data mutates ptr in place
        cfg.data,
    )
    embedding["num_graphs"] = len(ptr) - 1
    y_pred, _, _ = model(embedding)
    return y_pred.squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="Run directory with config.yaml + models/")
    parser.add_argument("--model_idx", type=int, default=0, help="Which model_run<idx>.pt to load")
    parser.add_argument("--data", default=None, help="Optional override path to the .npz dataset")
    parser.add_argument(
        "--group",
        default="auto",
        choices=["auto", *GROUPS],
        help="Symmetry group to test. 'auto' derives the residual symmetry from the config "
        "(spurions + tagging_features). Override only for debugging.",
    )
    parser.add_argument("--n_jets", type=int, default=128, help="Number of jets to test")
    parser.add_argument("--n_transforms", type=int, default=16, help="Random transforms to apply")
    parser.add_argument(
        "--per_jet",
        action="store_true",
        help="Use an independent group element per jet (default: one global element)",
    )
    parser.add_argument(
        "--std_eta", type=float, default=1.0, help="Boost rapidity std (boost groups)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    model, cfg = load_model(args.run_dir, args.model_idx, args.device)
    momentum_dtype = torch.float64 if cfg.data.momentum_float64 else torch.float32
    dtype = torch.float64 if cfg.use_float64 else torch.float32

    gens, auto_group = resolve_group(cfg.data)
    print(
        f"Residual symmetry from config: spurions(beam={cfg.data.beam_reference}, "
        f"time={cfg.data.add_time_reference}), tagging_features={cfg.data.tagging_features}"
    )
    print(f"Unbroken generators: {sorted(gens) or '(none)'} -> group: {auto_group}")

    if args.group == "auto":
        group = auto_group
        if group is None:
            raise SystemExit(
                "No lloca sampler matches the residual symmetry "
                f"{sorted(gens)}; pass --group explicitly to test a subgroup."
            )
    else:
        group = args.group
        if group != auto_group:
            print(f"  (override: testing '{group}' instead of the residual '{auto_group}')")

    batch = load_jets(cfg, args.n_jets, args.data, args.device)
    fourmomenta, scalars, ptr = batch.x, batch.scalars, batch.ptr
    if scalars.shape[-1] > 0:
        print(
            "WARNING: constituents carry scalar features that are NOT transformed; "
            "results are only meaningful if those features are Lorentz invariant."
        )

    n_jets = len(ptr) - 1
    batch_idx = batch.batch  # maps each constituent to its jet
    sampler = GROUPS[group]
    sampler_kwargs = {"device": args.device, "dtype": momentum_dtype, "generator": generator}
    if group in BOOST_GROUPS:
        sampler_kwargs["std_eta"] = args.std_eta

    with torch.no_grad():
        y0 = run_model(model, fourmomenta, scalars, ptr, cfg, momentum_dtype, dtype)

        outputs = []  # output for each transform
        for _ in range(args.n_transforms):
            shape = (n_jets,) if args.per_jet else (1,)
            g = sampler(torch.Size(shape), **sampler_kwargs)
            g_per_particle = g[batch_idx] if args.per_jet else g.expand(n_jets, 4, 4)[batch_idx]

            transformed = torch.einsum("nij,nj->ni", g_per_particle, fourmomenta.to(momentum_dtype))
            outputs.append(run_model(model, transformed, scalars, ptr, cfg, momentum_dtype, dtype))

    y0 = y0.double()
    y = torch.stack(outputs).double()  # (n_transforms, n_jets)

    # Errors per (transform, jet), in two representations of the output:
    #  - the raw classifier logit
    #  - the tagging probability sigmoid(logit), which is what the decision uses
    abs_err = (y - y0).abs()  # (T, J), logit units
    rel_err = abs_err / y0.abs().clamp_min(1e-8)  # (T, J), dimensionless

    p0, p = torch.sigmoid(y0), torch.sigmoid(y)
    abs_err_prob = (p - p0).abs()  # (T, J), probability units (bounded by 1)
    rel_err_prob = abs_err_prob / p0.abs().clamp_min(1e-8)  # (T, J), dimensionless

    def stats(x):
        return {"mean": x.mean().item(), "std": x.std(unbiased=False).item()}

    def reduce_over(mat, dim):
        # collapse `dim`, keep per-remaining-index mean and std
        mean = mat.mean(dim=dim)
        std = mat.std(dim=dim, unbiased=False)
        return [{"mean": m, "std": s} for m, s in zip(mean.tolist(), std.tolist(), strict=True)]

    def metric_block(mat):
        return {
            # one entry per transform: averaged over jets
            "per_transform": reduce_over(mat, dim=1),
            # one entry per jet: averaged over transforms
            "per_jet": reduce_over(mat, dim=0),
            # single number: averaged over everything
            "overall": stats(mat),
        }

    summary = {
        "run_dir": os.path.abspath(args.run_dir),
        "model_idx": args.model_idx,
        "group": group,
        "auto_group": auto_group,
        "unbroken_generators": sorted(gens),
        "n_jets": n_jets,
        "n_transforms": args.n_transforms,
        "per_jet_transform": args.per_jet,
        "std_eta": args.std_eta if group in BOOST_GROUPS else None,
        "seed": args.seed,
        "absolute_error": metric_block(abs_err),  # |y(g.x) - y(x)|, logit units
        "relative_error": metric_block(rel_err),  # |y(g.x) - y(x)| / |y(x)|
        "absolute_error_prob": metric_block(abs_err_prob),  # |p(g.x) - p(x)|
        "relative_error_prob": metric_block(rel_err_prob),  # |p(g.x) - p(x)| / |p(x)|
    }

    out_path = os.path.join(args.run_dir, f"equivariance_{group}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    abs_o = summary["absolute_error"]["overall"]
    rel_o = summary["relative_error"]["overall"]
    abs_p = summary["absolute_error_prob"]["overall"]
    rel_p = summary["relative_error_prob"]["overall"]
    print(f"\nEquivariance summary: group={group}, jets={n_jets}, transforms={args.n_transforms}")
    print(f"  logit  absolute error mean / std : {abs_o['mean']:.4e} / {abs_o['std']:.4e}")
    print(f"  logit  relative error mean / std : {rel_o['mean']:.4e} / {rel_o['std']:.4e}")
    print(f"  prob   absolute error mean / std : {abs_p['mean']:.4e} / {abs_p['std']:.4e}")
    print(f"  prob   relative error mean / std : {rel_p['mean']:.4e} / {rel_p['std']:.4e}")
    print("  (per-transform and per-jet breakdowns saved to json)")
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
