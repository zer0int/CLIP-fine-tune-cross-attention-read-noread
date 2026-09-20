#!/usr/bin/env python3
'probe rn control mechanism'
from __future__ import annotations

from probe_tools_repo import ensure_repo_root
ensure_repo_root()

import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from x_paper_reproduction.rn_control_mechinterp.synthetic import PatchAlignedSyntheticBank, DEFAULT_VARIANTS
from x_paper_reproduction.rn_control_mechinterp.core import load_model, preprocess_pil_batch, VisualConditionRunner
from x_paper_reproduction.rn_control_mechinterp.analysis import (
    DeltaCollector,
    randomized_svd_rows,
    principal_angles,
    rank_k_template,
    project_last_dim,
    displacement_metrics,
    b13_head_decomposition,
    pathway_decomposition,
    save_json,
)
from x_paper_reproduction.rn_control_mechinterp.jacobian import top_input_singular_directions
from x_paper_reproduction.rn_control_mechinterp.plots import render_all_plots


DEFAULT_CHECKPOINT = "REPLACE_WITH_CHECKPOINT.pt"


def parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def iter_batches(n: int, batch: int):
    for start in range(0, n, batch):
        yield list(range(start, min(n, start + batch)))


def region_mask_from_specs(specs, device: torch.device) -> torch.Tensor:
    grid = specs[0].grid_size
    out = torch.zeros(len(specs), grid * grid, dtype=torch.bool, device=device)
    for i, spec in enumerate(specs):
        x0, y0, x1, y1 = spec.bbox_patch
        for y in range(y0, y1):
            out[i, y * grid + x0:y * grid + x1] = True
    return out


def save_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # Diagnostic tables are intentionally allowed to contain row-specific
    # fields (e.g. mean_direction_cos exists only for cross-condition
    # alignments). Build a stable first-seen union of all keys instead of
    # assuming the first row defines the entire CSV schema.
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)


def average_decomp_rows(all_rows: list[list[dict]]) -> list[dict]:
    acc = defaultdict(lambda: defaultdict(list))
    for rows in all_rows:
        for r in rows:
            key = (r["head"], r["token_group"])
            for k, v in r.items():
                if k not in {"head", "token_group"}:
                    acc[key][k].append(float(v))
    out = []
    for (head, group), vals in sorted(acc.items()):
        row = {"head": head, "token_group": group}
        for k, xs in vals.items():
            row[k] = float(np.mean(xs))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description="READ_NULL single-block steering mechanistic analysis suite")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--module-root", default=".", help="Repository root containing attnclip_mechinterp_xattn")
    ap.add_argument("--output-dir", default="rn_control_mechinterp")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--n-scenes", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--variants", default="text,phase,shred,sine,checker,blank")
    ap.add_argument("--positions", default="top,center,bottom,left,right")
    ap.add_argument("--capture-blocks", default="12,13,14,15,16,17,18,19,20,21,22,23")
    ap.add_argument("--svd-k", type=int, default=8)
    ap.add_argument("--patch-samples-per-image", type=int, default=4)
    ap.add_argument("--svd-max-rows-per-view", type=int, default=8192,
                    help="Hard RAM bound for rows retained per SVD view/collector; <=0 disables the cap")
    ap.add_argument("--empty-cache-every", type=int, default=16,
                    help="On CUDA, release unused allocator cache every N batches; <=0 disables")
    ap.add_argument("--decomp-scenes", type=int, default=64)
    ap.add_argument("--contrast-scenes", type=int, default=256)
    ap.add_argument("--causal-scenes", type=int, default=128)
    ap.add_argument("--causal-ranks", default="1,2,4,8")
    ap.add_argument("--jacobian-scenes", type=int, default=2)
    ap.add_argument("--jacobian-k", type=int, default=4)
    ap.add_argument("--jacobian-iters", type=int, default=7)
    ap.add_argument("--font", action="append", default=[])
    ap.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False, help="Render diagnostic PNGs after numerical analysis (default: enabled)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    loaded = load_model(args.checkpoint, package_root=args.module_root, device=args.device)
    model, preprocess, device = loaded.model, loaded.preprocess, loaded.device
    runner = VisualConditionRunner(model)

    resolution = int(model.visual.input_resolution)
    patch_size = int(model.visual.conv1.kernel_size[0])
    grid = resolution // patch_size
    variants = parse_strs(args.variants)
    positions = parse_strs(args.positions)
    blocks = parse_ints(args.capture_blocks)
    ranks = parse_ints(args.causal_ranks)

    bank = PatchAlignedSyntheticBank(
        resolution=resolution,
        patch_size=patch_size,
        seed=args.seed,
        positions=positions,
        font_paths=args.font,
    )
    bank.preview(out / "synthetic_preview.png", n_scenes=min(5, args.n_scenes), variants=variants)

    settings = vars(args).copy()
    settings.update({
        "resolved_resolution": resolution,
        "resolved_patch_size": patch_size,
        "resolved_grid": grid,
        "rn_insert_block": runner.insert_block,
        "rn_norm": float(model.visual.read_null_token.detach().float().norm().cpu()),
    })
    save_json(out / "settings.json", settings)

    manifest_rows = []
    for i in range(args.n_scenes):
        spec, _ = bank.scene(i)
        manifest_rows.append(spec.to_dict())
    save_rows(out / "synthetic_manifest.csv", manifest_rows)

    embed_rows = []
    decomp_batches: list[list[dict]] = []
    pathway_batches: list[dict] = []

    token_count = 1 + grid * grid
    width = int(model.visual.conv1.out_channels)

    # IMPORTANT LOW-RAM DESIGN:
    # Process one visual variant at a time, immediately reduce its retained rows
    # to compact SVD results, then free the collectors before moving to the next
    # variant.  The previous implementation retained every variant x block row
    # bank simultaneously, so RAM scaled linearly with the entire experiment.
    svd_results = {}
    svd_summary = []
    max_rows = None if args.svd_max_rows_per_view <= 0 else args.svd_max_rows_per_view

    for variant in variants:
        print(f"[variant] {variant}: bounded SVD rows/view={max_rows}")
        variant_collectors: dict[int, DeltaCollector] = {}
        for b in blocks:
            variant_collectors[b] = DeltaCollector(
                width=width, tokens=token_count,
                patch_samples_per_image=args.patch_samples_per_image,
                max_rows_per_view=max_rows,
                seed=args.seed + 1009 * b + sum((i + 1) * ord(c) for i, c in enumerate(variant)) % 997,
            )

        for batch_index, ids in enumerate(iter_batches(args.n_scenes, args.batch_size)):
            specs, pil = [], []
            for sid in ids:
                spec, imgs = bank.scene(sid)
                specs.append(spec); pil.append(imgs[variant])
            batch = preprocess_pil_batch(pil, preprocess, device)
            region = region_mask_from_specs(specs, device)

            # Only BASE and TAG need captured block states.  FULL and NOP are
            # embedding-only controls, so do not duplicate seven blocks of
            # activations for them. Captured states are moved to CPU immediately.
            base = runner.run(batch, condition="base", capture_blocks=blocks)
            tag = runner.run(batch, condition="rn_tag", capture_blocks=blocks)

            for b in blocks:
                delta = tag.states[b] - base.states[b]
                variant_collectors[b].add(delta, region_masks_bp=region)

            eb = F.normalize(base.embedding.float(), dim=-1)
            et = F.normalize(tag.embedding.float(), dim=-1)

            full = runner.run(batch, condition="rn_full", capture_blocks=())
            ef = F.normalize(full.embedding.float(), dim=-1)
            del full
            nop = runner.run(batch, condition="rn_tag_zero_v", capture_blocks=())
            en = F.normalize(nop.embedding.float(), dim=-1)

            for j, sid in enumerate(ids):
                tag_shift = (tag.embedding[j] - base.embedding[j]).norm()
                nop_shift = (nop.embedding[j] - base.embedding[j]).norm()
                embed_rows.append({
                    "scene_id": sid,
                    "variant": variant,
                    "cos_tag_vs_full": float((et[j] * ef[j]).sum()),
                    "cos_base_vs_tag": float((eb[j] * et[j]).sum()),
                    "cos_base_vs_nop": float((eb[j] * en[j]).sum()),
                    "tag_shift_norm": float(tag_shift),
                    "nop_shift_norm": float(nop_shift),
                    "nop_over_tag_shift": float(nop_shift / tag_shift.clamp_min(1e-12)),
                })

            # Expensive exact B13 pathway decomposition only on an initial subset.
            decomp_ids = [sid for sid in ids if sid < args.decomp_scenes]
            if variant == "text" and decomp_ids:
                take = len(decomp_ids)
                pair = runner.block13_pair(batch[:take])
                decomp_batches.append(b13_head_decomposition(pair, None))
                pathway_batches.append(pathway_decomposition(pair))
                del pair

            del nop, base, tag, batch, region, eb, et, ef, en
            if device.type == "cuda" and args.empty_cache_every > 0 and (batch_index + 1) % args.empty_cache_every == 0:
                torch.cuda.empty_cache()
            if (batch_index + 1) % 32 == 0:
                gc.collect()

        # Reduce this variant NOW; only compact SVDResult objects survive.
        for b, collector in variant_collectors.items():
            mean_template = collector.mean_template()
            np.save(out / f"mean_template__{variant}__B{b}.npy", mean_template.numpy())
            universal = collector.universal_template_fraction()
            for view in ("cls", "patch_mean", "patch_sample", "region_mean", "outside_mean"):
                try:
                    rows = collector.rows(view)
                except KeyError:
                    continue
                res = randomized_svd_rows(rows, k=args.svd_k, center=False)
                key = (variant, b, view)
                svd_results[key] = res
                res.save(out / "svd" / f"{variant}__B{b}__{view}.npz")
                svd_summary.append({
                    "variant": variant, "block": b, "view": view,
                    "n_rows": res.n_rows,
                    "sv1": float(res.singular_values[0]),
                    "energy_pc1": float(res.explained_energy[0]),
                    "energy_topk": float(res.explained_energy.sum()),
                    "universal_mean_template_fraction": universal,
                })
                del rows
        del variant_collectors
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_rows(out / "embedding_condition_summary_per_scene.csv", embed_rows)
    save_rows(out / "svd_summary.csv", svd_summary)

    # RN text-delta subspace vs the other RN conditions.
    angle_rows = []
    for b in blocks:
        for view in ("cls", "patch_mean", "patch_sample", "region_mean", "outside_mean"):
            text_key = ("text", b, view)
            if text_key not in svd_results:
                continue
            for ctrl in variants:
                if ctrl == "text" or (ctrl, b, view) not in svd_results:
                    continue
                pa = principal_angles(
                    svd_results[text_key].right_vectors,
                    svd_results[(ctrl,b,view)].right_vectors,
                    k=min(args.svd_k, 8),
                )
                angle_rows.append({
                    "block": b, "view": view, "comparison": f"RN(text) vs RN({ctrl})",
                    "mean_subspace_cos": pa["mean_cosine"],
                    "min_angle_deg": pa["min_angle_deg"],
                    "max_angle_deg": pa["max_angle_deg"],
                })

    # Independent input-text contrast: base(text) - base(phase/sine).
    contrast_collectors = {}
    contrast_n = min(args.contrast_scenes, args.n_scenes)
    for ctrl in ("phase", "sine"):
        for b in blocks:
            contrast_collectors[(ctrl,b)] = DeltaCollector(
                width=width, tokens=token_count,
                patch_samples_per_image=args.patch_samples_per_image,
                max_rows_per_view=max_rows,
                seed=args.seed + 4001 + b,
            )
    for ids in iter_batches(contrast_n, args.batch_size):
        specs, imgs_text = [], []
        ctrl_imgs = {"phase": [], "sine": []}
        for sid in ids:
            spec, imgs = bank.scene(sid)
            specs.append(spec); imgs_text.append(imgs["text"])
            for ctrl in ctrl_imgs: ctrl_imgs[ctrl].append(imgs[ctrl])
        region = region_mask_from_specs(specs, device)
        tb = preprocess_pil_batch(imgs_text, preprocess, device)
        text_base = runner.run(tb, condition="base", capture_blocks=blocks)
        for ctrl, pils in ctrl_imgs.items():
            cb = preprocess_pil_batch(pils, preprocess, device)
            cbase = runner.run(cb, condition="base", capture_blocks=blocks)
            for b in blocks:
                contrast_collectors[(ctrl,b)].add(text_base.states[b] - cbase.states[b], region_masks_bp=region)

    for (ctrl,b), collector in contrast_collectors.items():
        for view in ("cls", "patch_mean", "patch_sample", "region_mean", "outside_mean"):
            try:
                cres = randomized_svd_rows(collector.rows(view), k=args.svd_k, center=False)
            except KeyError:
                continue
            cres.save(out / "svd" / f"input_text_minus_{ctrl}__B{b}__{view}.npz")
            rkey = ("text", b, view)
            if rkey in svd_results:
                pa = principal_angles(svd_results[rkey].right_vectors, cres.right_vectors, k=min(args.svd_k, 8))
                mean_rn = torch.from_numpy(svd_results[rkey].mean).float()
                mean_input = torch.from_numpy(cres.mean).float()
                mean_cos = float(F.cosine_similarity(mean_rn[None], mean_input[None], dim=-1)[0])
                angle_rows.append({
                    "block": b, "view": view,
                    "comparison": f"RN(text) vs input(text-{ctrl})",
                    "mean_subspace_cos": pa["mean_cosine"],
                    "min_angle_deg": pa["min_angle_deg"],
                    "max_angle_deg": pa["max_angle_deg"],
                    "mean_direction_cos": mean_cos,
                })
    save_rows(out / "subspace_alignment.csv", angle_rows)

    # Aggregate B13 decomposition.
    if decomp_batches:
        save_rows(out / "b13_head_kv_decomposition.csv", average_decomp_rows(decomp_batches))
        keys = pathway_batches[0].keys()
        pathway_avg = {k: float(np.mean([x[k] for x in pathway_batches])) for k in keys}
        save_json(out / "b13_attention_vs_mlp_pathway.json", pathway_avg)

    # Held-out causal compression test at post-B13 on readable text and sine.
    # Fit the steering subspace/template on an earlier split and test only on the tail split.
    causal_rows = []
    causal_n = min(args.causal_scenes, max(1, args.n_scenes // 3))
    test_start = max(1, args.n_scenes - causal_n)
    fit_n = test_start
    for variant in ("text", "sine"):
        if variant not in variants:
            continue
        fit_collector = DeltaCollector(
            width=width, tokens=token_count,
            patch_samples_per_image=max(args.patch_samples_per_image, 4),
            max_rows_per_view=max_rows,
            seed=args.seed + 7001 + (0 if variant == "text" else 1),
        )
        for ids in iter_batches(fit_n, args.batch_size):
            pils, specs = [], []
            for sid in ids:
                spec, imgs = bank.scene(sid); specs.append(spec); pils.append(imgs[variant])
            x = preprocess_pil_batch(pils, preprocess, device)
            region = region_mask_from_specs(specs, device)
            base = runner.run(x, condition="base", capture_blocks=[runner.insert_block])
            tag = runner.run(x, condition="rn_tag", capture_blocks=[runner.insert_block])
            fit_collector.add(tag.states[runner.insert_block] - base.states[runner.insert_block], region_masks_bp=region)

        template = fit_collector.mean_template().to(device)
        feature_rows = torch.cat([
            fit_collector.rows("cls"),
            fit_collector.rows("patch_sample"),
        ], dim=0)
        feat_res = randomized_svd_rows(feature_rows, k=max(ranks), center=False)
        basis_all = torch.from_numpy(feat_res.right_vectors).float().to(device)

        buckets = defaultdict(list)
        test_ids_all = list(range(test_start, args.n_scenes))
        for start in range(0, len(test_ids_all), args.batch_size):
            ids = test_ids_all[start:start + args.batch_size]
            pils = []
            for sid in ids:
                _, imgs = bank.scene(sid); pils.append(imgs[variant])
            x = preprocess_pil_batch(pils, preprocess, device)
            base = runner.run(x, condition="base", capture_blocks=[runner.insert_block])
            tag = runner.run(x, condition="rn_tag", capture_blocks=[runner.insert_block])
            bs = base.states[runner.insert_block].to(device)
            rs = tag.states[runner.insert_block].to(device)
            true_delta = rs - bs
            for rank in ranks:
                tm = rank_k_template(template, rank).to(device)
                suff_fixed = runner.continue_from_post_b13(bs + tm[None])
                basis = basis_all[:rank]
                proj = project_last_dim(true_delta, basis)
                suff_paired = runner.continue_from_post_b13(bs + proj)
                necessary_removed = runner.continue_from_post_b13(rs - proj)
                buckets[(rank,"fixed_mean_template")].append((base.embedding.cpu(), tag.embedding.cpu(), suff_fixed.cpu()))
                buckets[(rank,"paired_feature_projection")].append((base.embedding.cpu(), tag.embedding.cpu(), suff_paired.cpu()))
                buckets[(rank,"remove_feature_projection")].append((base.embedding.cpu(), tag.embedding.cpu(), necessary_removed.cpu()))

        for (rank, mode), triples in buckets.items():
            b = torch.cat([x[0] for x in triples])
            t = torch.cat([x[1] for x in triples])
            y = torch.cat([x[2] for x in triples])
            m = displacement_metrics(b, t, y)
            causal_rows.append({
                "variant":variant, "rank":rank, "mode":mode,
                "fit_scenes": fit_n, "test_scenes": len(test_ids_all), **m
            })
    save_rows(out / "causal_low_rank_test.csv", causal_rows)

    # Local Jacobian of frozen B13: RN input -> surviving post-B13 state.
    jn = min(args.jacobian_scenes, args.n_scenes)
    if jn > 0:
        pils = [bank.scene(i)[1]["text"] for i in range(jn)]
        xb = preprocess_pil_batch(pils, preprocess, device)
        for point in ("trained", "zero"):
            js = top_input_singular_directions(
                runner=runner, images=xb, k=args.jacobian_k,
                iterations=args.jacobian_iters, point=point, seed=args.seed,
            )
            js.save(out / "jacobian" / f"b13_rn_input_spectrum__{point}.npz")

    overview = {
        "checkpoint": args.checkpoint,
        "n_scenes": args.n_scenes,
        "variants": variants,
        "resolution": resolution,
        "patch_size": patch_size,
        "insert_block": runner.insert_block,
        "outputs": {
            "svd_summary": "svd_summary.csv",
            "subspace_alignment": "subspace_alignment.csv",
            "b13_head_decomposition": "b13_head_kv_decomposition.csv",
            "pathway_decomposition": "b13_attention_vs_mlp_pathway.json",
            "causal_low_rank_test": "causal_low_rank_test.csv",
            "jacobian": "jacobian/",
        },
    }
    save_json(out / "OVERVIEW.json", overview)
    if args.plots:
        try:
            made = render_all_plots(out)
            print(f"[plots] wrote {len(made)} figure(s) to {out / 'plots'}")
        except Exception as exc:
            print(f"[plots WARNING] numerical analysis completed, but plotting failed: {type(exc).__name__}: {exc}")
            print(f"[plots WARNING] rerun later with: python plot_existing_results.py \"{out}\"")
    print(f"[done] {out}")




if __name__ == "__main__":
    main()
