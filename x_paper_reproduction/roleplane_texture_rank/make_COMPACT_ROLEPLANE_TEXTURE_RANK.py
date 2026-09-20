#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, json, zipfile
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out_dir",required=True); args=ap.parse_args()
    out=Path(args.out_dir); plots=out/"plots"; plots.mkdir(exist_ok=True)

    # Cross-stage figure: QK rank emergence vs register-role populations.
    hp=out/"head_rank_rigidity_all_blocks.csv"; rp=out/"role_stage_summary.csv"
    if hp.is_file() and rp.is_file():
        h=pd.read_csv(hp).groupby("block").agg(
            qk_effrank=("qk_effrank_energy_blank","mean"),
            qk_rank=("qk_rank_rel1e4_blank","mean"),
            rigid_heads=("phenotype",lambda s:int((s=="rigid_or_gated_positional").sum())),
            tracker_heads=("phenotype",lambda s:int((s=="visual_tracker").sum())),
        ).reset_index()
        r=pd.read_csv(rp)
        # weighted average over DTD + synthetic source means; descriptive only.
        r["ord"]=r.stage.map(lambda s:24 if s=="post23" else int(str(s).replace("pre","")))
        rm=r.groupby("ord").agg(role_reg=("role_reg_mean","mean"),hidden_mu=("hidden_mu_mean","mean")).reset_index()
        q=h.merge(rm,left_on="block",right_on="ord",how="left")
        q.to_csv(out/"rank_role_transition_summary.csv",index=False)

        fig,ax=plt.subplots(figsize=(10,5.8))
        ax.plot(h.block,h.qk_effrank,marker="o",label="mean centered-QK energy effective rank")
        ax.set_xlabel("block"); ax.set_ylabel("QK effective rank")
        ax2=ax.twinx()
        ax2.plot(rm.ord,rm.role_reg,marker="s",linestyle="--",label="manifest REG")
        ax2.plot(rm.ord,rm.hidden_mu,marker="x",linestyle=":",label="hidden mu")
        ax2.set_ylabel("mean role-token count")
        lines=ax.get_lines()+ax2.get_lines(); ax.legend(lines,[x.get_label() for x in lines],fontsize=8,loc="best")
        ax.set_title("Head-rank development vs native role-plane manifestation")
        ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(plots/"12_TRUE_RANK_VS_ROLE_EMERGENCE.png",dpi=220); plt.close(fig)

    # Combined short report.
    lines=["# CLIP fractal-rabbit-hole compact handoff",""]
    for report in ["REPORT_TEXTURE_ROLEPLANE.md"]:
        p=out/report
        if p.is_file():
            lines += p.read_text(encoding="utf-8").splitlines()+[""]
    if hp.is_file():
        h=pd.read_csv(hp)
        lines+=["## Head-rank landmarks"]
        for b in sorted(h.block.unique()):
            g=h[h.block.eq(b)]
            lines.append(f"- B{b}: mean QK-effrank={g.qk_effrank_energy_blank.mean():.3f}; "
                         f"rigid={int((g.phenotype=='rigid_or_gated_positional').sum())}; "
                         f"trackers={int((g.phenotype=='visual_tracker').sum())}")
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    include=[
        "config.json","model_audit.json","head_probe_audit.json","texture_probe_audit.json",
        "mu_basis.npz","mu_basis_report.json","REPORT.md","REPORT_TEXTURE_ROLEPLANE.md",
        "head_rank_rigidity_all_blocks.csv","head_rank_rigidity_block_summary.csv","head_rank_rigidity_correlations.csv",
        "rank_role_transition_summary.csv","dtd_sampling_audit.csv","dtd_per_class_summary.csv",
        "synthetic_family_summary.csv","sine_registermaxxing.csv","top_registermaxxing.txt",
        "role_stage_summary.csv","role_birth_summary.csv","role_position_frequency.csv.gz",
        "last_inverse_patch_role_summary.csv","last_inverse_frequency_profiles.csv.gz",
        "address_bus_channel_candidates.csv","conv1_color_coordinate_census.csv",
        "dtd_per_image.csv.gz","synthetic_per_image.csv.gz"
    ]
    zpath=out/"compact_summary_conv1_roleplane_texture_rank_compact.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n in include:
            p=out/n
            if p.is_file(): z.write(p,arcname=n)
        for p in sorted(plots.glob("*.png")): z.write(p,arcname=f"plots/{p.name}")
    print(f"Compact handoff: {zpath}")

if __name__=="__main__": main()
