"""Regenerate every manuscript figure from the released decision-level dataset.

Produces, in ``figures/``:
  fig1.png  admission rate by justification framing, per model
  fig2.png  decomposition of the shift into generation, recording, risk content
  fig3.png  ROC-space operating points by framing, per model
  fig4.png  dose-response in the number of required risks, with verbosity control
  fig5.png  external replication of the valence contrast on MIMIC-IV-ED

Usage:
    python -m elicitation.figures [--tiff DIR]
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .analysis import MODELS, admission_rate, contrast, load, operating_point  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures"
# When set, each figure is also written here as a journal-ready TIFF.
TIFF_DIR: Path | None = None

GREY, TEAL, SLATE, RED, DARKRED = "#9aa7b4", "#3a9a7a", "#6c7a89", "#d1495b", "#8c2f39"


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _write_tiff(fig, path: Path, dpi=600, max_width_in=7.5):
    """RGB, LZW-compressed TIFF no wider than ``max_width_in`` at ``dpi``.

    These are the PLOS figure limits (300 to 600 dpi, 2.63 to 7.5 inches wide).
    Figures drawn wider than that are downsampled rather than re-laid out, so
    the TIFF matches the PNG exactly.
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    img = Image.open(buf).convert("RGB")
    max_px = round(max_width_in * dpi)
    if img.width > max_px:
        img = img.resize((max_px, round(img.height * max_px / img.width)), Image.LANCZOS)
    img.save(path, compression="tiff_lzw", dpi=(dpi, dpi))


def _save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    if TIFF_DIR is not None:
        _write_tiff(fig, TIFF_DIR / f"{stem}.tif")
    plt.close(fig)
    print(f"  wrote {stem}.png")


def fig1(by):
    arms = [("routine", "Unmonitored", GREY),
            ("valence_admit", "Risks of admitting", TEAL),
            ("valence_both", "Risks of both dispositions", SLATE),
            ("valence_discharge", "Risks of discharging", RED),
            ("mech_risk", "Documented reasoning + risks", DARKRED)]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(MODELS))
    w = 0.16
    for j, (arm, label, color) in enumerate(arms):
        vals, lo_err, hi_err = [], [], []
        for mk, _ in MODELS:
            r = admission_rate(by, arm, mk)
            vals.append(r["rate"])
            lo_err.append(r["rate"] - r["lo"])
            hi_err.append(r["hi"] - r["rate"])
        ax.bar(x + (j - (len(arms) - 1) / 2) * w, vals, w,
               yerr=[lo_err, hi_err], capsize=2, color=color, label=label,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in MODELS])
    ax.set_ylabel("Admission rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Naming which harm to document steers the disposition for identical patients",
                 fontsize=12, fontweight="bold")
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    _despine(ax)
    fig.tight_layout()
    _save(fig, "fig1")


def fig2(by):
    steps = [("Generate\nrationale", contrast(by, "routine", "mech_discard")["delta"], "#8fb0c9"),
             ("Record\n& audit", contrast(by, "mech_discard", "mech_record")["delta"], "#5b8db8"),
             ("Risk\ncontent", contrast(by, "mech_record", "mech_risk")["delta"], RED)]
    total = contrast(by, "routine", "mech_risk")["delta"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cum = 0.0
    for i, (label, val, color) in enumerate(steps):
        ax.bar(i, val, bottom=cum, color=color, edgecolor="white")
        ax.text(i, cum + val + 0.25, f"{val:+.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        cum += val
    ax.bar(len(steps), total, color=DARKRED, edgecolor="white")
    ax.text(len(steps), total + 0.25, f"{total:+.1f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(steps) + 1))
    ax.set_xticklabels([s[0] for s in steps] + ["Total\n(documented)"])
    ax.set_ylabel("Increase in admission rate vs baseline (pp)")
    ax.set_title("Anatomy of the explainability tax (pooled)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, total + 2)
    _despine(ax)
    fig.tight_layout()
    _save(fig, "fig2")


def fig3(by, gold):
    arms = [("routine", "Unmonitored", GREY),
            ("valence_admit", "Risks of admitting", TEAL),
            ("mech_record", "Recorded neutral note", "#8fb0c9"),
            ("valence_both", "Risks of both dispositions", SLATE),
            ("mech_risk", "Documented reasoning + risks", RED),
            ("valence_discharge", "Risks of discharging", DARKRED)]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharex=True, sharey=True)
    for ax, (mk, ml) in zip(axes, MODELS):
        pts = []
        for arm, label, color in arms:
            op = operating_point(by, gold, arm, mk)
            if op:
                pts.append((label, color, op["sens"], 100 - op["spec"],
                            admission_rate(by, arm, mk)["rate"]))
        traj = sorted(pts, key=lambda p: p[4])
        ax.plot([p[3] for p in traj], [p[2] for p in traj], "-", color="#bbbbbb", lw=1, zorder=1)
        for label, color, sens, fpr, _ in pts:
            ax.scatter(fpr, sens, s=70, color=color, edgecolor="white", zorder=3, label=label)
        ax.plot([0, 100], [0, 100], ":", color="#cccccc", lw=1)
        ax.set_title(ml, fontsize=11, fontweight="bold")
        ax.set_xlabel("1 \u2212 specificity\n(clinician discharges admitted)", fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        _despine(ax)
    axes[0].set_ylabel("Sensitivity\n(clinician admissions admitted)", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Justification framing shifts each model's operating point up the ROC space",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "fig3")


def fig4(by):
    d = [contrast(by, "mech_record", a)["delta"]
         for a in ("mech_risk1", "mech_risk3", "mech_risk5")]
    filler = contrast(by, "mech_record", "mech_filler")["delta"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot([0, 1, 3, 5], [0] + d, marker="o", color=RED, lw=2)
    ax.axhline(filler, ls=":", color=SLATE, lw=1.6)
    ax.text(5, filler + 0.2, f"length-matched irrelevant note ({filler:+.1f})",
            ha="right", va="bottom", fontsize=8.5, color="#4a4a4a")
    for x, y in zip((1, 3, 5), d):
        ax.text(x, y + 0.2, f"{y:+.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlabel("Number of risks required in the note")
    ax.set_ylabel("Change in admission rate vs neutral note (pp)")
    ax.set_title("The effect saturates: presence of risk, not dose", fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1, 3, 5])
    _despine(ax)
    fig.tight_layout()
    _save(fig, "fig4")


def fig5(primary, mimic):
    """Side-by-side replication of the directional valence contrast."""
    arms = [("valence_admit", "Risks of\nadmitting", TEAL),
            ("mech_record", "Recorded\nneutral note", "#8fb0c9"),
            ("valence_both", "Risks of both\ndispositions", SLATE),
            ("valence_discharge", "Risks of\ndischarging", DARKRED)]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)
    for ax, (by, name) in zip(axes, [(primary, "MC-MED (primary)"),
                                     (mimic, "MIMIC-IV-ED (replication)")]):
        vals, lo_err, hi_err, colors, labels = [], [], [], [], []
        for arm, label, color in arms:
            r = admission_rate(by, arm)
            if not r:
                continue
            vals.append(r["rate"])
            lo_err.append(r["rate"] - r["lo"])
            hi_err.append(r["hi"] - r["rate"])
            colors.append(color)
            labels.append(label)
        ax.bar(range(len(vals)), vals, 0.62, yerr=[lo_err, hi_err], capsize=3,
               color=colors, edgecolor="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 100)
        _despine(ax)
    axes[0].set_ylabel("Admission rate (%)")
    fig.suptitle("The direction of the shift replicates in an independent health system",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig5")


def main(argv=None):
    import argparse

    global TIFF_DIR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiff", type=Path, default=None,
                    help="also write 600 dpi LZW TIFFs (journal format) to this directory")
    args = ap.parse_args(argv)
    if args.tiff is not None:
        args.tiff.mkdir(parents=True, exist_ok=True)
        TIFF_DIR = args.tiff

    OUT.mkdir(parents=True, exist_ok=True)
    by, gold = load()
    fig1(by)
    fig2(by)
    # Fig 3 plots recommendations against the clinician disposition, which is
    # withheld from the public release. See cohort/export_labels.py.
    if any(gold.values()):
        fig3(by, gold)
    else:
        print("  skipped fig3: clinician labels unavailable; run "
              "python -m cohort.export_labels (requires credentialed data)")
    fig4(by)
    mimic, _ = load(dataset="mimic-iv-ed")
    if mimic:
        fig5(by, mimic)


if __name__ == "__main__":
    main()
