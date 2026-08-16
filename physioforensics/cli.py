"""Command line interface.

    physioforensics selftest                      validate the signal layer
    physioforensics synth      --out data/synth   render a labelled corpus
    physioforensics features   --root data/synth  video -> feature table
    physioforensics evaluate   --table feats.csv  run both protocols
    physioforensics importance --table feats.csv  rank the physical features
    physioforensics train      --table feats.csv  fit and persist a model
    physioforensics analyze    video.mp4          explain one video
    physioforensics waveforms  --out plot.png     forehead-vs-neck comparison plot
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _log(verbose: bool):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )


# --------------------------------------------------------------------------

def cmd_selftest(args) -> int:
    """Confirm the extractor recovers a known heart rate from known physiology.

    Needs no dataset or network, so the signal chain can be trusted before
    any real corpus arrives.
    """
    from .features import compute_features
    from .regions import RegionTraces
    from . import synth

    print(f"Synthetic validation  (rPPG method: {args.method})")
    print("-" * 78)
    print(f"{'class':<12}{'true bpm':>9}{'est bpm':>9}{'err':>7}"
          f"{'snr dB':>8}{'plv face':>10}{'plv neck':>10}{'spread':>8}")
    print("-" * 78)

    failures, rows = [], []
    for kind in synth.CLASSES:
        for seed in range(args.repeats):
            hr = 60.0 + 8.0 * seed
            traces, truth = synth.synth_region_traces(
                kind=kind, hr_bpm=hr, duration_sec=args.duration, seed=1000 + seed)
            rt = RegionTraces(traces=traces, fps=30.0, n_frames=truth["n_frames"],
                              source=f"synthetic:{kind}", roi_mode="fixed",
                              face_detected=False)
            f = compute_features(rt, method=args.method)
            est = f.get("hr_consensus_bpm", float("nan"))
            err = abs(est - hr) if np.isfinite(est) else float("nan")
            rows.append({"kind": kind, "true_hr": hr, "est_hr": est, "err": err,
                         "snr": f.get("snr_mean"),
                         "plv_face": f.get("plv_within_face_mean"),
                         "plv_neck": f.get("plv_face_neck_mean"),
                         "spread": f.get("hr_spread_bpm")})
            if seed == 0:
                print(f"{kind:<12}{hr:>9.1f}{est:>9.1f}{err:>7.1f}"
                      f"{f.get('snr_mean', float('nan')):>8.2f}"
                      f"{f.get('plv_within_face_mean', float('nan')):>10.3f}"
                      f"{f.get('plv_face_neck_mean', float('nan')):>10.3f}"
                      f"{f.get('hr_spread_bpm', float('nan')):>8.1f}")

    df = pd.DataFrame(rows)
    print("-" * 78)

    # heart rate must be recovered on physiologically valid video
    real_err = df[df.kind == "real"].err
    hr_ok = real_err.max() <= args.hr_tolerance
    print(f"[{'PASS' if hr_ok else 'FAIL'}] heart rate recovered on 'real' "
          f"within {args.hr_tolerance:.1f} bpm (worst {real_err.max():.2f})")
    if not hr_ok:
        failures.append("hr_recovery")

    # coherence must separate real from every failure mode
    real_plv = df[df.kind == "real"].plv_face.mean()
    for kind in [k for k in df.kind.unique() if k != "real"]:
        other = df[df.kind == kind]
        plv = other.plv_face.mean() if kind != "swap" else other.plv_neck.mean()
        ok = plv < real_plv - 0.2
        label = "plv_face" if kind != "swap" else "plv_neck"
        print(f"[{'PASS' if ok else 'FAIL'}] '{kind}' separated by {label}: "
              f"{plv:.3f} vs real {real_plv:.3f}")
        if not ok:
            failures.append(f"separation:{kind}")

    # coherence must catch cases per-region signal quality alone waves through
    inc = df[df.kind == "incoherent"]
    real = df[df.kind == "real"]
    snr_fails = inc.snr.mean() >= real.snr.mean()
    plv_works = inc.plv_face.mean() < real.plv_face.mean() - 0.2
    if snr_fails and plv_works:
        print("[NOTE] 'incoherent' has signal quality at or above genuine video "
              "yet is still caught by phase coherence -- exactly the case a "
              "quality-only detector misses.")

    print("-" * 78)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


def cmd_synth(args) -> int:
    from . import synth

    manifest = synth.build_synthetic_dataset(
        args.out, n_per_class=args.n, duration_sec=args.duration,
        fps=args.fps, seed=args.seed)
    pd.DataFrame(manifest).to_csv(Path(args.out) / "manifest.csv", index=False)
    print(f"Wrote {len(manifest)} videos to {args.out}")
    for kind in sorted({m['generator'] for m in manifest}):
        print(f"  {kind:<12} {sum(m['generator'] == kind for m in manifest)}")
    return 0


def cmd_features(args) -> int:
    from .dataset import build_feature_table

    df = build_feature_table(args.root, method=args.method, roi_mode=args.roi,
                             max_frames=args.max_frames, workers=args.workers)
    df.to_csv(args.out, index=False)
    ok = int((df.get("error", "").fillna("") == "").sum()) if "error" in df else len(df)
    print(f"Wrote {args.out}  ({len(df)} videos, {ok} succeeded, "
          f"{df.shape[1]} columns)")
    print(df.groupby("generator").size().to_string())
    return 0


def cmd_evaluate(args) -> int:
    from .train import run_full_evaluation

    df = pd.read_csv(args.table)
    results = run_full_evaluation(df, folds=args.folds, seed=args.seed)
    if results.empty:
        print("No evaluation could be run (need >=2 generators and both classes).")
        return 1
    results.to_csv(args.out, index=False)

    print("\n=== Random split (every generator seen in training) ===")
    rnd = results[results.protocol == "random"]
    print(rnd.pivot_table(index="family", columns="model", values="auc").round(3).to_string())

    print("\n=== Leave-one-generator-out (held-out generator unseen) ===")
    logo = results[(results.protocol == "logo")]
    print(logo.pivot_table(index="group", columns=["model", "family"],
                           values="auc").round(3).to_string())

    mean_rows = logo[logo.group == "MEAN"]
    if not mean_rows.empty:
        print("\n=== Contribution of cross-region coherence (LOGO mean AUC) ===")
        piv = mean_rows.pivot_table(index="model", columns="family", values="auc")
        if {"quality", "all"}.issubset(piv.columns):
            piv["delta_all_minus_quality"] = piv["all"] - piv["quality"]
        print(piv.round(3).to_string())

    print(f"\nWrote {args.out}")
    return 0


def cmd_importance(args) -> int:
    from .train import permutation_importance_report

    df = pd.read_csv(args.table)
    rep = permutation_importance_report(df, family=args.family, seed=args.seed)
    rep.to_csv(args.out, index=False)
    print(rep.head(args.top).round(4).to_string(index=False))
    print(f"\nWrote {args.out}")
    return 0


def cmd_train(args) -> int:
    from .train import train_and_save

    df = pd.read_csv(args.table)
    meta = train_and_save(df, args.out, family=args.family, model_kind=args.model,
                          seed=args.seed)
    print(f"Trained on {meta['n_train']} videos, {meta['n_features']} features "
          f"({meta['family']}) -> {meta['path']}")
    return 0


def cmd_analyze(args) -> int:
    from .features import compute_features, explain
    from .regions import extract_region_traces

    rt = extract_region_traces(args.video, roi_mode=args.roi, max_frames=args.max_frames)
    feats = compute_features(rt, method=args.method)

    print(f"\n{Path(args.video).name}")
    print("=" * 62)
    print(f"frames {rt.n_frames} @ {rt.fps:.1f} fps  "
          f"({rt.n_frames / rt.fps:.1f}s), face detected: {rt.face_detected}")
    print(f"regions with usable signal: {', '.join(rt.available()) or 'none'}")
    print("-" * 62)
    for line in explain(feats):
        print(f"  {line}")
    print("-" * 62)

    score = feats.get("coherence_score", float("nan"))
    if args.model:
        import joblib
        bundle = joblib.load(args.model)
        x = np.array([[feats.get(c, np.nan) for c in bundle["columns"]]], dtype=float)
        p = float(bundle["model"].predict_proba(x)[0, 1])
        print(f"  Model probability of forgery: {p:.3f}  "
              f"-> {'FAKE' if p > 0.5 else 'REAL'}")
    elif np.isfinite(score):
        print(f"  Physiological coherence score: {score:.3f} "
              f"(1.0 = fully consistent body; no classifier loaded)")
    else:
        print("  Insufficient signal for a physiological verdict.")

    if args.json:
        import json
        Path(args.json).write_text(json.dumps(feats, indent=2, default=float))
        print(f"\nWrote {args.json}")
    return 0


def cmd_waveforms(args) -> int:
    from .plotting import plot_pulse_comparison

    path = plot_pulse_comparison(
        args.out, duration_sec=args.duration, hr_bpm=args.hr_bpm, seed=args.seed)
    print(f"Wrote {path}")
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="physioforensics",
        description="Generator-agnostic deepfake detection from cross-region physiology.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("selftest", help="validate the signal layer on known physiology")
    s.add_argument("--method", default="pos", choices=["pos", "chrom", "green"])
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--duration", type=float, default=12.0)
    s.add_argument("--hr-tolerance", type=float, default=3.0)
    s.set_defaults(func=cmd_selftest)

    s = sub.add_parser("synth", help="render a labelled synthetic corpus")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("-n", type=int, default=12, help="videos per class")
    s.add_argument("--duration", type=float, default=10.0)
    s.add_argument("--fps", type=float, default=30.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("features", help="extract a feature table from a dataset root")
    s.add_argument("--root", required=True)
    s.add_argument("--out", default="features.csv")
    s.add_argument("--method", default="pos", choices=["pos", "chrom", "green"])
    s.add_argument("--roi", default="auto", choices=["auto", "fixed"])
    s.add_argument("--max-frames", type=int, default=None)
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(func=cmd_features)

    s = sub.add_parser("evaluate", help="random and leave-one-generator-out protocols")
    s.add_argument("--table", required=True)
    s.add_argument("--out", default="results.csv")
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("importance", help="rank features by permutation importance")
    s.add_argument("--table", required=True)
    s.add_argument("--out", default="importance.csv")
    s.add_argument("--family", default="all", choices=["quality", "coherence", "all"])
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_importance)

    s = sub.add_parser("train", help="fit a model on the whole table and save it")
    s.add_argument("--table", required=True)
    s.add_argument("--out", default="models/physio.joblib")
    s.add_argument("--family", default="all", choices=["quality", "coherence", "all"])
    s.add_argument("--model", default="gb", choices=["gb", "logreg"])
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("analyze", help="explain the physiology of a single video")
    s.add_argument("video")
    s.add_argument("--model", default=None, help="optional trained .joblib")
    s.add_argument("--method", default="pos", choices=["pos", "chrom", "green"])
    s.add_argument("--roi", default="auto", choices=["auto", "fixed"])
    s.add_argument("--max-frames", type=int, default=None)
    s.add_argument("--json", default=None, help="also dump features to this path")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("waveforms", help="render a forehead-vs-neck pulse comparison plot")
    s.add_argument("--out", default="docs/waveform_comparison.png")
    s.add_argument("--duration", type=float, default=12.0)
    s.add_argument("--hr-bpm", type=float, default=68.0)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_waveforms)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _log(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
