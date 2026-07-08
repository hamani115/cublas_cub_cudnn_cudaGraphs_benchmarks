#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def discover_csvs(paths):
    out = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out += list(pp.rglob("*.csv"))
        elif pp.is_file() and pp.suffix.lower() == ".csv":
            out.append(pp)
    return sorted(set(out), key=lambda x: str(x))


def infer_bench_from_path(csv_path: Path) -> str:
    s = str(csv_path).lower()
    for key in ("cudnn", "cublas", "cub"):
        if key in s:
            return key
    return "unknown"


def infer_mode_from_path(csv_path: Path):
    name = csv_path.name.lower()
    if "with_graph" in name or "with-graph" in name or "withgraphs" in name:
        return "Graph"
    if "without_graph" in name or "no_graph" in name or "without-graph" in name or "nograph" in name:
        return "NoGraph"
    return None


def clean_size(x):
    return str(x).strip() if not pd.isna(x) else x


def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def standardize_columns(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "cpu_idle_us_per_iter": "cpu_idle_us/iter",
        "launch_us_per_iter": "launch_us/iter",
        "total_us_per_iter": "total_us/iter",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    if "MODE" not in df.columns:
        m = infer_mode_from_path(csv_path)
        df["MODE"] = m if m is not None else "Unknown"

    if "Size" in df.columns:
        df["Size"] = df["Size"].map(clean_size)
    else:
        df["Size"] = "Unknown"

    df["Bench"] = infer_bench_from_path(csv_path)

    if "graph_create_ms" not in df.columns:
        df["graph_create_ms"] = 0.0

    ensure_numeric(df, ["iters", "launch_ms", "total_ms", "launch_us/iter", "total_us/iter",
                        "cpu_idle_us/iter", "graph_create_ms"])

    if "iters" in df.columns:
        it = df["iters"].replace(0, np.nan)

        if "launch_us/iter" not in df.columns or df["launch_us/iter"].isna().all():
            if "launch_ms" in df.columns:
                df["launch_us/iter"] = (df["launch_ms"] * 1000.0) / it

        if "total_us/iter" not in df.columns or df["total_us/iter"].isna().all():
            if "total_ms" in df.columns:
                df["total_us/iter"] = (df["total_ms"] * 1000.0) / it

        # Execution proxy = cpu_idle_us/iter = (total - launch)
        if "cpu_idle_us/iter" not in df.columns or df["cpu_idle_us/iter"].isna().all():
            if "total_us/iter" in df.columns and "launch_us/iter" in df.columns:
                df["cpu_idle_us/iter"] = df["total_us/iter"] - df["launch_us/iter"]

    return df


def safe_slug(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "", s)
    return s


def mode_label(mode: str) -> str:
    m = str(mode).strip().lower()
    if m == "graph":
        return "with graphs"
    if m == "nograph":
        return "without graphs"
    return str(mode)


def size_order_key(sz: str) -> int:
    s = str(sz).strip().lower()
    if "small" in s: return 0
    if "medium" in s: return 1
    if "big" in s: return 2
    return 99


def plot_metric_all_sizes_one_plot(df_agg, bench, metric, ylabel, title_prefix,
                                  gpu_label, outdir, annotate_last=True):
    bdf = df_agg[df_agg["Bench"] == bench].copy()
    if bdf.empty:
        return

    sizes = sorted(bdf["Size"].dropna().unique().tolist(), key=size_order_key)

    fig = plt.figure(figsize=(9.0, 5.0))
    ax = fig.add_subplot(1, 1, 1)

    for sz in sizes:
        sdf = bdf[bdf["Size"] == sz].copy()

        for mode in ["NoGraph", "Graph"]:
            mdf = sdf[sdf["MODE"] == mode].sort_values("iters")
            if mdf.empty:
                continue

            x = mdf["iters"].to_numpy(float)
            y = mdf[f"{metric}_mean"].to_numpy(float)
            yerr = mdf[f"{metric}_std"].to_numpy(float)

            label = f"{sz} | {mode_label(mode)}"
            ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", capsize=3, linewidth=1.5, label=label)

            if annotate_last and len(x) > 0:
                ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]),
                            textcoords="offset points", xytext=(6, 0),
                            ha="left", va="center", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Iterations (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"[{title_prefix}] {bench.upper()} on {gpu_label}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(title=gpu_label, loc="best", framealpha=0.95)

    outpath = Path(outdir) / bench / f"{bench}_{safe_slug(metric)}_all_sizes.png"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_metric_vs_size(df_agg, bench, metric, ylabel, title_prefix,
                        gpu_label, outdir):
    bdf = df_agg[df_agg["Bench"] == bench].copy()
    if bdf.empty:
        return

    g = bdf.groupby(["Size", "MODE"], dropna=False).agg(
        mean_value=(f"{metric}_mean", "mean"),
        std_value=(f"{metric}_mean", "std"),
    ).reset_index()

    g["Size"] = g["Size"].map(clean_size)
    sizes = sorted(g["Size"].dropna().unique().tolist(), key=size_order_key)
    x = np.arange(len(sizes))
    width = 0.35

    fig = plt.figure(figsize=(8.5, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    for offset, mode in [(-width/2, "NoGraph"), (width/2, "Graph")]:
        m = g[g["MODE"] == mode].set_index("Size").reindex(sizes)
        y = m["mean_value"].to_numpy(float)
        yerr = m["std_value"].fillna(0.0).to_numpy(float)

        ax.bar(x + offset, y, width, yerr=yerr, capsize=4, label=mode_label(mode))

    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Problem size")
    ax.set_ylabel(ylabel)
    ax.set_title(f"[{title_prefix}] {bench.upper()} on {gpu_label}")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(title=gpu_label, framealpha=0.95)

    outpath = Path(outdir) / bench / f"{bench}_{safe_slug(metric)}_vs_size.png"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_speedup_all_sizes(df_agg, bench, metric, ylabel, title_prefix,
                           gpu_label, outdir, annotate_last=True):
    """
    speedup = NoGraph_mean / Graph_mean, computed per (Size, iters)
    Plots 3 lines: Small/Medium/Big
    """
    bdf = df_agg[df_agg["Bench"] == bench].copy()
    if bdf.empty:
        return

    sizes = sorted(bdf["Size"].dropna().unique().tolist(), key=size_order_key)

    fig = plt.figure(figsize=(9.0, 5.0))
    ax = fig.add_subplot(1, 1, 1)

    for sz in sizes:
        sdf = bdf[bdf["Size"] == sz].copy()

        ng = sdf[sdf["MODE"] == "NoGraph"][["iters", f"{metric}_mean", f"{metric}_std"]].copy()
        gr = sdf[sdf["MODE"] == "Graph"][["iters", f"{metric}_mean", f"{metric}_std"]].copy()
        if ng.empty or gr.empty:
            continue

        merged = pd.merge(ng, gr, on="iters", suffixes=("_ng", "_g"))
        merged = merged.sort_values("iters")

        x = merged["iters"].to_numpy(float)
        y = (merged[f"{metric}_mean_ng"] / merged[f"{metric}_mean_g"]).to_numpy(float)

        eps = 1e-30
        rel_ng = (merged[f"{metric}_std_ng"] / (merged[f"{metric}_mean_ng"] + eps)) ** 2
        rel_g  = (merged[f"{metric}_std_g"]  / (merged[f"{metric}_mean_g"]  + eps)) ** 2
        yerr = (y * np.sqrt(rel_ng + rel_g)).to_numpy(float)

        ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="-", capsize=3, linewidth=1.5, label=str(sz))

        if annotate_last and len(x) > 0:
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]),
                        textcoords="offset points", xytext=(6, 0),
                        ha="left", va="center", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Iterations (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"[{title_prefix}] {bench.upper()} on {gpu_label}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(title="Size", loc="best", framealpha=0.95)

    outpath = Path(outdir) / bench / f"{bench}_speedup_{safe_slug(metric)}_all_sizes.png"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_graph_create_vs_size(df_agg, bench, gpu_label, outdir):
    """
    Graph creation usually does not depend on iters meaningfully, but you may have repeats.
    We aggregate across iters (mean of means) per Size for MODE=Graph.
    """
    bdf = df_agg[(df_agg["Bench"] == bench) & (df_agg["MODE"] == "Graph")].copy()
    if bdf.empty:
        return

    g = bdf.groupby("Size", dropna=False).agg(
        graph_create_ms_mean=("graph_create_ms_mean", "mean"),
        graph_create_ms_std=("graph_create_ms_mean", "std"),
    ).reset_index()

    g["Size"] = g["Size"].map(clean_size)
    g = g.sort_values("Size", key=lambda s: s.map(size_order_key))

    x = np.arange(len(g))
    y = g["graph_create_ms_mean"].to_numpy(float)
    yerr = g["graph_create_ms_std"].fillna(0.0).to_numpy(float)

    fig = plt.figure(figsize=(8.5, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    ax.errorbar(x, y, yerr=yerr, marker="o", linestyle="None", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(g["Size"].tolist())
    ax.set_xlabel("Size")
    ax.set_ylabel("Graph creation time (ms), mean ± σ")
    ax.set_title(f"[Graph Creation] {bench.upper()} on {gpu_label}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    outpath = Path(outdir) / bench / f"{bench}_graph_create_vs_size.png"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot CUDA Graphs benchmarks (all sizes on one plot, mean±std over repeats).")
    ap.add_argument("paths", nargs="+", help="CSV files or directories (directories scanned recursively for *.csv).")
    ap.add_argument("--outdir", default="plots", help="Output directory (default: plots).")
    ap.add_argument("--gpu-label", default="GPU", help='GPU label used in titles/legend title, e.g. "NGT H100 | CUDA 12.9".')
    ap.add_argument("--benches", default="cudnn,cublas,cub", help="Comma-separated benches to plot (default: cudnn,cublas,cub).")
    ap.add_argument("--no-annotate", action="store_true", help="Disable annotating last point values.")
    args = ap.parse_args()

    csvs = discover_csvs(args.paths)
    if not csvs:
        raise SystemExit("No CSV files found.")

    frames = []
    for p in csvs:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
            continue
        frames.append(standardize_columns(df, p))

    if not frames:
        raise SystemExit("No readable CSV data.")

    df_all = pd.concat(frames, ignore_index=True)

    benches = [b.strip().lower() for b in args.benches.split(",") if b.strip()]
    df_all = df_all[df_all["Bench"].isin(benches)]
    if df_all.empty:
        raise SystemExit("No rows matched requested benches.")

    group_keys = ["Bench", "MODE", "Size", "iters"]

    metric_cols = ["launch_us/iter", "cpu_idle_us/iter", "total_us/iter", "graph_create_ms"]
    for c in metric_cols:
        if c not in df_all.columns:
            df_all[c] = np.nan

    agg = df_all.groupby(group_keys, dropna=False).agg({
        "launch_us/iter": ["mean", "std", "count"],
        "cpu_idle_us/iter": ["mean", "std", "count"],
        "total_us/iter": ["mean", "std", "count"],
        "graph_create_ms": ["mean", "std", "count"],
    })
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    df_agg = agg.reset_index()

    for c in metric_cols:
        stdc = f"{c}_std"
        if stdc in df_agg.columns:
            df_agg[stdc] = df_agg[stdc].fillna(0.0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "aggregated_mean_std.csv"
    df_agg.to_csv(out_csv, index=False)
    print(f"[OK] aggregated: {out_csv}")

    annotate = (not args.no_annotate)

    # Generate plots per bench
    for bench in benches:

        plot_metric_all_sizes_one_plot(
            df_agg, bench,
            metric="launch_us/iter",
            ylabel="Launch time per iteration (µs), mean ± σ",
            title_prefix="Launch Time / Iteration",
            gpu_label=args.gpu_label,
            outdir=args.outdir,
            annotate_last=annotate,
        )
        
        plot_metric_vs_size(
            df_agg, bench,
            metric="total_us/iter",
            ylabel="Total execution time per iteration (µs), mean ± σ",
            title_prefix="Total Execution Time vs Size",
            gpu_label=args.gpu_label,
            outdir=args.outdir,
        )

        plot_metric_all_sizes_one_plot(
            df_agg, bench,
            metric="cpu_idle_us/iter",
            ylabel="CPU idle time per iteration (µs), mean ± σ",
            title_prefix="CPU Idle Time / Iteration",
            gpu_label=args.gpu_label,
            outdir=args.outdir,
            annotate_last=annotate,
        )
        
        plot_metric_all_sizes_one_plot(
            df_agg, bench,
            metric="total_us/iter",
            ylabel="Total execution time per iteration (µs), mean ± σ",
            title_prefix="Total Execution Time / Iteration",
            gpu_label=args.gpu_label,
            outdir=args.outdir,
            annotate_last=annotate,
        )

        plot_speedup_all_sizes(
            df_agg, bench,
            metric="launch_us/iter",
            ylabel="Speedup (without graphs / with graphs), mean ± σ",
            title_prefix="Speedup — Launch Time / Iteration",
            gpu_label=args.gpu_label,
            outdir=args.outdir,
            annotate_last=annotate,
        )
        plot_speedup_all_sizes(
            df_agg, bench,
            metric="cpu_idle_us/iter",
            ylabel="Speedup (without graphs / with graphs), mean ± σ",
            title_prefix="Speedup — CPU Idle Time / Iteration",
            gpu_label=args.gpu_label,
            outdir=args.outdir,
            annotate_last=annotate,
        )
        
        plot_speedup_all_sizes(
            df_agg, bench,
            metric="total_us/iter",
            ylabel="Speedup (without graphs / with graphs), mean ± σ",
            title_prefix="Speedup — Total Execution Time / Iteration",
            gpu_label=args.gpu_label,
            outdir=args.outdir,
            annotate_last=annotate,
        )

        plot_graph_create_vs_size(df_agg, bench, args.gpu_label, args.outdir)

    print(f"[OK] plots saved under: {Path(args.outdir).resolve()}/<bench>/")
    print("     (Each bench folder has: launch/execution, speedups, graph creation.)")


if __name__ == "__main__":
    main()