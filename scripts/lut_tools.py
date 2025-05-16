"""
Author: Fraser King
Date: May 16, 2025

lut_tools.py - Minimal utility module for interacting with the precipitation phase
lookup-table (King et al., 2025)

How to use:
    >>> from lut_tools import load_lookup_table, query_lookup, plot_lut_map
    >>> lut = load_lookup_table()
    >>> probs = query_lookup(lut, T=-2.0, Nt_log=3.0)
    >>> print(probs.sort_values(ascending=False).head())
    >>> plot_lut_map(lut, "lut_map.png")

The module can also be executed as a lightweight CLI:
    $ python lut_tools.py plot lut_map.png
    $ python lut_tools.py query -2.0 3.0

Dependencies
------------
python >= 3.9, pandas, numpy, pyarrow, matplotlib
Install via:
    pip install pandas numpy pyarrow matplotlib
"""


### Imports
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


### Public Constants
TEMP_COL: str = "t"
NT_LOG_COL: str = "Nt_log"
CLUSTER_COL: str = "hdb_cluster"
T_EDGES: np.ndarray = np.arange(-40.0, 32.5, 0.5)
LOGNT_EDGES: np.ndarray = np.arange(1.5, 6.8, 0.1)
BINS: Dict[str, np.ndarray] = {TEMP_COL: T_EDGES, NT_LOG_COL: LOGNT_EDGES}
COLORS: Tuple[str, ...] = (
    "black", "#3CB44B", "#4363D8", "#BFEF45", "#42D4F4",
    "#911EB4", "#FFE119", "#E6194B", "#F032E6", "#F58231",
)

CLUSTER_NAMES = {
    0: "Ambiguous",
    1: "Heavy Rain to Mixed-Phase",
    2: "Heavy Rain",
    3: "Light Rain to Mixed-Phase",
    4: "Drizzle",
    5: "Heavy Mixed-Phase",
    6: "Heavy Snow to Mixed-Phase",
    7: "Heavy Snow",
    8: "Light Mixed-Phase",
    9: "Light Snow",
}

__all__ = [
    "load_lookup_table",
    "query_lookup",
    "plot_lut_map",
    "BINS",
    "CLUSTER_NAMES",
]


### Helper funcs
def _ensure_cat(lut: pd.DataFrame) -> pd.DataFrame:
    for col in ("bin_T", "bin_Nt_log"):
        if not pd.api.types.is_categorical_dtype(lut[col]):
            lut[col] = pd.Categorical(lut[col])
    return lut
def _centre(edges: np.ndarray) -> np.ndarray:
    return (edges[:-1] + edges[1:]) / 2.0


### func: load_lookup_table(path)
### Args: path - where the parquet table is stored on disk
### Returns: lookup table dataframe
def load_lookup_table(path: str | Path = "umap_cluster_prior.parquet") -> pd.DataFrame:
    lut = pd.read_parquet(path)
    return _ensure_cat(lut)


### func: query_lookup(lut, T, Nt_log)
### Args: lut - (the pre-generated lookup table)
###       T - Temperature (deg Celsuis)
###       Nt_log - Total Particle Count (log-scaled)
### Returns: a Series of P(cluster | T, Nt_log)
def query_lookup(lut: pd.DataFrame, T: float, Nt_log: float, bins: dict = BINS, cluster_col: str = CLUSTER_COL) -> pd.Series:
    lut = _ensure_cat(lut)
    t_centres = _centre(bins[TEMP_COL])
    nt_centres = _centre(bins[NT_LOG_COL])

    # Find the nominal bin
    t_idx = np.clip(np.digitize(T, bins[TEMP_COL]) - 1, 0, len(t_centres)-1)
    nt_idx = np.clip(np.digitize(Nt_log, bins[NT_LOG_COL]) - 1, 0, len(nt_centres)-1)
    bin_T = pd.Interval(bins[TEMP_COL][t_idx], bins[TEMP_COL][t_idx+1], closed="left")
    bin_Nt_log = pd.Interval(bins[NT_LOG_COL][nt_idx], bins[NT_LOG_COL][nt_idx+1],closed="left")
    subset = lut[(lut["bin_T"] == bin_T) & (lut["bin_Nt_log"] == bin_Nt_log)]

    # Case where an exact bin for this combination of T, Nt doesn't exist
    if subset.empty:
        lut_codes_T = lut["bin_T"].cat.codes.to_numpy()
        lut_codes_Nt = lut["bin_Nt_log"].cat.codes.to_numpy()
        lut_T_centres = t_centres[lut_codes_T]
        lut_Nt_centres = nt_centres[lut_codes_Nt]

        # Perform squared Euclidean distance calc
        dist2 = (lut_T_centres - T)**2 + (lut_Nt_centres - Nt_log)**2
        nearest_idx = np.argmin(dist2)

        # Get the nearest bins
        nearest_T_bin = lut.iloc[nearest_idx]["bin_T"]
        nearest_Nt_log_bin = lut.iloc[nearest_idx]["bin_Nt_log"]
        subset = lut[(lut["bin_T"] == nearest_T_bin) & (lut["bin_Nt_log"] == nearest_Nt_log_bin)]

    # Return a series of P(cluster | T, Nt_log)
    return subset.set_index(cluster_col)["p"]


### func: plot_lut_map(lut)
### Args: lut - (the pre-generated lookup table)
### Args: out_path - saves lookup table plot to this path on disk
### Returns: N/A (plots to output)
def plot_lut_map(lut, out_path, bins=BINS, cluster_names=CLUSTER_NAMES) -> None:
    nT, nNt = len(bins[TEMP_COL]) - 1, len(bins[NT_LOG_COL]) - 1
    rgb_palette = np.array([mcolors.to_rgb(c) for c in COLORS[:10]])
    rgba = np.zeros((nT, nNt, 4))

    for (iT, jNt), g in lut.groupby([lut["bin_T"].cat.codes, lut["bin_Nt_log"].cat.codes]):
        probs = g.groupby(CLUSTER_COL)["p"].sum()
        dom = probs.idxmax()
        p_max = probs.max()
        rgba[iT, jNt, :3] = rgb_palette[dom]
        rgba[iT, jNt, 3] = p_max

    img = rgba.transpose(1, 0, 2)

    # Make figure
    plt.rcParams.update({'font.size': 20})
    _, ax = plt.subplots(figsize=(18, 12))
    ax.imshow(img, origin='lower',
              extent=[bins[TEMP_COL][0],  bins[TEMP_COL][-1],
                      bins[NT_LOG_COL][0], bins[NT_LOG_COL][-1]],
              aspect='auto')
    ax.set_xlabel("Surface temperature (deg C)", fontsize=20)
    ax.set_ylabel("log$_{10}$ Nt (total particle counts)", fontsize=20)
    ax.set_title("Cluster Lookup Table Mapping", fontsize=24)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in range(10)]
    labels  = [f"{k}: {cluster_names[k]}" for k in range(10)]
    ax.legend(handles, labels,
              loc="upper center",
              bbox_to_anchor=(0.5, -0.13),
              ncol=4,
              fontsize=16, frameon=True)
    
    plt.tight_layout()
    plt.savefig(out_path, DPI=300, transparent=True)
    plt.close()


### Simple command line interface for interacting with this API
def _cli() -> None:
    p = argparse.ArgumentParser(description="Quick CLI around lut_tools functions")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("plot", help="Save a PNG visualising the LUT")
    pp.add_argument("outfile", help="Output image path (e.g. lut.png)")
    pp.add_argument("--lut", default="umap_cluster_prior.parquet", help="Path to LUT parquet file")

    pq = sub.add_parser("query", help="Print cluster probabilities at (T, Nt_log)")
    pq.add_argument("T", type=float, help="Temperature in °C")
    pq.add_argument("Nt_log", type=float, help="log₁₀ particle count Nt")
    pq.add_argument("--lut", default="umap_cluster_prior.parquet", help="Path to LUT parquet file")

    args = p.parse_args()
    lut = load_lookup_table(args.lut)
    if args.cmd == "plot":
        plot_lut_map(lut, args.outfile)
    elif args.cmd == "query":
        probs = query_lookup(lut, args.T, args.Nt_log)
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(probs.to_frame("p"))
    else:
        p.error("Unknown command")

        
if __name__ == "__main__":
    try:
        _cli()
    except KeyboardInterrupt:
        sys.exit(130)