#!/usr/bin/env python3
"""Merge A+B results (results/secondorder/) with C+D results (results/secondorder_cd/)
then re-run significance analysis on the combined data."""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import subprocess

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / "results" / "secondorder"
CD = ROOT / "results" / "secondorder_cd"
MERGED = ROOT / "results" / "secondorder_merged"


def main():
    MERGED.mkdir(parents=True, exist_ok=True)

    for name in ("secondorder_results.csv", "secondorder_predictions.csv"):
        ab_path = AB / name
        cd_path = CD / name
        if not ab_path.exists():
            sys.exit(f"Missing {ab_path}")
        if not cd_path.exists():
            sys.exit(f"Missing {cd_path}")
        df = pd.concat([pd.read_csv(ab_path), pd.read_csv(cd_path)], ignore_index=True)
        out = MERGED / name
        df.to_csv(out, index=False)
        print(f"Merged {name}: {len(df)} rows -> {out}")

    merged_results = pd.read_csv(MERGED / "secondorder_results.csv")
    print("\nArm counts:")
    print(merged_results.groupby("arm")["seed"].count().to_string())

    print("\nRunning significance analysis on merged data...")
    subprocess.run([
        sys.executable,
        str(ROOT / "scripts" / "analyze_secondorder_significance.py"),
        "--results-dir", str(MERGED),
    ], check=True)


if __name__ == "__main__":
    main()
