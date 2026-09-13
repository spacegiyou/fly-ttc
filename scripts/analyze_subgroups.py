#!/usr/bin/env python3
"""Audit frozen v0 subgroups without changing scores, threshold, or split."""

import argparse
from pathlib import Path

from fly_ttc.eval.subgroups import analyze_subgroups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/v0")
    parser.add_argument("--out", default="outputs/v0_review")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    analyze_subgroups(args.run, args.out, args.seed, args.bootstrap_iterations)
    print(Path(args.out) / "REPORT.md")


if __name__ == "__main__":
    main()
