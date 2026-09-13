#!/usr/bin/env python3
"""Generate a report from an existing v0 run."""

import argparse

from fly_ttc.viz.report import make_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/v0")
    args = parser.parse_args()
    print(make_report(args.run))


if __name__ == "__main__":
    main()
