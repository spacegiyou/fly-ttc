"""Explicitly reject v1 until the v0 report has been reviewed."""
import sys
from fly_ttc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", "--model", "v1", *sys.argv[1:]]))
