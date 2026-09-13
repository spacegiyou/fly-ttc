"""Run the fixed v0 expansion pipeline."""
import sys
from fly_ttc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", "--model", "v0", *sys.argv[1:]]))
