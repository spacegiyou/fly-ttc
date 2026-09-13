"""Build a normalized train manifest from manually placed official Nexar files."""
import argparse
from fly_ttc.data.nexar import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Kaggle train.csv, normalized CSV, or HF metadata root directory")
    parser.add_argument("--raw-dir", default="data/raw/nexar")
    parser.add_argument("--output", default="data/manifests/subset_100_seed0.csv")
    parser.add_argument("--subset", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(build_manifest(args.source, args.raw_dir, args.output, subset=args.subset, seed=args.seed))


if __name__ == "__main__":
    main()
