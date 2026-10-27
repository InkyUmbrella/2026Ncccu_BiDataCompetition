"""Freeze the weights, preprocessing, source and precision used by predict/cost."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glaucoma.inference import freeze_spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--tta", choices=("none", "hflip"), default="none")
    args = parser.parse_args(argv)
    freeze_spec(args.checkpoint, args.output, args.precision, args.tta)
    print(f"Frozen inference spec: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
