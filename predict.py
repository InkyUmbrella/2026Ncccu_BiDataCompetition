"""Generate submission.csv with the same batch-one function used by cost.py."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from glaucoma.data.dataset import read_samples
from glaucoma.inference import predict_samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="inference_spec.json")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    report = predict_samples(args.spec, read_samples(args.data_root, "test", False), args.output, args.device)
    print(f"Wrote {report['count']} predictions: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
