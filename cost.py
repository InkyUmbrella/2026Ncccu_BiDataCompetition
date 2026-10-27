"""Usage: python cost.py submission.csv pictures_100 pictures_5 [--output-dir NEW_DIR]."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# run_cost calls glaucoma.inference's identical loader/per-image writer, as does predict.py.
from glaucoma.benchmark import run_cost


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission")
    parser.add_argument("pictures_100")
    parser.add_argument("pictures_5")
    parser.add_argument("--spec", default="inference_spec.json")
    parser.add_argument("--output-dir", help="New report directory; defaults to a unique outputs/cost run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ages-csv", help="Required only for an age-dependent model")
    parser.add_argument("--diagnostic", action="store_true", help="Allow non-T4/software engineering checks")
    args = parser.parse_args(argv)
    output_dir = args.output_dir or ("outputs/cost/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                    + "_" + uuid.uuid4().hex[:8])
    report = run_cost(args.spec, args.submission, args.pictures_100, args.pictures_5,
                      output_dir, args.device, args.ages_csv, args.diagnostic)
    print(f"Cost report: {Path(output_dir).resolve()}")
    return 0 if report["within_latency_limit"] and report["predictions_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
