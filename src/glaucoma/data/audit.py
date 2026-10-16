"""Full image decoding and optional byte-identical duplicate detection."""

from collections import Counter
import hashlib

from .dataset import load_rgb


def audit_samples(splits, hashes=False):
    report = {"splits": {}, "errors": [], "duplicate_groups": [], "hashes_checked": hashes}
    digest_groups = {}
    for split, samples in splits.items():
        sizes, labels = Counter(), Counter()
        for index, sample in enumerate(samples, 1):
            try:
                image = load_rgb(sample.path)
                sizes[f"{image.width}x{image.height}"] += 1
                image.close()
                if hashes:
                    digest = hashlib.sha256()
                    with sample.path.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    digest_groups.setdefault(digest.hexdigest(), []).append({"split": split, "name": sample.name})
            except Exception as exc:
                report["errors"].append({"split": split, "name": sample.name, "error": str(exc)})
            if sample.label is not None:
                labels[str(sample.label)] += 1
            if index % 1000 == 0:
                print(f"[audit] {split}: {index}/{len(samples)}", flush=True)
        report["splits"][split] = {"count": len(samples), "labels": dict(labels), "sizes": dict(sizes)}
    report["duplicate_groups"] = [group for group in digest_groups.values() if len(group) > 1]
    return report
