"""Fusion, paired bootstrap, report scanner and fuse-script tests (CPU only)."""

import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

from glaucoma.fusion import (LogitStandardizer, align_by_name, apply_standardizer,
                             average_ranks, fit_standardizer, fuse, group_same_age_pairs,
                             paired_bootstrap, paired_bootstrap_auroc, rank_scores,
                             ranking_metrics, sigmoid, standardise_logits)


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_tables(rows=120, positives=24, seed=0):
    rng = np.random.default_rng(seed)
    names = [f"img_{index:04d}" for index in range(rows)]
    targets = np.zeros(rows, dtype=np.int64)
    targets[:positives] = 1
    rng.shuffle(targets)
    ref = rng.normal(size=rows)
    new = ref + 0.8 * targets + rng.normal(scale=0.05, size=rows)
    return names, targets, ref, new


def write_predictions(path, names, targets, scores):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Image_name", "True_Label", "Logit", "Probability"])
        for name, target, score in zip(names, targets, scores):
            writer.writerow([name, int(target), format(float(score), ".10g"),
                             format(float(sigmoid(score)), ".10g")])


class RankTests(unittest.TestCase):
    def test_average_ranks_and_ties(self):
        self.assertEqual(average_ranks([0.1, 0.2, 0.2, 0.4]).tolist(), [1.0, 2.5, 2.5, 4.0])
        self.assertEqual(average_ranks([3.0, 1.0, 2.0]).tolist(), [3.0, 1.0, 2.0])
        scores = rank_scores([0.1, 0.2, 0.2, 0.4])
        self.assertTrue(((scores > 0) & (scores < 1)).all())
        self.assertTrue((np.diff(scores) >= 0).all())

    def test_rejects_empty_and_nonfinite(self):
        with self.assertRaises(ValueError):
            average_ranks([])
        with self.assertRaises(ValueError):
            average_ranks([1.0, float("nan")])


class StandardizerTests(unittest.TestCase):
    def test_round_trip_and_reuse(self):
        values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        mean, std = fit_standardizer(values)
        np.testing.assert_allclose(apply_standardizer(values, mean, std),
                                   (values - values.mean()) / values.std(), atol=0, rtol=0)
        standardizer = LogitStandardizer.fit(values)
        z, same = standardise_logits(values, standardizer)
        np.testing.assert_allclose(z, (values - mean) / std, atol=1e-12)
        self.assertEqual(same, standardizer)
        self.assertEqual(LogitStandardizer.from_dict(standardizer.to_dict()), standardizer)

    def test_constant_sample_and_bad_dict_raise(self):
        with self.assertRaises(ValueError):
            fit_standardizer([0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            LogitStandardizer.from_dict({"mean": 0.0, "std": 0.0})


class FuseTests(unittest.TestCase):
    def test_logit_equal_weights_and_reuse(self):
        names, targets, ref, new = synthetic_tables()
        tables = [(names, ref, targets), (names, new, targets)]
        fused = fuse(tables, method="logit")
        expected = (LogitStandardizer.fit(ref).apply(ref)
                    + LogitStandardizer.fit(new).apply(new)) / 2.0
        np.testing.assert_allclose(fused, expected, atol=1e-12)
        standardizers = [LogitStandardizer.fit(ref), LogitStandardizer.fit(new)]
        fuse(tables, method="logit", standardizers=standardizers)

    def test_weights_are_normalised(self):
        names, targets, ref, new = synthetic_tables()
        tables = [(names, ref, targets), (names, new, targets)]
        fused = fuse(tables, weights=[1.0, 3.0], method="logit")
        expected = (LogitStandardizer.fit(ref).apply(ref) * 0.25
                    + LogitStandardizer.fit(new).apply(new) * 0.75)
        np.testing.assert_allclose(fused, expected, atol=1e-12)

    def test_prob_and_rank_methods(self):
        names, targets, ref, new = synthetic_tables()
        tables = [(names, ref, targets), (names, new, targets)]
        prob = fuse(tables, method="prob")
        np.testing.assert_allclose(prob, (sigmoid(ref) + sigmoid(new)) / 2.0, atol=1e-12)
        rank = fuse(tables, method="rank")
        expected = (rank_scores(ref) + rank_scores(new)) / 2.0
        np.testing.assert_allclose(rank, expected, atol=1e-12)
        self.assertTrue(((rank > 0) & (rank < 1)).all())

    def test_single_model_is_identity_for_logit(self):
        names, targets, ref, _ = synthetic_tables()
        fused = fuse([(names, ref, targets)], method="logit")
        np.testing.assert_allclose(fused, LogitStandardizer.fit(ref).apply(ref), atol=1e-12)

    def test_validation_errors(self):
        names, targets, ref, new = synthetic_tables()
        with self.assertRaises(ValueError):
            fuse([(names, ref, targets), (names, new, targets)], method="nope")
        reordered = list(reversed(names))
        index = [names.index(name) for name in reordered]
        with self.assertRaisesRegex(ValueError, "order differs"):
            fuse([(names, ref, targets), (reordered, new[index], targets[index])])
        with self.assertRaisesRegex(ValueError, "name set differs"):
            fuse([(names, ref, targets), (names[:-1], new[:-1], targets[:-1])])
        with self.assertRaisesRegex(ValueError, "targets differ"):
            flipped = targets.copy()
            flipped[0] = 1 - flipped[0]
            fuse([(names, ref, targets), (names, new, flipped)])
        with self.assertRaises(ValueError):
            fuse([(names, ref, targets), (names, new, targets)], weights=[1.0])
        with self.assertRaises(ValueError):
            fuse([(names, ref, targets), (names, new, targets)], weights=[0.0, 0.0])
        with self.assertRaises(ValueError):
            fuse([(names, ref, targets), (names, new, targets)], weights=[1.0, -1.0])

    def test_input_kind_probabilities(self):
        names, targets, ref, new = synthetic_tables()
        tables = [(names, sigmoid(ref), targets), (names, sigmoid(new), targets)]
        fused = fuse(tables, method="logit", input_kind="probabilities")
        expected = fuse([(names, ref, targets), (names, new, targets)], method="logit")
        np.testing.assert_allclose(fused, expected, atol=1e-10)


class AlignTests(unittest.TestCase):
    def test_reorder_and_errors(self):
        names = ["a", "b", "c"]
        first = {"names": names, "logits": [1.0, 2.0, 3.0]}
        second = {"names": ["c", "a", "b"], "logits": [30.0, 10.0, 20.0]}
        aligned = align_by_name([first, second])
        self.assertEqual(aligned[0]["names"], names)
        self.assertEqual(aligned[1]["logits"], [10.0, 20.0, 30.0])
        with self.assertRaisesRegex(ValueError, "name set differs"):
            align_by_name([first, {"names": ["a", "b"], "logits": [1.0, 2.0]}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            align_by_name([{"names": ["a", "a"], "logits": [1.0, 2.0]}])
        with self.assertRaises(ValueError):
            align_by_name([first, {"names": ["a", "b", "c"], "logits": [1.0]}])

    def test_official_order_override(self):
        prediction = {"names": ["c", "a", "b"], "logits": [3.0, 1.0, 2.0]}
        aligned = align_by_name([prediction], order=["a", "b", "c"])
        self.assertEqual(aligned[0]["logits"], [1.0, 2.0, 3.0])


class BootstrapTests(unittest.TestCase):
    def test_determinism_ci_and_probabilities(self):
        names, targets, ref, new = synthetic_tables(rows=200, positives=40)
        first = paired_bootstrap(targets, new, ref, n=200, seed=7)
        second = paired_bootstrap(targets, new, ref, n=200, seed=7)
        self.assertEqual(first, second)
        block = first["metrics"]["auroc"]
        self.assertGreater(block["delta_mean"], 0.1)
        self.assertGreater(block["ci_low"], 0)
        self.assertLessEqual(block["ci_low"], block["ci_high"])
        self.assertGreaterEqual(block["p_win"], 0.0)
        self.assertLessEqual(block["p_win"], 1.0)
        self.assertAlmostEqual(block["p_win"] + block["p_le_zero"], 1.0)
        self.assertEqual((first["n_pos"], first["n_neg"], first["rows"]), (40, 160, 200))

    def test_null_is_centred_and_grouped_mode_runs(self):
        names, targets, ref, _ = synthetic_tables(rows=120, positives=24)
        null = paired_bootstrap(targets, ref, ref, n=50, seed=3)
        self.assertEqual(null["metrics"]["auroc"]["delta_mean"], 0.0)
        self.assertEqual(null["metrics"]["auroc"]["ci_low"], 0.0)
        groups = [index // 2 for index in range(len(targets))]
        grouped = paired_bootstrap(targets, ref, ref, n=30, seed=3, groups=groups)
        self.assertTrue(grouped["grouped"])
        self.assertEqual(grouped["metrics"]["auroc"]["delta_mean"], 0.0)

    def test_auroc_wrapper_and_degenerate_input(self):
        _, targets, ref, new = synthetic_tables(rows=100, positives=20)
        result = paired_bootstrap_auroc(targets, new, ref, n=50, seed=1)
        self.assertEqual(set(result), {"delta_mean", "ci_low", "ci_high", "p_win", "n",
                                       "n_pos", "n_neg"})
        with self.assertRaises(ValueError):
            paired_bootstrap([1, 1, 1], [0.1, 0.2, 0.3], [0.0, 0.0, 0.0], n=5)

    def test_group_same_age_pairs(self):
        groups = group_same_age_pairs([50, 50, 51, float("nan"), 52, 52, 52, 140, 140])
        self.assertEqual(groups, [0, 0, 1, 2, 3, 3, 3, 4, 5])
        self.assertEqual(group_same_age_pairs([50, 51, 50]), [0, 1, 2])

    def test_ranking_metrics(self):
        _, targets, ref, _ = synthetic_tables(rows=100, positives=20)
        metrics = ranking_metrics(targets, ref)
        from sklearn.metrics import roc_auc_score
        self.assertAlmostEqual(metrics["auroc"], roc_auc_score(targets, ref), places=12)
        self.assertEqual(metrics["count"], 100)
        self.assertEqual(metrics["positives"], 20)


class ReportRunsTests(unittest.TestCase):
    def test_scanner_table_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "exp_a" / "seed42_20260101T000000Z_abcd1234"
            run.mkdir(parents=True)
            (run / "config.json").write_text(json.dumps({"experiment": "exp_a", "seed": 42}))
            (run / "metrics.json").write_text(json.dumps({
                "best_epoch": 17, "completed_epochs": 60, "training_complete": True,
                "valid": {"count": 4, "positives": 2, "auroc": 0.91, "auprc_trapz": 0.55,
                          "ap": 0.60, "sensitivity_at_specificity95": 0.5}}))
            (run / "history.json").write_text(json.dumps([{"seconds": 1.25}, {"seconds": 2.75}]))
            names, targets, _, _ = synthetic_tables(rows=4, positives=2)
            write_predictions(run / "valid_predictions.csv", names, targets, [0.1, 0.2, 0.3, 0.4])

            module = load_script("report_runs")
            rows = module.scan_runs(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment"], "exp_a")
            self.assertEqual(rows[0]["seed"], 42)
            self.assertEqual(rows[0]["best_epoch"], 17)
            self.assertAlmostEqual(rows[0]["auroc"], 0.91)
            self.assertAlmostEqual(rows[0]["wall_time_s"], 4.0)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                module.main(["--root", str(root), "--json"])
            payload = json.loads(buffer.getvalue())
            self.assertEqual(payload[0]["path"], str(run.resolve()))

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                module.main(["--root", str(root)])
            self.assertIn("exp_a", buffer.getvalue())
            self.assertIn("0.9100", buffer.getvalue())

    def test_scanner_uses_predictions_when_metrics_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "only_csv"
            run.mkdir()
            names, targets, ref, _ = synthetic_tables(rows=60, positives=12)
            write_predictions(run / "predictions.csv", names, targets, ref)
            rows = load_script("report_runs").scan_runs(root)
            self.assertEqual(len(rows), 1)
            self.assertIsNotNone(rows[0]["auroc"])
            self.assertIsNone(rows[0]["best_epoch"])

    def test_scanner_ignores_unreadable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "broken"
            run.mkdir()
            (run / "metrics.json").write_text("{not json")
            self.assertEqual(load_script("report_runs").scan_runs(root), [])


class FuseScriptTests(unittest.TestCase):
    def test_end_to_end_offline_fuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names, targets, ref, new = synthetic_tables()
            write_predictions(root / "ref.csv", names, targets, ref)
            write_predictions(root / "new.csv", names, targets, new)
            out, report = root / "fused.csv", root / "fused.json"
            module = load_script("fuse_predictions")
            module.main(["--prediction", f"ref={root / 'ref.csv'}",
                         "--prediction", f"new={root / 'new.csv'}",
                         "--method", "logit", "--weights", "1,1",
                         "--output", str(out), "--report", str(report),
                         "--bootstrap", "50"])
            with open(out, encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), ["Image_name", "True_Label", "Logit", "Probability"])
            self.assertEqual(len(rows), len(names))
            self.assertEqual([row["Image_name"] for row in rows], names)
            payload = json.loads(report.read_text())
            self.assertEqual(payload["method"], "logit")
            self.assertEqual(payload["members"], ["ref", "new"])
            self.assertEqual(payload["standardizer"]["fitted_on"], "input_leaky")
            self.assertEqual(payload["rows"], len(names))
            self.assertIsNotNone(payload["fused"]["auroc"])
            self.assertIn("bootstrap_vs_best_single", payload)
            self.assertEqual(payload["bootstrap_vs_best_single"]["n"], 50)

    def test_rank_and_prob_methods_write_valid_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names, targets, ref, new = synthetic_tables()
            write_predictions(root / "ref.csv", names, targets, ref)
            write_predictions(root / "new.csv", names, targets, new)
            module = load_script("fuse_predictions")
            for method in ("rank", "prob"):
                out = root / f"{method}.csv"
                module.main(["--prediction", f"ref={root / 'ref.csv'}",
                             "--prediction", f"new={root / 'new.csv'}",
                             "--method", method, "--output", str(out),
                             "--report", str(root / f"{method}.json")])
                with open(out, encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                probabilities = np.asarray([float(row["Probability"]) for row in rows])
                self.assertTrue(((probabilities > 0) & (probabilities < 1)).all())

    def test_test_mode_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [f"test_{index:03d}" for index in range(20)]
            (root / "test.csv").write_text("Image_name,Age\n" + "\n".join(
                f"{name},60" for name in names) + "\n")
            for member, shift in (("ref", 0.0), ("new", 0.4)):
                with open(root / f"{member}.csv", "w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["Image_name", "Probability", "Pred_Label"])
                    for index, name in enumerate(names):
                        probability = float(sigmoid(index / 5.0 + shift))
                        writer.writerow([name, format(probability, ".10g"),
                                         int(probability >= 0.5)])
            out = root / "submission.csv"
            module = load_script("fuse_predictions")
            module.main(["--test", "--test-csv", str(root / "test.csv"),
                         "--prediction", f"ref={root / 'ref.csv'}",
                         "--prediction", f"new={root / 'new.csv'}",
                         "--method", "logit", "--output", str(out)])
            with open(out, encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["Image_name"] for row in rows], names)
            self.assertEqual(list(rows[0]), ["Image_name", "Probability", "Pred_Label"])
            for row in rows:
                self.assertIn(row["Pred_Label"], ("0", "1"))

    def test_missing_member_width_and_name_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names, targets, ref, new = synthetic_tables(rows=30, positives=6)
            write_predictions(root / "ref.csv", names, targets, ref)
            write_predictions(root / "new.csv", names[:-1], targets[:-1], new[:-1])
            module = load_script("fuse_predictions")
            with self.assertRaisesRegex(ValueError, "name set differs"):
                module.main(["--prediction", f"ref={root / 'ref.csv'}",
                             "--prediction", f"new={root / 'new.csv'}",
                             "--output", str(root / "out.csv")])
            with self.assertRaises(SystemExit):
                module.main(["--prediction", f"ref={root / 'ref.csv'}",
                             "--output", str(root / "out2.csv")])


class CompareScriptTests(unittest.TestCase):
    def test_verdict_and_grouped_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names, targets, ref, new = synthetic_tables(rows=200, positives=40)
            write_predictions(root / "ref.csv", names, targets, ref)
            write_predictions(root / "new.csv", names, targets, new)
            (root / "valid.csv").write_text("Image_name,Age\n" + "\n".join(
                f"{name},{50 + index // 2}" for index, name in enumerate(names)) + "\n")
            module = load_script("compare_predictions")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                module.main([str(root / "new.csv"), str(root / "ref.csv"), "--reps", "100"])
            output = buffer.getvalue()
            self.assertIn("Verdict: WIN", output)
            self.assertIn("ΔAUROC", output)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                module.main([str(root / "new.csv"), str(root / "ref.csv"), "--reps", "100",
                             "--group-by-age-pair", "--data-root", str(root)])
            grouped = buffer.getvalue()
            self.assertIn("sensitivity", grouped)
            self.assertIn("Verdict:", grouped)


if __name__ == "__main__":
    unittest.main()
