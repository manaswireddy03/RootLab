import tempfile
import unittest
from pathlib import Path

from rootlab_storage import (
    calculate_error_statistics,
    get_experiments,
    initialize_database,
    record_experiment,
)


class RootLabStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "experiments.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_records_persist_after_reinitialization(self):
        initialize_database(self.database_path)
        experiment = {
            "run_id": "run-1",
            "expression": "x^2 - 4",
            "method": "Bisection",
            "interval_a": 0.0,
            "interval_b": 3.0,
            "tolerance": 1e-4,
            "approximate_root": 2.00001,
            "iteration_count": 14,
            "convergence_status": "Converged.",
            "reference_root": 2.0,
            "timestamp": "2026-10-28T12:00:00+00:00",
        }

        self.assertTrue(record_experiment(experiment, self.database_path))
        self.assertFalse(record_experiment(experiment, self.database_path))
        initialize_database(self.database_path)

        records = get_experiments(self.database_path)
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["absolute_error"], 0.00001)
        self.assertEqual(records[0]["interval_a"], 0.0)
        self.assertIsNone(records[0]["initial_guess"])

    def test_error_statistics_exclude_missing_or_invalid_references(self):
        records = [
            {"approximate_root": 2.1, "reference_root": 2.0},
            {"approximate_root": 3.8, "reference_root": 4.0},
            {"approximate_root": 10.0, "reference_root": None},
            {"approximate_root": 10.0, "reference_root": float("nan")},
            {"approximate_root": 1e308, "reference_root": -1e308},
        ]

        statistics = calculate_error_statistics(records)

        self.assertEqual(statistics["count"], 2)
        self.assertAlmostEqual(statistics["mean"], 0.15)
        self.assertAlmostEqual(statistics["population_variance"], 0.0025)


if __name__ == "__main__":
    unittest.main()