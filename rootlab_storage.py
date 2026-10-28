"""SQLite persistence and error statistics for RootLab experiments."""

from datetime import datetime, timezone
from contextlib import closing
import math
from pathlib import Path
import sqlite3
from statistics import mean, pvariance


DEFAULT_DATABASE_PATH = Path(__file__).with_name("rootlab_experiments.sqlite3")


def initialize_database(database_path=DEFAULT_DATABASE_PATH):
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id TEXT PRIMARY KEY,
                    expression TEXT NOT NULL,
                    method TEXT NOT NULL,
                    initial_guess REAL,
                    interval_a REAL,
                    interval_b REAL,
                    tolerance REAL NOT NULL,
                    approximate_root REAL NOT NULL,
                    iteration_count INTEGER NOT NULL,
                    convergence_status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    reference_root REAL,
                    absolute_error REAL
                )
                """
            )
    return path


def record_experiment(experiment, database_path=DEFAULT_DATABASE_PATH):
    path = initialize_database(database_path)
    root = float(experiment["approximate_root"])
    tolerance = float(experiment["tolerance"])
    iteration_count = int(experiment["iteration_count"])
    if not math.isfinite(root) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Root and tolerance must be finite, and tolerance must be positive.")
    if iteration_count < 0:
        raise ValueError("Iteration count cannot be negative.")

    reference_root = experiment.get("reference_root")
    try:
        reference_root = float(reference_root) if reference_root is not None else None
    except (TypeError, ValueError):
        reference_root = None
    if reference_root is not None and not math.isfinite(reference_root):
        reference_root = None

    absolute_error = abs(root - reference_root) if reference_root is not None else None
    if absolute_error is not None and not math.isfinite(absolute_error):
        reference_root = None
        absolute_error = None
    timestamp = experiment.get("timestamp") or datetime.now(timezone.utc).isoformat()

    with closing(sqlite3.connect(path)) as connection:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO experiments (
                    run_id, expression, method, initial_guess, interval_a, interval_b,
                    tolerance, approximate_root, iteration_count, convergence_status,
                    timestamp, reference_root, absolute_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO NOTHING
                """,
                (
                    str(experiment["run_id"]),
                    str(experiment["expression"]),
                    str(experiment["method"]),
                    experiment.get("initial_guess"),
                    experiment.get("interval_a"),
                    experiment.get("interval_b"),
                    tolerance,
                    root,
                    iteration_count,
                    str(experiment["convergence_status"]),
                    timestamp,
                    reference_root,
                    absolute_error,
                ),
            )
    return cursor.rowcount == 1


def get_experiments(database_path=DEFAULT_DATABASE_PATH):
    path = initialize_database(database_path)
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        with connection:
            rows = connection.execute(
                "SELECT * FROM experiments ORDER BY timestamp DESC, rowid DESC"
            ).fetchall()
    return [dict(row) for row in rows]


def calculate_error_statistics(experiments):
    errors = []
    for experiment in experiments:
        reference = experiment.get("reference_root")
        root = experiment.get("approximate_root")
        try:
            reference = float(reference)
            root = float(root)
        except (TypeError, ValueError):
            continue
        if math.isfinite(reference) and math.isfinite(root):
            absolute_error = abs(root - reference)
            if math.isfinite(absolute_error):
                errors.append(absolute_error)

    if not errors:
        return {"count": 0, "mean": None, "population_variance": None}
    return {
        "count": len(errors),
        "mean": mean(errors),
        "population_variance": pvariance(errors),
    }