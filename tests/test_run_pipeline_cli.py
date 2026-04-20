import io
import tempfile
import unittest
from contextlib import redirect_stdout
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from tennis_pipeline.cli import run_pipeline as rp


class RunPipelineCliTests(unittest.TestCase):
    def test_clustering_method_reaches_step_config(self) -> None:
        seen_step_configs: dict[str, dict[str, object]] = {}
        real_import_module = importlib.import_module

        def fake_import_module(name: str):
            if not name.startswith("tennis_pipeline.steps."):
                return real_import_module(name)
            step_name = name.rsplit(".", 1)[-1]

            def _run(_current, config=None):
                seen_step_configs[step_name] = dict(config or {})
                return pd.DataFrame(
                    {
                        "event_id": ["e1", "e2"],
                        "match_id": ["m1", "m2"],
                        "match_date": ["2024-01-01", "2024-01-02"],
                        "match_seq": [1, 2],
                        "team1_player_id": ["p1", "p2"],
                        "team2_player_id": ["q1", "q2"],
                        "team1_wins": [1, 0],
                    }
                )

            return SimpleNamespace(run=_run)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            input_path.write_text("a\n1\n", encoding="utf-8")
            with (
                patch.object(
                    rp,
                    "STEP_MODULES",
                    ("01_load_raw", "06c_build_features_clustering", "07_finalize_model_table"),
                ),
                patch.object(rp.importlib, "import_module", side_effect=fake_import_module),
                patch.object(rp, "run_stage_checks"),
                patch.object(rp, "run_model_training_experiments", return_value={}),
                patch.object(pd.DataFrame, "to_parquet", return_value=None),
            ):
                rp.run_pipeline(
                    input_path=input_path,
                    output_dir=tmpdir,
                    clustering_method="both",
                )

        self.assertEqual("both", seen_step_configs["06c_build_features_clustering"]["method"])

    def test_feature_set_mode_enables_debug_leakage_for_training(self) -> None:
        calls: list[dict[str, object]] = []

        real_import_module = importlib.import_module

        def fake_import_module(name: str):
            if not name.startswith("tennis_pipeline.steps."):
                return real_import_module(name)

            def _run(_current, config=None):
                return pd.DataFrame(
                    {
                        "event_id": ["e1", "e2", "e3"],
                        "match_id": ["m1", "m2", "m3"],
                        "match_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                        "match_seq": [1, 2, 3],
                        "team1_player_id": ["p1", "p2", "p3"],
                        "team2_player_id": ["q1", "q2", "q3"],
                        "feature_x": [0.1, 0.2, 0.3],
                        "team1_wins": [1, 0, 1],
                    }
                )

            return SimpleNamespace(run=_run)

        def fake_run_model_training_experiments(_df, *, output_dir, config=None):
            calls.append({"kind": "baseline", "config": dict(config or {}), "output_dir": str(output_dir)})
            return {}

        def fake_run_feature_set_training_experiment(
            _feature_set_tables, *, output_dir, config=None, start_run_index=None, total_runs=None
        ):
            calls.append({"kind": "feature_sets", "config": dict(config or {}), "output_dir": str(output_dir)})
            return {}

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            input_path.write_text("a\n1\n", encoding="utf-8")
            with (
                patch.object(rp, "STEP_MODULES", ("01_load_raw", "07_finalize_model_table")),
                patch.object(rp.importlib, "import_module", side_effect=fake_import_module),
                patch.object(rp, "run_stage_checks"),
                patch.object(rp, "materialize_feature_sets", return_value={"raw_only": pd.DataFrame({"team1_wins": [1]})}),
                patch.object(rp, "run_model_training_experiments", side_effect=fake_run_model_training_experiments),
                patch.object(
                    rp,
                    "run_feature_set_training_experiment",
                    side_effect=fake_run_feature_set_training_experiment,
                ),
                patch.object(pd.DataFrame, "to_parquet", return_value=None),
            ):
                rp.run_pipeline(
                    input_path=input_path,
                    output_dir=tmpdir,
                    run_feature_set_experiment=True,
                )

        self.assertEqual(2, len(calls))
        self.assertTrue(calls[0]["config"]["debug_leakage"])
        self.assertTrue(calls[1]["config"]["debug_leakage"])

    def test_progress_updates_print_to_console(self) -> None:
        real_import_module = importlib.import_module

        def fake_import_module(name: str):
            if not name.startswith("tennis_pipeline.steps."):
                return real_import_module(name)

            def _run(_current, config=None):
                return pd.DataFrame(
                    {
                        "event_id": ["e1", "e2", "e3"],
                        "match_id": ["m1", "m2", "m3"],
                        "match_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                        "match_seq": [1, 2, 3],
                        "team1_player_id": ["p1", "p2", "p3"],
                        "team2_player_id": ["q1", "q2", "q3"],
                        "feature_x": [0.1, 0.2, 0.3],
                        "team1_wins": [1, 0, 1],
                    }
                )

            return SimpleNamespace(run=_run)

        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            input_path.write_text("a\n1\n", encoding="utf-8")
            with (
                patch.object(rp, "STEP_MODULES", ("01_load_raw", "07_finalize_model_table")),
                patch.object(rp.importlib, "import_module", side_effect=fake_import_module),
                patch.object(rp, "run_stage_checks"),
                patch.object(rp, "materialize_feature_sets", return_value={}),
                patch.object(rp, "run_model_training_experiments", return_value={}),
                patch.object(pd.DataFrame, "to_parquet", return_value=None),
                redirect_stdout(stdout),
            ):
                rp.run_pipeline(
                    input_path=input_path,
                    output_dir=tmpdir,
                    run_feature_set_experiment=True,
                )

        output = stdout.getvalue()
        self.assertIn("[pipeline] --run-feature-set-experiment enabled", output)
        self.assertIn("[pipeline] step 01_load_raw: start", output)
        self.assertIn("[pipeline] running baseline model-training experiment", output)
        self.assertIn("[pipeline] pipeline run complete", output)


if __name__ == "__main__":
    unittest.main()
