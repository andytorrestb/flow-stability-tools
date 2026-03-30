import json
import tempfile
import unittest
from pathlib import Path

from analysis.registry import get_analysis_registry
from io.artifacts import ArtifactManager
from config.schema import SimulationConfig
from core.case import Case
from core.context import PipelineContext
from core.results import AnalysisSummary, ProfileScanResult, ScanResult


class TestAnalysisRegistry(unittest.TestCase):
    def test_default_strategy_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            results_dir.mkdir(parents=True, exist_ok=True)

            config = SimulationConfig.model_validate(
                {
                    "case_name": "test",
                    "solver": {"type": "rayleigh_inviscid", "profiles": ["tanh_shear"]},
                    "numerical": {"L": 1.0, "N": 8},
                    "alpha_scan": {"alpha_min": 0.1, "alpha_max": 0.2, "alpha_count": 3},
                    "output": {"enable_plots": False, "export_vtk": False, "export_sympy_pdf": False},
                    "pipeline": {"steps": ["setup", "solve", "analysis"]},
                }
            )

            case = Case(root_dir=results_dir, name="test")
            context = PipelineContext()

            profile_scan = ProfileScanResult(
                z=[-1.0, 0.0, 1.0],
                U=[1.0, 0.0, 1.0],
                Upp=[2.0, 2.0, 2.0],
                sympy={},
                scan={
                    "alpha": [0.1, 0.15, 0.2],
                    "dominant_growth": [0.0, 0.1, 0.0],
                    "dominant_frequency": [1.0, 1.0, 1.0],
                    "dominant_omega_real": [1.0, 1.0, 1.0],
                    "dominant_omega_imag": [0.0, 0.1, 0.0],
                    "dominant_eigenfunctions": [None, None, None],
                    "summary": {
                        "alpha_star": 0.15,
                        "omega_star_real": 1.0,
                        "omega_star_imag": 0.1,
                        "growth_rate": 0.1,
                        "frequency": 1.0,
                        "eigenfunction_star": None,
                    },
                },
                dominant_eigenfunction=None,
            )

            context.scan_results = ScanResult(
                temporal_convention="tc",
                equation="eq",
                expected_behavior={"tanh_shear": "note"},
                profiles={"tanh_shear": profile_scan},
                refinement={"enabled": False},
            )

            artifact_manager = ArtifactManager(
                results_dir=results_dir,
                growth_plot_filename="growth.png",
                frequency_plot_filename="freq.png",
                sympy_pdf_filename="sympy.pdf",
                vtk_filename_pattern="{profile}_field.vtk",
                time_series_mp4_filename="time_series.mp4",
                velocity_time_series_mp4_filename="velocity_time_series.mp4",
            )

            strategy = get_analysis_registry().get("default")
            strategy(case=case, context=context, config=config, artifact_manager=artifact_manager)

            summary_path = results_dir / "profile_summaries.json"
            self.assertTrue(summary_path.exists())
            data = json.loads(summary_path.read_text())
            self.assertIn("tanh_shear", data["profiles"])
            self.assertEqual(data["profiles"]["tanh_shear"]["alpha_star"], 0.15)


if __name__ == "__main__":
    unittest.main()