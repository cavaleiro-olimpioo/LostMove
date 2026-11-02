"""Testes unitários da análise de ciclos e do exportador CSV."""

import csv
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from lostmove.analysis import analyze_motion, interpolate_short_gaps
from lostmove.export import export_cycles_csv
from main import cycle_label_for_frame


class TesteAnaliseDoBalanco(unittest.TestCase):
    """Confere os tempos e as medidas do balanço."""

    def test_preenche_falhas_pequenas(self) -> None:
        """Completa falhas curtas entre duas posições conhecidas."""
        values = interpolate_short_gaps(
            [None, 10.0, None, None, 16.0, None, None, None, None, None, 30.0],
            max_gap_frames=2,
        )
        self.assertTrue(np.isnan(values[0]))
        np.testing.assert_allclose(values[1:5], [10.0, 12.0, 14.0, 16.0])
        self.assertTrue(np.isnan(values[5:10]).all())

    def test_mede_todos_os_balancos(self) -> None:
        """Encontra todos os ciclos completos, inclusive mais de dez."""
        fps = 30.0
        frames = np.arange(1600)
        time_s = frames / fps
        positions = 250 + 180 * np.exp(-0.005 * time_s) * np.sin(
            2 * np.pi * time_s / 2.0
        )

        result = analyze_motion(
            x_positions=positions.tolist(),
            fps=fps,
            origin_x_px=250.0,
            hysteresis_px=40.0,
            max_gap_frames=8,
        )

        self.assertGreater(len(result.cycles), 10)
        self.assertEqual(
            [cycle.number for cycle in result.cycles],
            list(range(1, len(result.cycles) + 1)),
        )
        self.assertAlmostEqual(result.mean_cycle_time_s, 2.0, delta=0.03)
        self.assertEqual(
            result.first_cycle_time_s,
            result.cycles[0].duration_s,
        )
        self.assertEqual(
            result.last_cycle_time_s,
            result.cycles[-1].duration_s,
        )
        expected_time_change = (
            result.last_cycle_time_s - result.first_cycle_time_s
        ) / 9
        self.assertAlmostEqual(
            result.mean_cycle_time_change_s_per_cycle,
            expected_time_change,
        )
        self.assertGreater(result.mean_energy_loss_percent_per_cycle, 0)
        self.assertLess(
            result.cycles[-1].energy_proxy_relative,
            result.cycles[0].energy_proxy_relative,
        )
        expected_loss = (
            1
            - (
                result.cycles[-1].energy_proxy_relative
                / result.cycles[0].energy_proxy_relative
            )
            ** (1 / (len(result.cycles) - 1))
        ) * 100
        self.assertAlmostEqual(
            result.mean_energy_loss_percent_per_cycle,
            expected_loss,
        )

    def test_identifica_ciclo_e_extremo_no_quadro(self) -> None:
        """Mostra o número automático do ciclo e o extremo correspondente."""
        frames = np.arange(1600)
        time_s = frames / 30.0
        positions = 250 + 180 * np.exp(-0.005 * time_s) * np.sin(
            2 * np.pi * time_s / 2.0
        )
        analysis = analyze_motion(
            x_positions=positions.tolist(),
            fps=30.0,
            origin_x_px=250.0,
            hysteresis_px=40.0,
            max_gap_frames=8,
        )
        result = SimpleNamespace(analysis=analysis)
        first_cycle = analysis.cycles[0]

        label = cycle_label_for_frame(first_cycle.right_extremum_frame, result)

        self.assertEqual(
            label,
            f"Ciclo 1 de {len(analysis.cycles)} | extremo direito",
        )

    def test_avisa_quando_nao_ha_ciclo_completo(self) -> None:
        """Avisa quando o vídeo não contém nenhuma ida e volta completa."""
        frames = np.arange(20)
        positions = 250 + 100 * np.sin(2 * np.pi * frames / 60)
        with self.assertRaisesRegex(ValueError, "ciclos completos"):
            analyze_motion(
                x_positions=positions.tolist(),
                fps=30.0,
                origin_x_px=250.0,
                hysteresis_px=20.0,
                max_gap_frames=2,
            )

    def test_salva_balancos_e_resumo_no_csv(self) -> None:
        """Salva todos os ciclos e uma linha com os valores médios."""
        frames = np.arange(1600)
        time_s = frames / 30.0
        positions = 250 + 180 * np.exp(-0.005 * time_s) * np.sin(
            2 * np.pi * time_s / 2.0
        )
        analysis = analyze_motion(
            x_positions=positions.tolist(),
            fps=30.0,
            origin_x_px=250.0,
            hysteresis_px=40.0,
            max_gap_frames=8,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "results" / "cycles.csv"
            exported_path = export_cycles_csv(analysis, 30.0, output_path)
            with exported_path.open(newline="", encoding="utf-8-sig") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(
            len([row for row in rows if row["tipo"] == "Ciclo"]),
            len(analysis.cycles),
        )
        summary = next(row for row in rows if row["tipo"] == "Resumo")
        self.assertIn("comparar os balancos", summary["observação"])
        self.assertIn("tempo de ida e volta (segundos)", rows[0])
        self.assertIn("tempo do primeiro ciclo (segundos)", summary)
        self.assertIn("tempo do último ciclo (segundos)", summary)
        self.assertIn("mudança média no tempo por ciclo (segundos)", summary)


if __name__ == "__main__":
    unittest.main()
