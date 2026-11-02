"""Testes unitários da análise de ciclos e do exportador CSV."""

import csv
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np

from lostmove.analysis import analyze_motion, interpolate_short_gaps
from lostmove.catalog import choose_video, find_videos, tested_video_names
from lostmove.export import export_cycles_csv
from lostmove.pipeline import VideoAnalysisResult
from lostmove.video_io import VideoMetadata
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
            exported_path = export_cycles_csv(
                analysis, 30.0, output_path, "pendulo.mp4"
            )
            with exported_path.open(newline="", encoding="utf-8-sig") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(
            len([row for row in rows if row["tipo"] == "Ciclo"]),
            len(analysis.cycles),
        )
        summary = next(row for row in rows if row["tipo"] == "Resumo")
        self.assertIn("comparar os balancos", summary["observação"])
        self.assertIn("tempo de ida e volta (segundos)", rows[0])
        self.assertEqual(rows[0]["vídeo"], "pendulo.mp4")
        self.assertIn("tempo do primeiro ciclo (segundos)", summary)
        self.assertIn("tempo do último ciclo (segundos)", summary)
        self.assertIn("mudança média no tempo por ciclo (segundos)", summary)

    def test_lista_videos_e_mostra_quais_ja_foram_testados(self) -> None:
        """Lista formatos aceitos e mostra o estado de cada vídeo."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "b.MP4").write_bytes(b"")
            (data_dir / "anotacoes.txt").write_text("ignorar", encoding="utf-8")
            subfolder = data_dir / "subpasta"
            subfolder.mkdir()
            (subfolder / "a.avi").write_bytes(b"")

            videos = find_videos(data_dir)
            output: list[str] = []
            choices = iter(("x", "2"))
            selected = choose_video(
                videos,
                data_dir,
                {"b.MP4"},
                input_func=lambda _: next(choices),
                output_func=output.append,
            )

        self.assertEqual(
            [path.relative_to(data_dir).as_posix() for path in videos],
            ["b.MP4", "subpasta/a.avi"],
        )
        self.assertEqual(selected, videos[1])
        self.assertTrue(any("[TESTADO] b.MP4" in line for line in output))
        self.assertTrue(any("[NOVO] subpasta/a.avi" in line for line in output))
        self.assertTrue(any("Escolha inválida" in line for line in output))

    def test_csv_preserva_outro_video_e_atualiza_reteste(self) -> None:
        """Guarda resultados de vários vídeos sem duplicar um reteste."""
        frames = np.arange(1600)
        time_s = frames / 30.0
        positions = 250 + 180 * np.exp(-0.005 * time_s) * np.sin(
            2 * np.pi * time_s / 2.0
        )
        analysis = analyze_motion(positions.tolist(), 30.0, 250.0, 40.0, 8)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "cycles.csv"
            export_cycles_csv(analysis, 30.0, output_path, "video-a.mp4")
            export_cycles_csv(analysis, 30.0, output_path, "video-b.mp4")
            export_cycles_csv(analysis, 30.0, output_path, "video-a.mp4")

            with output_path.open(newline="", encoding="utf-8-sig") as csv_file:
                rows = list(csv.DictReader(csv_file))
            tested = tested_video_names(output_path)

        self.assertEqual(tested, {"video-a.mp4", "video-b.mp4"})
        self.assertEqual(
            sum(row["vídeo"] == "video-a.mp4" for row in rows),
            len(analysis.cycles) + 1,
        )
        self.assertEqual(
            sum(row["vídeo"] == "video-b.mp4" for row in rows),
            len(analysis.cycles) + 1,
        )

    def test_menu_salva_e_marca_video_automaticamente(self) -> None:
        """Escolhe um vídeo, salva seus dados e marca-o no próximo menu."""
        import main

        frames = np.arange(1600)
        time_s = frames / 30.0
        positions = 250 + 180 * np.exp(-0.005 * time_s) * np.sin(
            2 * np.pi * time_s / 2.0
        )
        analysis = analyze_motion(positions.tolist(), 30.0, 250.0, 40.0, 8)
        result = VideoAnalysisResult(
            metadata=VideoMetadata(
                fps=30.0,
                width=480,
                height=854,
                total_frames=len(frames),
            ),
            detected_frames=len(frames),
            analysis=analysis,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)
            data_dir = project_dir / "data"
            data_dir.mkdir()
            (data_dir / "ensaio.mp4").write_bytes(b"video de teste")
            csv_path = project_dir / "results" / "ciclos.csv"
            output = StringIO()

            with (
                patch.object(main.config, "DATA_DIR", data_dir),
                patch.object(main.config, "RESULTS_CSV_PATH", csv_path),
                patch.object(main, "analyze_video", return_value=result),
                patch("builtins.input", return_value="1"),
                redirect_stdout(output),
            ):
                self.assertEqual(main.main(["--sem-janelas"]), 0)

            output.seek(0)
            output.truncate(0)
            with (
                patch.object(main.config, "DATA_DIR", data_dir),
                patch.object(main.config, "RESULTS_CSV_PATH", csv_path),
                patch.object(main, "analyze_video", return_value=result),
                patch("builtins.input", return_value="1"),
                redirect_stdout(output),
            ):
                self.assertEqual(main.main(["--sem-janelas"]), 0)

            with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertIn("[TESTADO] ensaio.mp4", output.getvalue())
        self.assertEqual(
            sum(row["vídeo"] == "ensaio.mp4" for row in rows),
            len(analysis.cycles) + 1,
        )


if __name__ == "__main__":
    unittest.main()
