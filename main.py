"""Orquestra pré-processamento, detecção e visualização interativa."""

import argparse
from collections import deque
import importlib.util
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path


def configure_opencv_window_system() -> None:
    """Escolhe XCB quando Wayland não está incluído no OpenCV instalado."""
    if sys.platform != "linux" or not os.environ.get("WAYLAND_DISPLAY"):
        return
    if os.environ.get("QT_QPA_PLATFORM"):
        return

    cv2_spec = importlib.util.find_spec("cv2")
    if cv2_spec is None or cv2_spec.origin is None:
        return

    platform_folder = Path(cv2_spec.origin).parent / "qt" / "plugins" / "platforms"
    has_wayland = any(platform_folder.glob("libqwayland*.so"))
    has_xcb = any(platform_folder.glob("libqxcb.so"))
    if has_xcb and not has_wayland:
        # A sessão tem Wayland, mas algumas versões do pacote OpenCV trazem
        # somente XCB; o display X disponível mantém as janelas funcionais.
        os.environ["QT_QPA_PLATFORM"] = "xcb"


configure_opencv_window_system()

import cv2

from lostmove import config
from lostmove.catalog import choose_video, find_videos, tested_video_names
from lostmove.export import export_cycles_csv
from lostmove.detector import detect_ball
from lostmove.preprocess import Preprocessor
from lostmove.pipeline import VideoAnalysisResult, analyze_video
from lostmove.video_io import VideoOpenError, open_video
from lostmove.viewer import PlaybackState, VideoViewer


def cycle_label_for_frame(frame_idx: int, result: VideoAnalysisResult) -> str:
    """Descreve o ciclo e a parte do movimento mostrados neste quadro."""
    cycles = result.analysis.cycles
    for cycle in cycles:
        if cycle.start_frame <= frame_idx <= cycle.end_frame:
            if frame_idx == cycle.right_extremum_frame:
                phase = "extremo direito"
            elif frame_idx == cycle.left_extremum_frame:
                phase = "extremo esquerdo"
            elif frame_idx < cycle.right_extremum_frame:
                phase = "indo para a direita"
            elif frame_idx < cycle.opposite_origin_frame:
                phase = "voltando ao centro"
            elif frame_idx < cycle.left_extremum_frame:
                phase = "indo para a esquerda"
            else:
                phase = "voltando ao centro"
            return f"Ciclo {cycle.number} de {len(cycles)} | {phase}"

    return f"Fora dos ciclos completos | {len(cycles)} ciclos encontrados"


def show_video_with_cycles(
    result: VideoAnalysisResult, video_path: Path
) -> None:
    """Reproduz o vídeo e identifica os ciclos encontrados na imagem."""
    reader = open_video(video_path)

    metadata = reader.metadata
    preprocessor = Preprocessor()
    playback = PlaybackState()
    centers: list[tuple[int, int] | None] = [None] * metadata.total_frames
    viewer: VideoViewer | None = None

    try:
        viewer = VideoViewer()
        while True:
            frame_started = time.perf_counter()
            current = reader.read_frame(playback.frame_idx)
            if current is None:
                break
            timestamp_s, frame = current
            mask = preprocessor.process(frame)
            detection = detect_ball(mask)

            if detection is not None:
                centers[playback.frame_idx] = (
                    round(detection.centroid[0]),
                    round(detection.centroid[1]),
                )

            trail_start = max(0, playback.frame_idx - config.TRAIL_LENGTH + 1)
            trail: deque[tuple[int, int]] = deque(
                (
                    center
                    for center in centers[trail_start : playback.frame_idx + 1]
                    if center is not None
                ),
                maxlen=config.TRAIL_LENGTH,
            )

            display_frame = viewer.render_original(
                frame,
                playback.frame_idx,
                timestamp_s,
                detection,
                trail,
                cycle_label_for_frame(playback.frame_idx, result),
            )
            viewer.show(display_frame, mask)

            if playback.paused:
                delay_ms = 0
            else:
                frame_period_ms = 1000 / metadata.fps
                elapsed_ms = (time.perf_counter() - frame_started) * 1000
                delay_ms = max(1, round(frame_period_ms - elapsed_ms))
            key = viewer.wait_key(delay_ms)
            key &= 0xFF
            if key == ord("q"):
                break

            if playback.paused:
                if key == ord(" "):
                    playback.paused = False
                    playback.frame_idx = min(
                        playback.frame_idx + 1, metadata.total_frames - 1
                    )
                elif key == ord("d"):
                    playback.frame_idx = min(
                        playback.frame_idx + 1, metadata.total_frames - 1
                    )
                elif key == ord("a"):
                    playback.frame_idx = max(playback.frame_idx - 1, 0)
            elif key == ord(" "):
                playback.paused = True
            else:
                playback.frame_idx += 1
    finally:
        reader.release()
        if viewer is not None:
            viewer.close()


def print_results(result: VideoAnalysisResult, csv_path: str) -> None:
    """Mostra em palavras simples um resumo dos resultados."""
    analysis = result.analysis
    print(
        f"Quadros em que a bola foi encontrada: "
        f"{result.detected_frames}/{result.metadata.total_frames}"
    )
    print(f"Centro do balanço: {analysis.origin_x_px:.2f} pixels")
    print(f"Tempo médio de ida e volta: {analysis.mean_cycle_time_s:.3f} segundos")
    print(f"Tempo do primeiro ciclo: {analysis.first_cycle_time_s:.3f} segundos")
    print(f"Tempo do último ciclo: {analysis.last_cycle_time_s:.3f} segundos")
    print(
        "Mudança média no tempo por ciclo: "
        f"{analysis.mean_cycle_time_change_s_per_cycle:+.4f} segundos"
    )
    print("Um valor negativo significa que o ciclo ficou mais curto.")
    print(f"Distância média do centro: {analysis.mean_amplitude_px:.2f} pixels")
    print(
        "Perda média estimada de movimento: "
        f"{analysis.mean_energy_loss_percent_per_cycle:.3f}% por ida e volta"
    )
    print(
        "Perda estimada do primeiro ao último ciclo: "
        f"{analysis.overall_energy_loss_percent:.3f}%"
    )
    print(f"Total de ciclos completos encontrados: {len(analysis.cycles)}")
    print(f"Planilha salva em: {csv_path}")
    print(
        "A perda é uma estimativa para comparar os balanços, "
        "não mede a energia exata."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Conta todos os ciclos completos, mostra o vídeo e salva a planilha.

    Args:
        argv: Argumentos opcionais de linha de comando.

    Returns:
        Código de saída do processo.
    """
    parser = argparse.ArgumentParser(
        description="Conta todos os ciclos completos da bola e calcula os resultados."
    )
    parser.add_argument(
        "--sem-janelas",
        action="store_true",
        help="analisa o vídeo sem abrir as janelas",
    )
    arguments = parser.parse_args(argv)

    videos = find_videos(config.DATA_DIR)
    tested_names = tested_video_names(config.RESULTS_CSV_PATH)
    video_path = choose_video(videos, config.DATA_DIR, tested_names)
    if video_path is None:
        return 1 if not videos else 0

    video_name = video_path.relative_to(config.DATA_DIR).as_posix()
    if video_name in tested_names:
        print("Este vídeo já foi testado; os resultados antigos serão atualizados.")

    try:
        print(f"Analisando {video_name} e contando os ciclos completos...")
        result = analyze_video(video_path)
    except (VideoOpenError, ValueError, cv2.error) as error:
        print(f"[ERRO] {error}", file=sys.stderr)
        return 1

    csv_path = export_cycles_csv(
        result.analysis,
        result.metadata.fps,
        config.RESULTS_CSV_PATH,
        video_name,
    )
    print_results(result, str(csv_path))
    if not arguments.sem_janelas:
        show_video_with_cycles(result, video_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())