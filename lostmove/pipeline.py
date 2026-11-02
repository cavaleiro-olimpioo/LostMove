"""Orquestra a leitura sequencial e a análise offline do vídeo."""

from dataclasses import dataclass
from pathlib import Path

from . import config
from .analysis import MotionAnalysis, analyze_motion
from .detector import detect_ball
from .preprocess import Preprocessor
from .video_io import VideoMetadata, open_video


@dataclass(frozen=True)
class VideoAnalysisResult:
    """Análise do movimento e estatísticas da leitura de vídeo."""

    metadata: VideoMetadata
    detected_frames: int
    analysis: MotionAnalysis


def analyze_video(video_path: Path) -> VideoAnalysisResult:
    """Processa todos os frames e calcula os primeiros ciclos completos.

    Args:
        video_path: Arquivo de vídeo a analisar.

    Returns:
        Metadados, quantidade de detecções e medidas dos ciclos.

    Raises:
        ValueError: Se houver menos ciclos completos que o configurado.
        VideoOpenError: Se o vídeo não puder ser aberto ou lido.
    """
    reader = open_video(video_path)
    positions: list[float | None] = []
    detected_frames = 0
    preprocessor = Preprocessor()

    try:
        for _, _, frame in reader.iter_frames():
            detection = detect_ball(preprocessor.process(frame))
            if detection is None:
                positions.append(None)
            else:
                positions.append(detection.centroid[0])
                detected_frames += 1

        analysis = analyze_motion(
            x_positions=positions,
            fps=reader.metadata.fps,
            origin_x_px=config.OSCILLATION_ORIGIN_X_PX,
            hysteresis_px=config.ORIGIN_HYSTERESIS_PX,
            max_gap_frames=config.MAX_INTERPOLATION_GAP_FRAMES,
        )
        return VideoAnalysisResult(
            metadata=reader.metadata,
            detected_frames=detected_frames,
            analysis=analysis,
        )
    finally:
        reader.release()