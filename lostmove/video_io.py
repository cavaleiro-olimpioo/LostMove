"""Abertura de vídeo, leitura de metadados e iteração de frames."""

from collections.abc import Iterator
from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np


class VideoOpenError(Exception):
    """Indica que o vídeo não pôde ser aberto ou lido corretamente."""


@dataclass(frozen=True)
class VideoMetadata:
    """Metadados de vídeo extraídos do contêiner."""

    fps: float
    width: int
    height: int
    total_frames: int


class VideoReader:
    """Mantém a captura aberta e fornece frames com seus timestamps."""

    def __init__(
        self,
        capture: cv2.VideoCapture,
        metadata: VideoMetadata,
        first_frame: np.ndarray,
    ) -> None:
        """Inicializa o leitor preservando o primeiro frame já validado."""
        self._capture = capture
        self.metadata = metadata
        self._first_frame: np.ndarray | None = first_frame
        self._next_frame_idx = 1
        self._cached_frame_idx = 0
        self._cached_frame: np.ndarray | None = first_frame

    def iter_frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Produz índice, timestamp em segundos e imagem de cada frame."""
        frame_idx = 0
        frame = self._first_frame
        self._first_frame = None

        while frame is not None:
            yield frame_idx, frame_idx / self.metadata.fps, frame
            frame_idx += 1
            success, next_frame = self._capture.read()
            frame = next_frame if success else None

    def read_frame(self, frame_idx: int) -> tuple[float, np.ndarray] | None:
        """Busca um frame pelo índice, útil para navegação pausada.

        Args:
            frame_idx: Índice zero-based do frame desejado.

        Returns:
            Timestamp em segundos e imagem, ou None fora do vídeo.
        """
        if frame_idx < 0 or frame_idx >= self.metadata.total_frames:
            return None

        if frame_idx == self._cached_frame_idx and self._cached_frame is not None:
            return frame_idx / self.metadata.fps, self._cached_frame

        if frame_idx != self._next_frame_idx:
            if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
                return None

        success, frame = self._capture.read()
        if not success or frame is None:
            return None
        self._cached_frame_idx = frame_idx
        self._cached_frame = frame
        self._next_frame_idx = frame_idx + 1
        return frame_idx / self.metadata.fps, frame

    def release(self) -> None:
        """Libera o recurso de captura do OpenCV."""
        self._capture.release()


def open_video(video_path: Path) -> VideoReader:
    """Abre um vídeo e valida seus metadados e primeiro frame.

    Args:
        video_path: Caminho do arquivo de vídeo.

    Returns:
        Um leitor pronto para iterar desde o primeiro frame.

    Raises:
        VideoOpenError: Se o arquivo não existir, não abrir ou não decodificar
            nenhum frame, ou se FPS/resolução forem inválidos.
    """
    if not video_path.is_file():
        raise VideoOpenError(f"Arquivo de vídeo não encontrado: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoOpenError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise VideoOpenError(f"FPS inválido nos metadados do vídeo: {fps}")
    if width <= 0 or height <= 0:
        capture.release()
        raise VideoOpenError(
            f"Resolução inválida nos metadados: {width}x{height}"
        )
    if not math.isfinite(total_frames_value) or total_frames_value < 0:
        capture.release()
        raise VideoOpenError(
            f"Total de frames inválido nos metadados: {total_frames_value}"
        )

    success, first_frame = capture.read()
    if not success or first_frame is None:
        capture.release()
        raise VideoOpenError(f"O vídeo não contém frames legíveis: {video_path}")

    metadata = VideoMetadata(
        fps=fps,
        width=width,
        height=height,
        total_frames=int(total_frames_value),
    )
    return VideoReader(capture, metadata, first_frame)