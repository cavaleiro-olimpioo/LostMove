"""Exibição OpenCV e controles de reprodução para depuração visual."""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from . import config
from .detector import BallDetection


@dataclass
class PlaybackState:
    """Estado mutável da reprodução e navegação do vídeo."""

    frame_idx: int = 0
    paused: bool = False

    def handle_key(self, key: int, total_frames: int) -> bool:
        """Atualiza estado com a tecla pressionada; False encerra no q."""
        key &= 0xFF
        if key == ord("q"):
            return False
        if key == ord(" "):
            self.paused = not self.paused
            return True

        if self.paused and key == ord("d"):
            self.frame_idx = min(self.frame_idx + 1, max(total_frames - 1, 0))
        elif self.paused and key == ord("a"):
            self.frame_idx = max(self.frame_idx - 1, 0)
        elif not self.paused:
            self.frame_idx += 1
        return True


class VideoViewer:
    """Renderiza vídeo anotado e máscara em janelas OpenCV separadas."""

    ORIGINAL_WINDOW = "Video e posicao da bola"
    MASK_WINDOW = "Imagem usada para encontrar a bola"

    def __init__(self) -> None:
        """Cria janelas redimensionáveis para as duas visualizações."""
        cv2.namedWindow(self.ORIGINAL_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.MASK_WINDOW, cv2.WINDOW_NORMAL)

    @staticmethod
    def render_original(
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        detection: BallDetection | None,
        trail: deque[tuple[int, int]],
        cycle_label: str | None = None,
    ) -> np.ndarray:
        """Desenha trilha, posição, fase do ciclo e dados da detecção."""
        rendered = frame.copy()
        height, width = rendered.shape[:2]

        if len(trail) > 1:
            points = np.asarray(trail, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(rendered, [points], False, (255, 180, 0), 2)

        if detection is None:
            coordinate_text = "posicao: nao encontrada"
        else:
            center_x, center_y = detection.centroid
            point = (round(center_x), round(center_y))
            radius = max(1, round(detection.radius))
            cv2.circle(rendered, point, radius, (0, 255, 0), 2)
            cv2.circle(rendered, point, 2, (0, 0, 255), -1)
            coordinate_text = f"x: {center_x:.2f}, y: {center_y:.2f}"

            if config.CROSSHAIR_ORIENTATION in ("vertical", "both"):
                cv2.line(
                    rendered,
                    (point[0], 0),
                    (point[0], height - 1),
                    (0, 255, 255),
                    1,
                )
            if config.CROSSHAIR_ORIENTATION in ("horizontal", "both"):
                cv2.line(
                    rendered,
                    (0, point[1]),
                    (width - 1, point[1]),
                    (0, 255, 255),
                    1,
                )
            if config.CROSSHAIR_ORIENTATION not in (
                "vertical",
                "horizontal",
                "both",
                "none",
            ):
                raise ValueError("CROSSHAIR_ORIENTATION inválida.")

        cv2.putText(
            rendered,
            f"Quadro: {frame_idx} | Tempo: {timestamp_s:.3f} s | {coordinate_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        if cycle_label is not None:
            cv2.putText(
                rendered,
                cycle_label,
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
        return rendered

    def show(self, original: np.ndarray, mask: np.ndarray) -> None:
        """Apresenta o frame anotado e a máscara do detector."""
        cv2.imshow(self.ORIGINAL_WINDOW, original)
        cv2.imshow(self.MASK_WINDOW, mask)

    @staticmethod
    def wait_key(delay_ms: int) -> int:
        """Aguarda uma tecla pelo intervalo informado em milissegundos."""
        return cv2.waitKey(delay_ms)

    @staticmethod
    def close() -> None:
        """Fecha todas as janelas abertas pelo OpenCV."""
        cv2.destroyAllWindows()