"""Detecção da bola em uma máscara binária pré-processada."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class BallDetection:
    """Resultado de uma detecção de bola em coordenadas do frame."""

    centroid: tuple[float, float]
    area: float
    circularity: float
    radius: float
    aspect_ratio: float
    solidity: float


def detect_ball(mask: np.ndarray) -> BallDetection | None:
    """Seleciona o contorno mais circular que satisfaz os filtros.

    Args:
        mask: Máscara binária uint8 produzida pelo pré-processamento.

    Returns:
        Detecção com centroide subpixel, ou None se não houver candidato.
    """
    if mask.ndim != 2:
        raise ValueError("A máscara precisa ser uma imagem de um canal.")

    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates: list[BallDetection] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if not config.BALL_MIN_AREA <= area <= config.BALL_MAX_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < config.BALL_MIN_CIRCULARITY:
            continue

        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        if not config.BALL_MIN_RADIUS_PX <= radius <= config.BALL_MAX_RADIUS_PX:
            continue

        _, _, width, height = cv2.boundingRect(contour)
        if height == 0:
            continue
        aspect_ratio = width / height
        if not (
            config.BALL_MIN_ASPECT_RATIO
            <= aspect_ratio
            <= config.BALL_MAX_ASPECT_RATIO
        ):
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < config.BALL_MIN_SOLIDITY:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        centroid = (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )
        candidates.append(
            BallDetection(
                centroid=centroid,
                area=area,
                circularity=circularity,
                radius=radius,
                aspect_ratio=aspect_ratio,
                solidity=solidity,
            )
        )

    if not candidates:
        return None

    # Circularidade, compacidade e escala favorecem a esfera sobre suportes,
    # bordas da mesa e restos de sombra que sobreviveram à máscara.
    return max(
        candidates,
        key=lambda item: (item.circularity, item.solidity, -abs(item.area - 200)),
    )