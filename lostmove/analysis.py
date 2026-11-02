"""Análise de ciclos e amortecimento a partir do sinal horizontal x(t)."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ZeroCrossing:
    """Passagem interpolada pela origem e seu sentido de movimento."""

    frame: float
    direction: int


@dataclass(frozen=True)
class OscillationCycle:
    """Medidas temporais, extremos e energia relativa de um ciclo completo."""

    number: int
    start_frame: float
    right_extremum_frame: int
    right_extremum_x_px: float
    opposite_origin_frame: float
    left_extremum_frame: int
    left_extremum_x_px: float
    end_frame: float
    duration_s: float
    right_amplitude_px: float
    left_amplitude_px: float
    amplitude_px: float
    energy_proxy_relative: float
    energy_loss_percent_vs_previous: float | None


@dataclass(frozen=True)
class MotionAnalysis:
    """Resultado dos ciclos e estatísticas médias do movimento."""

    origin_x_px: float
    cycles: tuple[OscillationCycle, ...]
    mean_cycle_time_s: float
    first_cycle_time_s: float
    last_cycle_time_s: float
    mean_cycle_time_change_s_per_cycle: float
    mean_amplitude_px: float
    mean_energy_loss_percent_per_cycle: float
    overall_energy_loss_percent: float


def interpolate_short_gaps(
    x_positions: Sequence[float | None], max_gap_frames: int
) -> np.ndarray:
    """Interpola somente lacunas curtas entre duas detecções válidas.

    Args:
        x_positions: Coordenada x por frame; None representa falha de detecção.
        max_gap_frames: Maior lacuna interna que pode ser interpolada.

    Returns:
        Array float com lacunas curtas preenchidas e outras como NaN.
    """
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames não pode ser negativo.")
    values = np.asarray(
        [np.nan if value is None else float(value) for value in x_positions],
        dtype=float,
    )
    valid_indices = np.flatnonzero(np.isfinite(values))
    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        gap_size = right - left - 1
        if 0 < gap_size <= max_gap_frames:
            values[left + 1 : right] = np.linspace(
                values[left], values[right], gap_size + 2
            )[1:-1]
    return values


def find_origin_crossings(
    x_positions: np.ndarray,
    origin_x_px: float,
    hysteresis_px: float,
) -> list[ZeroCrossing]:
    """Encontra passagens confirmadas pela origem com histerese.

    A histerese exige que a bola alcance uma faixa lateral antes de validar
    o cruzamento, evitando ciclos falsos causados por ruído junto ao centro.

    Args:
        x_positions: Coordenada x interpolada por frame; NaN quebra a sequência.
        origin_x_px: Posição horizontal de equilíbrio.
        hysteresis_px: Distância mínima para confirmar que a bola mudou de lado.

    Returns:
        Cruzamentos subframe, com direção +1 para o lado direito e -1 para o
        lado esquerdo.
    """
    if hysteresis_px <= 0:
        raise ValueError("hysteresis_px deve ser maior que zero.")

    crossings: list[ZeroCrossing] = []
    confirmed_side = 0
    pending: ZeroCrossing | None = None
    previous: tuple[int, float] | None = None

    for frame_idx, x_value in enumerate(x_positions):
        if not math.isfinite(float(x_value)):
            previous = None
            pending = None
            continue

        delta = float(x_value) - origin_x_px
        side = 1 if delta > 0 else -1 if delta < 0 else 0
        threshold_side = 1 if delta >= hysteresis_px else (
            -1 if delta <= -hysteresis_px else 0
        )

        if confirmed_side == 0:
            if threshold_side != 0:
                confirmed_side = threshold_side
            previous = (frame_idx, delta)
            continue

        if pending is not None:
            if threshold_side == confirmed_side:
                pending = None
            elif threshold_side == pending.direction:
                crossings.append(pending)
                confirmed_side = pending.direction
                pending = None
        elif previous is not None:
            previous_delta = previous[1]
            crossed_between_frames = previous_delta * delta < 0
            crossed_on_current_frame = delta == 0 and previous_delta != 0
            continued_from_origin = previous_delta == 0 and delta != 0
            if (
                crossed_between_frames
                or crossed_on_current_frame
                or continued_from_origin
            ):
                crossing_delta = delta if delta != 0 else previous_delta
                direction = 1 if crossing_delta > 0 else -1
                if direction == -confirmed_side:
                    if crossed_between_frames:
                        fraction = abs(previous_delta) / (
                            abs(previous_delta) + abs(delta)
                        )
                        crossing_frame = previous[0] + fraction * (
                            frame_idx - previous[0]
                        )
                    elif crossed_on_current_frame:
                        crossing_frame = float(frame_idx)
                    else:
                        crossing_frame = float(previous[0])
                    pending = ZeroCrossing(crossing_frame, direction)

        previous = (frame_idx, delta)

    return crossings


def analyze_motion(
    x_positions: Sequence[float | None],
    fps: float,
    origin_x_px: float,
    hysteresis_px: float,
    max_gap_frames: int,
) -> MotionAnalysis:
    """Calcula ciclos completos e perda média de energia relativa.

    A energia é estimada por uma proxy proporcional a amplitude ao quadrado,
    adequada para comparar ciclos no mesmo pêndulo em pequenas oscilações.
    Sem massa e geometria do pêndulo, o resultado não tem unidade física.

    Args:
        x_positions: Coordenada horizontal por frame; None indica ausência.
        fps: Quadros por segundo do vídeo.
        origin_x_px: Coordenada horizontal da posição de equilíbrio.
        hysteresis_px: Limiar de confirmação de lado em pixels.
        max_gap_frames: Limite para interpolar falhas de detecção curtas.
    Returns:
        Todos os ciclos completos medidos e as estatísticas médias.

    Raises:
        ValueError: Se a configuração for inválida ou nenhum ciclo completo
            for encontrado.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps deve ser um valor positivo e finito.")
    values = interpolate_short_gaps(x_positions, max_gap_frames)
    crossings = find_origin_crossings(values, origin_x_px, hysteresis_px)
    raw_cycles: list[tuple[float, int, float, int, float, float, float]] = []

    for index in range(len(crossings) - 2):
        start, middle, end = crossings[index : index + 3]
        if (start.direction, middle.direction, end.direction) != (1, -1, 1):
            continue

        right_start = max(0, math.ceil(start.frame))
        right_end = min(len(values) - 1, math.floor(middle.frame))
        left_start = max(0, math.ceil(middle.frame))
        left_end = min(len(values) - 1, math.floor(end.frame))
        if right_end < right_start or left_end < left_start:
            continue

        right_segment = values[right_start : right_end + 1]
        left_segment = values[left_start : left_end + 1]
        if not np.isfinite(right_segment).any() or not np.isfinite(left_segment).any():
            continue

        right_extremum = right_start + int(np.nanargmax(right_segment))
        left_extremum = left_start + int(np.nanargmin(left_segment))
        right_x = float(values[right_extremum])
        left_x = float(values[left_extremum])
        raw_cycles.append(
            (
                start.frame,
                right_extremum,
                right_x,
                middle.frame,
                left_extremum,
                left_x,
                end.frame,
            )
        )

    if not raw_cycles:
        raise ValueError("Não foram encontrados ciclos completos no vídeo.")
    selected = raw_cycles
    amplitudes = [
        (abs(item[2] - origin_x_px) + abs(item[5] - origin_x_px)) / 2
        for item in selected
    ]
    initial_amplitude = amplitudes[0]
    if initial_amplitude <= 0:
        raise ValueError("A amplitude inicial não pode ser zero.")

    cycles: list[OscillationCycle] = []
    for index, (raw, amplitude) in enumerate(zip(selected, amplitudes)):
        start_frame, right_frame, right_x, middle_frame, left_frame, left_x, end_frame = raw
        energy_relative = (amplitude / initial_amplitude) ** 2
        previous_energy = (
            cycles[-1].energy_proxy_relative if cycles else None
        )
        energy_loss = (
            None
            if previous_energy is None
            else (previous_energy - energy_relative) / previous_energy * 100
        )
        cycles.append(
            OscillationCycle(
                number=index + 1,
                start_frame=start_frame,
                right_extremum_frame=right_frame,
                right_extremum_x_px=right_x,
                opposite_origin_frame=middle_frame,
                left_extremum_frame=left_frame,
                left_extremum_x_px=left_x,
                end_frame=end_frame,
                duration_s=(end_frame - start_frame) / fps,
                right_amplitude_px=abs(right_x - origin_x_px),
                left_amplitude_px=abs(left_x - origin_x_px),
                amplitude_px=amplitude,
                energy_proxy_relative=energy_relative,
                energy_loss_percent_vs_previous=energy_loss,
            )
        )

    overall_loss = (
        1 - cycles[-1].energy_proxy_relative / cycles[0].energy_proxy_relative
    ) * 100
    first_cycle_time = cycles[0].duration_s
    last_cycle_time = cycles[-1].duration_s
    mean_cycle_time_change = (
        0.0
        if len(cycles) == 1
        else (last_cycle_time - first_cycle_time) / (len(cycles) - 1)
    )
    mean_energy_loss = (
        0.0
        if len(cycles) == 1
        else (
            1
            - (
                cycles[-1].energy_proxy_relative
                / cycles[0].energy_proxy_relative
            )
            ** (1 / (len(cycles) - 1))
        )
        * 100
    )
    return MotionAnalysis(
        origin_x_px=origin_x_px,
        cycles=tuple(cycles),
        mean_cycle_time_s=float(np.mean([cycle.duration_s for cycle in cycles])),
        first_cycle_time_s=first_cycle_time,
        last_cycle_time_s=last_cycle_time,
        mean_cycle_time_change_s_per_cycle=mean_cycle_time_change,
        mean_amplitude_px=float(np.mean([cycle.amplitude_px for cycle in cycles])),
        mean_energy_loss_percent_per_cycle=float(mean_energy_loss),
        overall_energy_loss_percent=overall_loss,
    )