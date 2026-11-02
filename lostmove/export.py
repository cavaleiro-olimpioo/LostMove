"""Exportação CSV dos ciclos e do resumo de amortecimento."""

import csv
from pathlib import Path

from .analysis import MotionAnalysis


CSV_FIELDS = (
    "vídeo",
    "tipo",
    "ciclo",
    "centro do balanço (pixels)",
    "início (quadro)",
    "início (segundos)",
    "ponto mais à direita (quadro)",
    "ponto mais à direita (segundos)",
    "posição à direita (pixels)",
    "passagem pelo centro (quadro)",
    "passagem pelo centro (segundos)",
    "ponto mais à esquerda (quadro)",
    "ponto mais à esquerda (segundos)",
    "posição à esquerda (pixels)",
    "volta ao centro (quadro)",
    "volta ao centro (segundos)",
    "tempo de ida e volta (segundos)",
    "distância à direita (pixels)",
    "distância à esquerda (pixels)",
    "distância média do centro (pixels)",
    "movimento restante estimado (percentual)",
    "mudança desde o ciclo anterior (percentual)",
    "tempo médio de ida e volta (segundos)",
    "tempo do primeiro ciclo (segundos)",
    "tempo do último ciclo (segundos)",
    "mudança média no tempo por ciclo (segundos)",
    "distância média de todos os balanços (pixels)",
    "perda média estimada por volta (percentual)",
    "perda total estimada (percentual)",
    "observação",
)


def export_cycles_csv(
    analysis: MotionAnalysis,
    fps: float,
    output_path: Path,
    video_name: str,
) -> Path:
    """Salva os ciclos e as médias sem apagar resultados de outros vídeos.

    Se o mesmo vídeo já estiver na planilha, seus resultados anteriores são
    trocados pelos novos para evitar linhas duplicadas.

    Args:
        analysis: Ciclos e estatísticas calculados.
        fps: FPS usado para converter frames em segundos.
        output_path: Caminho de destino do CSV.
        video_name: Nome do vídeo relativo à pasta de entrada.

    Returns:
        Caminho absoluto do CSV gerado.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, str]] = []
    if output_path.is_file():
        with output_path.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row.get("vídeo") == video_name:
                    continue
                existing_rows.append(
                    {field: row.get(field, "") or "" for field in CSV_FIELDS}
                )

    new_rows: list[dict[str, object]] = []
    for cycle in analysis.cycles:
        new_rows.append(
            {
                "vídeo": video_name,
                "tipo": "Ciclo",
                "ciclo": cycle.number,
                "centro do balanço (pixels)": analysis.origin_x_px,
                "início (quadro)": cycle.start_frame,
                "início (segundos)": cycle.start_frame / fps,
                "ponto mais à direita (quadro)": cycle.right_extremum_frame,
                "ponto mais à direita (segundos)": cycle.right_extremum_frame / fps,
                "posição à direita (pixels)": cycle.right_extremum_x_px,
                "passagem pelo centro (quadro)": cycle.opposite_origin_frame,
                "passagem pelo centro (segundos)": cycle.opposite_origin_frame / fps,
                "ponto mais à esquerda (quadro)": cycle.left_extremum_frame,
                "ponto mais à esquerda (segundos)": cycle.left_extremum_frame / fps,
                "posição à esquerda (pixels)": cycle.left_extremum_x_px,
                "volta ao centro (quadro)": cycle.end_frame,
                "volta ao centro (segundos)": cycle.end_frame / fps,
                "tempo de ida e volta (segundos)": cycle.duration_s,
                "distância à direita (pixels)": cycle.right_amplitude_px,
                "distância à esquerda (pixels)": cycle.left_amplitude_px,
                "distância média do centro (pixels)": cycle.amplitude_px,
                "movimento restante estimado (percentual)": (
                    cycle.energy_proxy_relative * 100
                ),
                "mudança desde o ciclo anterior (percentual)": (
                    cycle.energy_loss_percent_vs_previous
                ),
            }
        )

    new_rows.append(
        {
            "vídeo": video_name,
            "tipo": "Resumo",
            "centro do balanço (pixels)": analysis.origin_x_px,
            "tempo médio de ida e volta (segundos)": analysis.mean_cycle_time_s,
            "tempo do primeiro ciclo (segundos)": analysis.first_cycle_time_s,
            "tempo do último ciclo (segundos)": analysis.last_cycle_time_s,
            "mudança média no tempo por ciclo (segundos)": (
                analysis.mean_cycle_time_change_s_per_cycle
            ),
            "distância média de todos os balanços (pixels)": (
                analysis.mean_amplitude_px
            ),
            "perda média estimada por volta (percentual)": (
                analysis.mean_energy_loss_percent_per_cycle
            ),
            "perda total estimada (percentual)": (
                analysis.overall_energy_loss_percent
            ),
            "observação": (
                "Perda aproximada calculada pelo tamanho do balanco; "
                "serve para comparar os balancos, nao mede a energia exata. "
                "Tempo negativo indica ciclo mais curto."
            ),
        }
    )

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(new_rows)

    return output_path.resolve()