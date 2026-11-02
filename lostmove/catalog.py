"""Lista de vídeos disponíveis e seleção pelo terminal."""

import csv
from collections.abc import Callable, Sequence
from pathlib import Path

from .config import VIDEO_EXTENSIONS


def find_videos(data_dir: Path) -> list[Path]:
    """Encontra vídeos aceitos na pasta de dados e em suas subpastas.

    Args:
        data_dir: Pasta que contém os vídeos de entrada.

    Returns:
        Caminhos dos vídeos em ordem alfabética, relativos à pasta informada.
    """
    if not data_dir.is_dir():
        return []

    extensions = {extension.casefold() for extension in VIDEO_EXTENSIONS}
    return sorted(
        (
            path
            for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in extensions
        ),
        key=lambda path: path.relative_to(data_dir).as_posix().casefold(),
    )


def tested_video_names(csv_path: Path) -> set[str]:
    """Lê da planilha os nomes dos vídeos que já têm resultados.

    Args:
        csv_path: Planilha de resultados existente.

    Returns:
        Nomes relativos dos vídeos que já foram analisados.
    """
    if not csv_path.is_file():
        return set()

    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "vídeo" not in reader.fieldnames:
            return set()
        return {
            row["vídeo"]
            for row in reader
            if row.get("vídeo") and row.get("tipo") == "Ciclo"
        }


def choose_video(
    videos: Sequence[Path],
    data_dir: Path,
    tested_names: set[str],
    input_func: Callable[[str], str] | None = None,
    output_func: Callable[[str], None] | None = None,
) -> Path | None:
    """Mostra o catálogo e pede o número do vídeo ao usuário.

    Args:
        videos: Vídeos encontrados na pasta de entrada.
        data_dir: Pasta-base usada para mostrar caminhos curtos.
        tested_names: Nomes relativos presentes na planilha de resultados.
        input_func: Função que recebe a escolha do usuário.
        output_func: Função que mostra instruções e mensagens.

    Returns:
        Vídeo escolhido, ou None se não houver vídeos ou o usuário sair.
    """
    if input_func is None:
        input_func = input
    if output_func is None:
        output_func = print

    if not videos:
        output_func(f"Nenhum vídeo encontrado em: {data_dir}")
        return None

    output_func("Vídeos disponíveis:")
    for index, video_path in enumerate(videos, start=1):
        video_name = video_path.relative_to(data_dir).as_posix()
        status = "TESTADO" if video_name in tested_names else "NOVO"
        output_func(f"{index}. [{status}] {video_name}")

    while True:
        choice = input_func("Digite o número do vídeo ou 'q' para sair: ").strip()
        if choice.casefold() == "q":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(videos):
            return videos[int(choice) - 1]
        output_func("Escolha inválida. Digite um número da lista ou 'q'.")