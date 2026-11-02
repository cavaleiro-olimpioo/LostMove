"""Configurações ajustáveis do analisador de vídeo."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "data" / "video.mp4"
RESULTS_CSV_PATH = PROJECT_ROOT / "results" / "pendulum_cycles.csv"

# ROI calibrada para a faixa percorrida pela bola neste vídeo; use None para
# desativá-la ou ajuste (x, y, largura, altura) para outra gravação.
ROI: tuple[int, int, int, int] | None = (35, 545, 410, 80)
BACKGROUND_METHOD = "static"  # Opções: "static" ou "mog2".
USE_GRAYSCALE = True
GAUSSIAN_KERNEL_SIZE = 5
USE_BACKGROUND_SUBTRACTION = True
STATIC_THRESHOLD = 25
MOG2_HISTORY = 500
MOG2_VARIANCE_THRESHOLD = 32
MOG2_DETECT_SHADOWS = False
USE_OTSU_THRESHOLD = False
USE_ADAPTIVE_THRESHOLD = False
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 5
MORPHOLOGY_KERNEL_SIZE = 3
MORPHOLOGY_OPEN_ITERATIONS = 1
MORPHOLOGY_CLOSE_ITERATIONS = 2
MORPHOLOGY_CLOSE_KERNEL_SIZE = 5
USE_HSV_COLOR_FILTER = True
# A bola prateada é pouco saturada e mais escura que a parede clara; o limite
# superior de V elimina grande parte do fundo e preserva o núcleo do metal.
HSV_COLOR_LOWER = (0, 0, 20)
HSV_COLOR_UPPER = (179, 170, 165)

# Filtros calibrados nos contornos visíveis da bola neste vídeo de 480x854.
BALL_MIN_AREA = 120.0
BALL_MAX_AREA = 450.0
BALL_MIN_CIRCULARITY = 0.55
BALL_MIN_RADIUS_PX = 8.0
BALL_MAX_RADIUS_PX = 16.0
BALL_MIN_ASPECT_RATIO = 0.55
BALL_MAX_ASPECT_RATIO = 1.60
BALL_MIN_SOLIDITY = 0.75

# Visualização e controles da Etapa 4.
TRAIL_LENGTH = 50
CROSSHAIR_ORIENTATION = "both"  # Opções: "vertical", "horizontal", "both", "none".

# Análise do sinal horizontal. O eixo central foi calibrado pelo ponto médio
# dos extremos observados neste clipe e deve ser revisto para outra câmera.
OSCILLATION_ORIGIN_X_PX = 250.0
ORIGIN_HYSTERESIS_PX = 40.0
MAX_INTERPOLATION_GAP_FRAMES = 8