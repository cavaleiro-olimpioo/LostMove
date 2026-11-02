"""Tratamento configurável de frames para destacar objetos em movimento."""

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class Preprocessor:
    """Mantém o estado necessário ao método de fundo selecionado."""

    reference_gray: np.ndarray | None = None
    background_subtractor: cv2.BackgroundSubtractor | None = None

    def __post_init__(self) -> None:
        """Cria MOG2 somente quando esse método foi escolhido."""
        if config.BACKGROUND_METHOD == "mog2":
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=config.MOG2_HISTORY,
                varThreshold=config.MOG2_VARIANCE_THRESHOLD,
                detectShadows=config.MOG2_DETECT_SHADOWS,
            )
        elif config.BACKGROUND_METHOD != "static":
            raise ValueError(
                "BACKGROUND_METHOD deve ser 'static' ou 'mog2'."
            )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Gera uma máscara binária pré-processada a partir de um frame.

        Args:
            frame: Frame BGR ou em escala de cinza.

        Returns:
            Máscara uint8 com valores 0 e 255.
        """
        if frame.size == 0:
            raise ValueError("O frame recebido está vazio.")

        gray = self._to_gray(frame)
        if config.GAUSSIAN_KERNEL_SIZE > 1:
            kernel_size = config.GAUSSIAN_KERNEL_SIZE
            if kernel_size % 2 == 0:
                raise ValueError("GAUSSIAN_KERNEL_SIZE deve ser ímpar.")
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        if config.USE_BACKGROUND_SUBTRACTION:
            mask = self._subtract_background(gray)
        else:
            mask = gray

        if config.USE_OTSU_THRESHOLD:
            _, mask = cv2.threshold(
                mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
        elif config.USE_ADAPTIVE_THRESHOLD:
            block_size = config.ADAPTIVE_BLOCK_SIZE
            if block_size <= 1 or block_size % 2 == 0:
                raise ValueError("ADAPTIVE_BLOCK_SIZE deve ser ímpar e > 1.")
            mask = cv2.adaptiveThreshold(
                mask,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                config.ADAPTIVE_C,
            )
        else:
            _, mask = cv2.threshold(
                mask, config.STATIC_THRESHOLD, 255, cv2.THRESH_BINARY
            )

        if config.USE_HSV_COLOR_FILTER:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.asarray(config.HSV_COLOR_LOWER, dtype=np.uint8)
            upper = np.asarray(config.HSV_COLOR_UPPER, dtype=np.uint8)
            color_mask = cv2.inRange(hsv, lower, upper)
            # A diferença de fundo localiza movimento; a máscara HSV elimina
            # a parede clara sem cortar o brilho especular antes do fechamento.
            mask = cv2.bitwise_and(mask, color_mask)

        mask = self._apply_roi(mask)
        return self._apply_morphology(mask)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Converte BGR em cinza quando habilitado na configuração."""
        if frame.ndim == 2 or not config.USE_GRAYSCALE:
            return frame.copy() if frame.ndim == 2 else cv2.cvtColor(
                frame, cv2.COLOR_BGR2GRAY
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _subtract_background(self, gray: np.ndarray) -> np.ndarray:
        """Subtrai fundo fixo ou aplica o modelo adaptativo MOG2."""
        if config.BACKGROUND_METHOD == "mog2":
            if self.background_subtractor is None:
                raise RuntimeError("O subtrator MOG2 não foi inicializado.")
            return self.background_subtractor.apply(gray)

        if self.reference_gray is None:
            # O primeiro frame não tem a bola; guardá-lo evita que o objeto
            # seja incorporado ao fundo, como pode acontecer com MOG2.
            self.reference_gray = gray.copy()
        if self.reference_gray.shape != gray.shape:
            raise ValueError("Todos os frames devem manter a mesma dimensão.")
        return cv2.absdiff(gray, self.reference_gray)

    @staticmethod
    def _apply_roi(mask: np.ndarray) -> np.ndarray:
        """Remove pixels fora da ROI sem recortar as coordenadas do frame."""
        if config.ROI is None:
            return mask

        x, y, width, height = config.ROI
        frame_height, frame_width = mask.shape[:2]
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("ROI deve ter origem não negativa e dimensões > 0.")
        if x + width > frame_width or y + height > frame_height:
            raise ValueError("ROI excede os limites do frame.")

        roi_mask = np.zeros_like(mask)
        roi_mask[y : y + height, x : x + width] = mask[y : y + height, x : x + width]
        return roi_mask

    @staticmethod
    def _apply_morphology(mask: np.ndarray) -> np.ndarray:
        """Remove ruído pequeno e fecha lacunas antes da detecção."""
        kernel_size = config.MORPHOLOGY_KERNEL_SIZE
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("MORPHOLOGY_KERNEL_SIZE deve ser ímpar e > 0.")

        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        if config.MORPHOLOGY_OPEN_ITERATIONS > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                open_kernel,
                iterations=config.MORPHOLOGY_OPEN_ITERATIONS,
            )
        if config.MORPHOLOGY_CLOSE_ITERATIONS > 0:
            close_size = config.MORPHOLOGY_CLOSE_KERNEL_SIZE
            if close_size <= 0 or close_size % 2 == 0:
                raise ValueError("MORPHOLOGY_CLOSE_KERNEL_SIZE deve ser ímpar e > 0.")
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_size, close_size)
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                close_kernel,
                iterations=config.MORPHOLOGY_CLOSE_ITERATIONS,
            )
        return mask