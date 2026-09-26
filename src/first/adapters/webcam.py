"""Adaptador de webcam: cv2.VideoCapture como FrameSource."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from ..ports import FrameSource


class WebcamSource(FrameSource):
    """Cámara local indexada; para ensayar el cerebro con visión real."""

    def __init__(self, indice: int = 0, ancho: int = 640, alto: int = 480) -> None:
        self.__cap = cv2.VideoCapture(indice)
        if not self.__cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la camara {indice}")
        self.__cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        self.__cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)

    def read(self) -> NDArray[np.uint8] | None:
        ok, frame = self.__cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.__cap.release()
