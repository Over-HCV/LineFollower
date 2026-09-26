"""Adaptador de archivo de video: reproduce en bucle los ensayos del docente."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ..ports import FrameSource


class VideoFileSource(FrameSource):
    """Video que se reinicia al terminar, para pruebas repetibles."""

    def __init__(self, ruta: str | Path) -> None:
        self.__cap = cv2.VideoCapture(str(ruta))
        if not self.__cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {ruta}")

    def read(self) -> NDArray[np.uint8] | None:
        ok, frame = self.__cap.read()
        if not ok:
            self.__cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.__cap.read()
            if not ok:
                return None
        return frame

    def release(self) -> None:
        self.__cap.release()
