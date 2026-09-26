"""Bucle OpenCV compartido para webcam y video: frame -> cerebro -> ventana."""

from __future__ import annotations

import cv2

from ..core.brain import Brain
from ..ports import FrameSource


def bucle_cv2(fuente: FrameSource, brain: Brain, nombre_ventana: str = "first") -> None:
    """Muestra el frame con la telemetría del cerebro. 'q' sale, 'r' reinicia."""
    while True:
        frame = fuente.read()
        if frame is None:
            break
        comando, telemetria = brain.procesar(frame)
        vista = telemetria.debug if telemetria.debug is not None else frame
        cv2.imshow(nombre_ventana, vista)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("r"):
            brain.reiniciar()
    fuente.release()
    cv2.destroyAllWindows()
