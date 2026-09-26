"""Reconocimiento de las señales PARE (octágono rojo) y SIGA (octágono verde).

Segmentación por color en HSV (el rojo necesita dos rangos porque el matiz
da la vuelta en 0/179), morfología de limpieza, contornos y verificación
de forma: aproximación poligonal de 7 a 9 lados + circularidad + relación
de aspecto.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from ..types import DeteccionSenal, Senal

Mascara = NDArray[np.uint8]

ROJO_BAJO_1 = np.array([0, 80, 60], dtype=np.uint8)
ROJO_ALTO_1 = np.array([12, 255, 255], dtype=np.uint8)
ROJO_BAJO_2 = np.array([165, 80, 60], dtype=np.uint8)
ROJO_ALTO_2 = np.array([179, 255, 255], dtype=np.uint8)

VERDE_BAJO = np.array([35, 60, 60], dtype=np.uint8)
VERDE_ALTO = np.array([90, 255, 255], dtype=np.uint8)


def __limpiar_mascara(mascara: Mascara) -> Mascara:
    """Apertura (quita motas) + dilatación (cierra huecos pequeños)."""
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    return cv2.dilate(mascara, kernel, iterations=1)


def __mascara_color(hsv: NDArray[np.uint8], senal: Senal) -> Mascara:
    """Segmenta rojo (dos rangos de matiz) o verde."""
    if senal is Senal.PARE:
        m1 = cv2.inRange(hsv, ROJO_BAJO_1, ROJO_ALTO_1)
        m2 = cv2.inRange(hsv, ROJO_BAJO_2, ROJO_ALTO_2)
        mascara = cv2.bitwise_or(m1, m2)
    else:
        mascara = cv2.inRange(hsv, VERDE_BAJO, VERDE_ALTO)
    return __limpiar_mascara(mascara)


def __es_octagono(contorno: NDArray[np.int32], precision: float = 4.0) -> bool:
    """Verifica forma: 7-9 lados y relleno compacto.

    Los rangos de circularidad y aspecto son tolerantes a la distorsión de
    perspectiva de la cámara (el octágono lejano se estira horizontalmente).
    """
    perimetro = cv2.arcLength(contorno, True)
    if perimetro <= 0:
        return False
    aproximacion = cv2.approxPolyDP(contorno, (precision / 100.0) * perimetro, True)
    if not 7 <= len(aproximacion) <= 9:
        return False
    area = cv2.contourArea(contorno)
    if area <= 0:
        return False
    circularidad = 4.0 * np.pi * area / (perimetro**2)  # octágono regular ≈ 0.95
    if not 0.40 <= circularidad <= 1.10:
        return False
    _, _, w, h = cv2.boundingRect(contorno)
    if h == 0:
        return False
    return 0.55 <= w / float(h) <= 2.00


def detectar_senal(
    frame: NDArray[np.uint8], area_minima: float = 500.0
) -> DeteccionSenal:
    """Busca PARE con prioridad (seguridad) y luego SIGA sobre el frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for senal in (Senal.PARE, Senal.SIGA):
        mascara = __mascara_color(hsv, senal)
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidatos = [
            c
            for c in contornos
            if cv2.contourArea(c) >= area_minima and __es_octagono(c)
        ]
        if candidatos:
            mayor = max(candidatos, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(mayor)
            return DeteccionSenal(
                senal=senal, caja=(x, y, w, h), area=cv2.contourArea(mayor)
            )
    return DeteccionSenal(senal=None, caja=None)
