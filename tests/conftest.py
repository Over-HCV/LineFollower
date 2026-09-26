"""Fixtures compartidas: frames sintéticos y reloj falso para tests deterministas."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from first.adapters.sim.world import vertices_octagono

FONDO: tuple[int, int, int] = (205, 205, 205)
ROJO: tuple[int, int, int] = (0, 0, 210)
VERDE: tuple[int, int, int] = (0, 190, 0)


def frame_con_linea(
    x_linea: int = 160,
    ancho_linea: int = 28,
    ancho: int = 320,
    alto: int = 240,
    color: tuple[int, int, int] = (45, 45, 45),
) -> NDArray[np.uint8]:
    """Frame con una línea vertical oscura que cruza todo el alto."""
    frame = np.full((alto, ancho, 3), FONDO, dtype=np.uint8)
    frame[:, x_linea - ancho_linea // 2 : x_linea + ancho_linea // 2] = color
    return frame


def frame_con_dos_lineas(
    x_principal: int = 160,
    x_vecina: int = 40,
    ancho_principal: int = 26,
    ancho_vecina: int = 34,
    ancho: int = 320,
    alto: int = 240,
) -> NDArray[np.uint8]:
    """Dos líneas verticales: la propia y una vecina más gruesa en el borde.

    Es el caso de la chicane y del ocho: la vecina tiene más masa y la misma
    cobertura vertical, así que sin atención central se la lleva el gato.
    """
    frame = frame_con_linea(x_linea=x_principal, ancho_linea=ancho_principal, ancho=ancho, alto=alto)
    frame[:, x_vecina - ancho_vecina // 2 : x_vecina + ancho_vecina // 2] = (45, 45, 45)
    return frame


def frame_con_octagono(
    color_bgr: tuple[int, int, int],
    centro: tuple[int, int] = (160, 80),
    radio: float = 40.0,
    ancho: int = 320,
    alto: int = 240,
) -> NDArray[np.uint8]:
    """Frame con un octágono del color indicado en la zona superior."""
    frame = np.full((alto, ancho, 3), FONDO, dtype=np.uint8)
    cv2.fillPoly(frame, [vertices_octagono(centro, radio)], color_bgr)
    return frame


def frame_linea_y_octagono(
    color_bgr: tuple[int, int, int], x_linea: int = 160
) -> NDArray[np.uint8]:
    """Línea centrada de guía + señal arriba (escenario típico de la pista)."""
    frame = frame_con_linea(x_linea=x_linea)
    cv2.fillPoly(frame, [vertices_octagono((160, 80), 40.0)], color_bgr)
    return frame


class RelojFalso:
    """Reloj inyectable que avanza manualmente (tests deterministas)."""

    def __init__(self, t0: float = 1000.0) -> None:
        self.t: float = t0

    def __call__(self) -> float:
        return self.t

    def avanzar(self, dt: float) -> None:
        self.t += dt
