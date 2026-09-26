"""Geometría de las pistas: poli-línea cerrada muestreada de una spline.

Los puntos de control definen el trazado; la spline Catmull-Rom cerrada los
convierte en una poli-línea densa que es, a la vez, la línea que se dibuja y
la referencia para medir progreso, distancia y tangentes.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from .lienzo import COLOR_LINEA

PUNTOS_CONTROL: list[tuple[float, float]] = [
    (1320.0, 600.0),
    (1200.0, 1000.0),
    (800.0, 1120.0),
    (464.0, 936.0),
    (280.0, 600.0),
    (400.0, 200.0),
    (800.0, 80.0),
    (1136.0, 264.0),
]


def __pista_ovalo() -> list[tuple[float, float]]:
    return [
        (
            800.0 + 520.0 * math.cos(k * 2.0 * math.pi / 12.0),
            600.0 + 470.0 * math.sin(k * 2.0 * math.pi / 12.0),
        )
        for k in range(12)
    ]


def __pista_chicane(lobulos: int = 6, amplitud: float = 40.0) -> list[tuple[float, float]]:
    """Óvalo ondulado: cadena de curvas cóncavas y convexas encadenadas.

    El radio base se modula con una senoidal de ``lobulos`` periodos, así que
    el trazo alterna hacia dentro y hacia fuera sin parar. Es la pista dura:
    los tramos vecinos quedan a ~220px, y la cámara abarca ±210px, o sea que
    casi siempre se ven DOS líneas y el detector tiene que elegir la propia.

    La amplitud está calibrada para que el radio de curvatura mínimo sea
    ~100px: el carrito gira con radio ~59px (PRESETS), así que las curvas son
    exigentes pero tomables. La versión anterior tenía esquinas de 39px, por
    debajo del radio de giro: eran físicamente imposibles.
    """
    rx, ry = 530.0, 410.0
    muestras = lobulos * 8
    puntos: list[tuple[float, float]] = []
    for i in range(muestras):
        angulo = 2.0 * math.pi * i / muestras
        onda = amplitud * math.sin(lobulos * angulo)
        puntos.append(
            (
                800.0 + (rx + onda) * math.cos(angulo),
                600.0 + (ry + onda) * math.sin(angulo),
            )
        )
    return puntos


def __pista_ocho() -> list[tuple[float, float]]:
    """Lemniscata: la línea se cruza a sí misma en el centro (como el ocho)."""
    return [
        (
            800.0 + 560.0 * math.sin(k * 2.0 * math.pi / 16.0),
            600.0 + 430.0 * math.sin(2.0 * k * 2.0 * math.pi / 16.0),
        )
        for k in range(16)
    ]


PISTAS: dict[str, list[tuple[float, float]]] = {
    "cacahuate": PUNTOS_CONTROL,
    "ovalo": __pista_ovalo(),
    "ocho": __pista_ocho(),
    "chicane": __pista_chicane(),
}



class Pista:
    """Poli-línea cerrada muestreada desde la spline de control."""

    def __init__(
        self,
        puntos_control: list[tuple[float, float]] | None = None,
        muestras: int = 560,
        ancho_linea: int = 26,
    ) -> None:
        control = np.asarray(puntos_control or PUNTOS_CONTROL, dtype=np.float64)
        self.puntos: NDArray[np.float64] = self.__catmull_rom_cerrada(control, muestras)
        self.ancho_linea = ancho_linea

    @staticmethod
    def __catmull_rom_cerrada(
        control: NDArray[np.float64], muestras: int
    ) -> NDArray[np.float64]:
        """Spline cerrada Catmull-Rom (Hermite con tangentes cardinales)."""
        n = len(control)
        por_tramo = max(1, muestras // n)
        ts = np.linspace(0.0, 1.0, por_tramo, endpoint=False)
        t2, t3 = ts**2, ts**3
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + ts
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        trozos: list[NDArray[np.float64]] = []
        for i in range(n):
            p0, p1, p2, p3 = (
                control[(i - 1) % n],
                control[i],
                control[(i + 1) % n],
                control[(i + 2) % n],
            )
            m1 = 0.5 * (p2 - p0)
            m2 = 0.5 * (p3 - p1)
            q = (
                p1[None, :] * h00[:, None]
                + m1[None, :] * h10[:, None]
                + p2[None, :] * h01[:, None]
                + m2[None, :] * h11[:, None]
            )
            trozos.append(q)
        return np.concatenate(trozos)

    def indice(self, s: float) -> int:
        return int((s % 1.0) * len(self.puntos))

    def punto(self, s: float) -> tuple[float, float]:
        x, y = self.puntos[self.indice(s)]
        return (float(x), float(y))

    def tangente(self, s: float) -> tuple[float, float]:
        i = self.indice(s)
        anterior = self.puntos[i - 1]
        siguiente = self.puntos[(i + 1) % len(self.puntos)]
        dx, dy = siguiente - anterior
        norma = math.hypot(dx, dy)
        if norma == 0:
            return (1.0, 0.0)
        return (dx / norma, dy / norma)

    def __distancias2(self, p: tuple[float, float]) -> NDArray[np.float64]:
        return (self.puntos[:, 0] - p[0]) ** 2 + (self.puntos[:, 1] - p[1]) ** 2

    def distancia_minima(self, p: tuple[float, float]) -> float:
        return float(np.sqrt(self.__distancias2(p).min()))

    def indice_cercano(self, p: tuple[float, float]) -> int:
        """Índice de la muestra de la pista más cercana al punto."""
        return int(self.__distancias2(p).argmin())

    def fraccion_cercana(self, p: tuple[float, float]) -> float:
        """Fracción de pista (0-1) del punto más cercano: proyecta sobre la línea."""
        return self.indice_cercano(p) / len(self.puntos)

    def normal(self, s: float) -> tuple[float, float]:
        """Normal unitaria a la izquierda de la tangente (lado +1 del trazo)."""
        tx, ty = self.tangente(s)
        return (-ty, tx)

    def dibujar(self, lienzo: NDArray[np.uint8]) -> None:
        cv2.polylines(
            lienzo,
            [self.puntos.astype(np.int32)],
            isClosed=True,
            color=COLOR_LINEA,
            thickness=self.ancho_linea,
        )
