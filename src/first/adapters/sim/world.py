"""Mundo simulado: pista cerrada (spline Catmull-Rom), señales y carrito.

La física es deliberadamente simple: cada comando discreto es un preset
(velocidad, giro) de un modelo tipo diferencial que gira mientras avanza.
La misma tabla de presets es la que se mapearía al Arduino real.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from ...core.types import Command, Senal

LIENZO_ANCHO: int = 1600
LIENZO_ALTO: int = 1200

COLOR_SUELO: tuple[int, int, int] = (205, 205, 205)
COLOR_LINEA: tuple[int, int, int] = (40, 40, 40)
COLOR_SENAL: dict[Senal, tuple[int, int, int]] = {
    Senal.PARE: (0, 0, 210),  # BGR
    Senal.SIGA: (0, 190, 0),
}

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

# (velocidad en px/s, giro en rad/s): girar avanzando.
PRESETS: dict[Command, tuple[float, float]] = {
    Command.UP: (115.0, 0.0),
    Command.LEFT: (95.0, -1.6),
    Command.RIGHT: (95.0, 1.6),
    Command.DOWN: (0.0, 0.0),
}


@dataclass(frozen=True, slots=True)
class Signo:
    """Señal pintada sobre el suelo junto a la línea."""

    s: float  # posición a lo largo de la pista (fracción 0-1)
    tipo: Senal
    lado: int  # +1 o -1 (a qué lado de la línea)
    offset: float = 70.0
    radio: float = 42.0


SIGNOS_BASE: list[Signo] = [
    Signo(0.22, Senal.PARE, lado=1),
    Signo(0.30, Senal.SIGA, lado=1),
    Signo(0.60, Senal.PARE, lado=-1),
    Signo(0.68, Senal.SIGA, lado=-1),
]


@dataclass(slots=True)
class Carrito:
    """Pose del carrito en coordenadas del mundo (y hacia abajo)."""

    x: float
    y: float
    theta: float


def vertices_octagono(
    centro: tuple[float, float], radio: float
) -> NDArray[np.int32]:
    """Octágono regular (vértices desfasados para quedar 'plano' arriba)."""
    angulos = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False) + np.pi / 8.0
    xs = centro[0] + radio * np.cos(angulos)
    ys = centro[1] + radio * np.sin(angulos)
    return np.stack([xs, ys], axis=1).astype(np.int32)


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

    def distancia_minima(self, p: tuple[float, float]) -> float:
        return float(
            np.sqrt(
                ((self.puntos[:, 0] - p[0]) ** 2 + (self.puntos[:, 1] - p[1]) ** 2).min()
            )
        )

    def dibujar(self, lienzo: NDArray[np.uint8]) -> None:
        cv2.polylines(
            lienzo,
            [self.puntos.astype(np.int32)],
            isClosed=True,
            color=COLOR_LINEA,
            thickness=self.ancho_linea,
        )


class Mundo:
    """Pista + señales + carrito + métricas que espejan la rúbrica."""

    def __init__(
        self,
        pista: Pista | None = None,
        signos: list[Signo] | None = None,
        umbral_descarrilamiento: float = 42.0,
    ) -> None:
        self.pista = pista or Pista()
        self.signos = signos if signos is not None else list(SIGNOS_BASE)
        self.umbral_descarrilamiento = umbral_descarrilamiento

        x0, y0 = self.pista.punto(0.0)
        tx, ty = self.pista.tangente(0.0)
        self.carrito = Carrito(x0, y0, math.atan2(ty, tx))
        self.lienzo = self.__renderizar()

        self.t: float = 0.0
        self.vueltas: int = 0
        self.descarrilamientos: int = 0
        self.__s_prev: float = self.progreso()
        self.__descarrilado: bool = False
        self.__direccion: float = 1.0

    @staticmethod
    def __normalizar(angulo: float) -> float:
        return (angulo + math.pi) % (2.0 * math.pi) - math.pi

    def __posicion_signo(self, signo: Signo) -> tuple[float, float]:
        px, py = self.pista.punto(signo.s)
        tx, ty = self.pista.tangente(signo.s)
        nx, ny = -ty, tx  # normal a la línea
        return (
            px + nx * (signo.lado * signo.offset),
            py + ny * (signo.lado * signo.offset),
        )

    def __renderizar(self) -> NDArray[np.uint8]:
        lienzo = np.full((LIENZO_ALTO, LIENZO_ANCHO, 3), COLOR_SUELO, dtype=np.uint8)
        self.pista.dibujar(lienzo)
        for signo in self.signos:
            centro = self.__posicion_signo(signo)
            cv2.fillPoly(
                lienzo,
                [vertices_octagono(centro, signo.radio)],
                color=COLOR_SENAL[signo.tipo],
            )
        return lienzo

    def progreso(self) -> float:
        """Fracción de pista (0-1) más cercana al carrito."""
        distancias = (
            (self.pista.puntos[:, 0] - self.carrito.x) ** 2
            + (self.pista.puntos[:, 1] - self.carrito.y) ** 2
        )
        return float(distancias.argmin()) / len(self.pista.puntos)

    def paso(self, comando: Command, dt: float) -> None:
        """Integra el preset del comando durante dt y actualiza métricas."""
        velocidad, giro = PRESETS[comando]
        self.carrito.theta = self.__normalizar(self.carrito.theta + giro * dt)
        self.carrito.x += velocidad * math.cos(self.carrito.theta) * dt
        self.carrito.y += velocidad * math.sin(self.carrito.theta) * dt
        self.t += dt

        s = self.progreso()
        cerca_de_pista = self.pista.distancia_minima(
            (self.carrito.x, self.carrito.y)
        ) < 150.0
        # Solo cuenta vueltas recorridas hacia adelante (evita contar el
        # ping-pong de un robot que media-vuelta).
        delta = s - self.__s_prev
        if delta < -0.5:
            delta += 1.0
        elif delta > 0.5:
            delta -= 1.0
        if abs(delta) > 1e-9:
            self.__direccion = delta
        if self.__s_prev > 0.9 and s < 0.1 and cerca_de_pista and self.__direccion > 0:
            self.vueltas += 1
        self.__s_prev = s

        fuera = self.pista.distancia_minima(
            (self.carrito.x, self.carrito.y)
        ) > self.umbral_descarrilamiento
        if fuera and not self.__descarrilado:
            self.descarrilamientos += 1
        self.__descarrilado = fuera
