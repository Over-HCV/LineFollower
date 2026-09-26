"""Cámara sintética del carrito: extrae un parche rotado del mundo (vista cenital
hacia adelante), le aplica un warp de perspectiva, gradiente de iluminación y
ruido gaussiano. El resultado es un frame BGR que entra al mismo Brain que
usará la cámara real."""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from .lienzo import COLOR_SUELO
from .world import Carrito, Mundo


class CamaraSintetica:
    """Modelo de cámara inclinada hacia adelante montada al frente del carrito."""

    def __init__(
        self,
        tamaño_frame: tuple[int, int] = (320, 240),  # (ancho, alto) de salida
        parche: tuple[int, int] = (420, 300),  # (ancho, alto) del parche cenital
        cerca: float = 10.0,  # px más cercanos visibles al frente del carrito
        alcance: float = 300.0,  # profundidad total visible
        ruido: float = 5.0,
        semilla: int = 7,
        perspectiva: bool = False,
    ) -> None:
        self.tamaño_frame = tamaño_frame
        self.parche = parche
        self.cerca = cerca
        self.alcance = alcance
        self.ruido = ruido
        self.perspectiva = perspectiva
        self.__rng = np.random.default_rng(semilla)
        alto = tamaño_frame[1]
        self.__gradiente = np.linspace(0.90, 1.0, alto, dtype=np.float32).reshape(
            alto, 1, 1
        )

    def render(self, mundo: Mundo, carrito: Carrito) -> NDArray[np.uint8]:
        """Retorna el frame BGR que 've' el carrito desde su pose actual."""
        wp, hp = self.parche
        d = np.array([math.cos(carrito.theta), math.sin(carrito.theta)])
        der = np.array([-d[1], d[0]])  # lado derecho del robot (y hacia abajo)
        avance = -self.alcance / (hp - 1)

        # Mapa afín (u, v) del parche -> (x, y) del mundo.
        m_fwd = np.array(
            [
                [
                    der[0],
                    d[0] * avance,
                    carrito.x + d[0] * (self.cerca + self.alcance) - der[0] * wp / 2,
                ],
                [
                    der[1],
                    d[1] * avance,
                    carrito.y + d[1] * (self.cerca + self.alcance) - der[1] * wp / 2,
                ],
            ],
            dtype=np.float64,
        )
        m_inv = cv2.invertAffineTransform(m_fwd)
        parche = cv2.warpAffine(
            mundo.lienzo,
            m_inv,
            (wp, hp),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=COLOR_SUELO,  # fuera del mundo se ve piso, no negro
        )

        if self.perspectiva:
            # Opción de realismo extra: trapecio -> rectángulo (comprime las
            # filas cercanas). Por defecto se usa cámara cenital limpia, como
            # la cámara frontal inclinada de un seguidor de línea real.
            src = np.float32(
                [
                    (wp * 0.12, 0.0),
                    (wp * 0.88, 0.0),
                    (wp * 0.98, hp - 1.0),
                    (wp * 0.02, hp - 1.0),
                ]
            )
            dst = np.float32(
                [(0.0, 0.0), (wp - 1.0, 0.0), (wp - 1.0, hp - 1.0), (0.0, hp - 1.0)]
            )
            m_persp = cv2.getPerspectiveTransform(src, dst)
            parche = cv2.warpPerspective(
                parche,
                m_persp,
                (wp, hp),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=COLOR_SUELO,
            )

        frame = cv2.resize(parche, self.tamaño_frame, interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) * self.__gradiente
        frame += self.__rng.normal(0.0, self.ruido, frame.shape).astype(np.float32)
        return np.clip(frame, 0.0, 255.0).astype(np.uint8)
