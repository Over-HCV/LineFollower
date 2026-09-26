"""Defectos del suelo y de la luz: manchas de suciedad y reflejos especulares.

Ninguno borra la geometría de la pista: son candidatos falsos (manchas) o
cortes visuales sobre la línea (reflejos) que el detector debe descartar o
cruzar en modo hueco.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .pista import Pista
from .realismo import Realismo


def __lejos_de_signos(i: int, indices_signos: Sequence[int], n: int) -> bool:
    return all(abs((i - i0 + n // 2) % n - n // 2) >= 30 for i0 in indices_signos)


def dibujar_reflejos(
    lienzo: NDArray[np.uint8],
    pista: Pista,
    r: Realismo,
    indices_signos: Sequence[int],
    rng: np.random.Generator,
) -> None:
    """Brillo especular blanco sobre la línea (reflejo de la luz).

    El reflejo no borra la geometría: la línea sigue ahí, pero la cámara
    la ve rota en pedazos. El modo hueco del cerebro debe cruzarla recto.
    """
    if r.reflejos <= 0:
        return
    p = pista.puntos
    n = len(p)
    colocados = 0
    intentos = 0
    while colocados < r.reflejos and intentos < r.reflejos * 20:
        intentos += 1
        i = int(rng.integers(0, n))
        if not __lejos_de_signos(i, indices_signos, n):
            continue
        tx, ty = pista.tangente(i / n)
        largo = float(rng.uniform(r.reflejo_largo_min, r.reflejo_largo_max))
        ancho_r = float(rng.uniform(6.0, r.reflejo_ancho_max))
        brillo = int(rng.uniform(215.0, 242.0))
        color = (brillo, brillo - 2, brillo - 4)  # blanco grisáceo, S baja
        cv2.ellipse(
            lienzo,
            (int(round(p[i, 0])), int(round(p[i, 1]))),
            (int(round(largo / 2.0)), int(round(ancho_r / 2.0))),
            math.degrees(math.atan2(ty, tx)),
            0.0,
            360.0,
            color,
            -1,
        )
        colocados += 1


def dibujar_manchas(
    lienzo: NDArray[np.uint8],
    pista: Pista,
    r: Realismo,
    indices_signos: Sequence[int],
    rng: np.random.Generator,
) -> None:
    """Manchas de suciedad oscuras y desaturadas junto a la línea."""
    if r.manchas <= 0:
        return
    p = pista.puntos
    n = len(p)
    colocadas = 0
    intentos = 0
    while colocadas < r.manchas and intentos < r.manchas * 20:
        intentos += 1
        i = int(rng.integers(0, n))
        if not __lejos_de_signos(i, indices_signos, n):
            continue
        lado = 1 if rng.random() < 0.5 else -1
        lateral = lado * float(rng.uniform(r.mancha_lateral_min, 120.0))
        tx, ty = pista.tangente(i / n)
        cx = p[i, 0] - ty * lateral
        cy = p[i, 1] + tx * lateral
        eje_a = float(rng.uniform(r.mancha_radio_min, r.mancha_radio_max))
        eje_b = eje_a * float(rng.uniform(0.5, 0.9))
        gris = int(rng.uniform(60.0, 110.0))
        color = (gris, gris - int(rng.uniform(0, 8)), gris - int(rng.uniform(0, 8)))
        cv2.ellipse(
            lienzo,
            (int(round(cx)), int(round(cy))),
            (int(round(eje_a)), int(round(eje_b))),
            float(rng.uniform(0.0, 180.0)),
            0.0,
            360.0,
            color,
            -1,
        )
        colocadas += 1
