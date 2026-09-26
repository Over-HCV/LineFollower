"""Superposición de depuración: ROIs, centroide, objetivo, estado y atención.

Lo interesante para depurar no es el frame, es POR QUÉ el detector eligió un
segmento y no otro: por eso se dibuja la campana de atención (la gaussiana
que pondera los candidatos) junto a su centro y la cortina lateral.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from ..types import Command, DeteccionSenal, LineInfo
from .params import ParamsLinea

ALTURA_CAMPANA: int = 46  # px de alto de la curva dibujada
COLOR_CAMPANA: tuple[int, int, int] = (0, 200, 255)  # ámbar (BGR)
COLOR_CENTRO: tuple[int, int, int] = (0, 140, 255)
COLOR_CORTINA: tuple[int, int, int] = (40, 40, 40)


def dibujar(
    frame: NDArray[np.uint8],
    linea: LineInfo,
    deteccion: DeteccionSenal,
    comando: Command,
    estado: str,
    params: ParamsLinea,
    centro_esperado: float | None = None,
) -> NDArray[np.uint8]:
    """Devuelve una copia del frame con toda la superposición de depuración."""
    lienzo = frame.copy()
    alto, ancho = lienzo.shape[:2]
    y_cerca = int(alto * params.roi_cerca[0])
    y_lejos = int(alto * params.roi_lejos[0])
    cv2.line(lienzo, (0, y_cerca), (ancho, y_cerca), (255, 255, 0), 1)
    cv2.line(lienzo, (0, y_lejos), (ancho, y_lejos), (255, 0, 255), 1)
    cv2.line(lienzo, (ancho // 2, y_lejos), (ancho // 2, alto), (0, 255, 255), 1)

    __dibujar_atencion(lienzo, params, centro_esperado)

    if linea.presente:
        cv2.circle(lienzo, (linea.cx, y_cerca + linea.cy), 6, (0, 255, 0), 2)
        cv2.line(
            lienzo,
            (linea.objetivo_px, y_lejos),
            (linea.objetivo_px, alto),
            (0, 0, 255),
            2,
        )
    if deteccion.caja is not None:
        x, y, w, h = deteccion.caja
        cv2.rectangle(lienzo, (x, y), (x + w, y + h), (0, 255, 255), 2)
        nombre = deteccion.senal.value.upper() if deteccion.senal else ""
        cv2.putText(
            lienzo,
            nombre,
            (x, max(y - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
    cv2.putText(
        lienzo,
        f"{comando.value.upper()} | {estado}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return lienzo


def __dibujar_atencion(
    lienzo: NDArray[np.uint8], params: ParamsLinea, centro_esperado: float | None
) -> None:
    """Campana de atención: cuánto pesa cada columna al elegir el segmento.

    Es exactamente la gaussiana que multiplica la evidencia en
    ``line.__segmento_dominante``, dibujada a escala sobre el frame: donde la
    curva está alta, un candidato compite; donde está pegada al suelo, se
    ignora aunque tenga más masa que la línea buena.
    """
    alto, ancho = lienzo.shape[:2]
    margen = max(0, min(int(params.margen_lateral), ancho // 2 - 20))
    util = ancho - 2 * margen
    if util <= 1:
        return
    centro = ancho / 2.0 if centro_esperado is None else float(centro_esperado)
    sigma = max(1.0, params.sigma_atencion * util)

    if margen:  # la cortina: columnas que ni se miran
        oscuro = lienzo[:, :margen].astype(np.int16) - 60
        lienzo[:, :margen] = np.clip(oscuro, 0, 255).astype(np.uint8)
        oscuro = lienzo[:, ancho - margen :].astype(np.int16) - 60
        lienzo[:, ancho - margen :] = np.clip(oscuro, 0, 255).astype(np.uint8)
        cv2.line(lienzo, (margen, 0), (margen, alto), COLOR_CORTINA, 1)
        cv2.line(
            lienzo, (ancho - margen, 0), (ancho - margen, alto), COLOR_CORTINA, 1
        )

    base = alto - 3
    puntos = [
        (
            x,
            base
            - int(ALTURA_CAMPANA * math.exp(-0.5 * ((x - centro) / sigma) ** 2)),
        )
        for x in range(margen, ancho - margen)
    ]
    cv2.polylines(
        lienzo, [np.array(puntos, dtype=np.int32)], False, COLOR_CAMPANA, 2
    )
    x_centro = int(round(centro))
    if margen <= x_centro < ancho - margen:
        cv2.line(
            lienzo,
            (x_centro, base - ALTURA_CAMPANA - 6),
            (x_centro, base),
            COLOR_CENTRO,
            1,
        )
    cv2.putText(  # esquina inferior derecha: no choca con las líneas de ROI
        lienzo,
        f"atencion {params.sigma_atencion:.2f}",
        (ancho - 118, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        COLOR_CAMPANA,
        1,
    )
