"""Constantes del lienzo del mundo simulado (tamaño y paleta del trazo).

Módulo hoja: lo importan la pista, el realismo, el mundo y la cámara sin
crear ciclos entre ellos.
"""

from __future__ import annotations

LIENZO_ANCHO: int = 1600
LIENZO_ALTO: int = 1200

COLOR_SUELO: tuple[int, int, int] = (205, 205, 205)
COLOR_LINEA: tuple[int, int, int] = (40, 40, 40)
COLOR_LINEA_CLARA: tuple[int, int, int] = (98, 94, 90)  # marcador descargado
