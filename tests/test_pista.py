"""Tests de la geometría de la pista (proyección de puntos sobre el trazo)."""

from __future__ import annotations

import math

import numpy as np

from first.adapters.sim.pista import PISTAS, Pista


def test_fraccion_cercana_ida_y_vuelta() -> None:
    pista = Pista()
    for s in (0.0, 0.17, 0.5, 0.83):
        punto = pista.punto(s)
        assert abs(pista.fraccion_cercana(punto) - s) < 0.01


def test_fraccion_cercana_proyecta_punto_fuera_de_la_pista() -> None:
    pista = Pista(PISTAS["ovalo"])
    s = 0.25
    px, py = pista.punto(s)
    nx, ny = pista.normal(s)
    fuera = (px + nx * 80.0, py + ny * 80.0)  # 80px al lado de la línea
    assert abs(pista.fraccion_cercana(fuera) - s) < 0.02
    assert math.isclose(pista.distancia_minima(fuera), 80.0, rel_tol=0.05)


def test_normal_es_perpendicular_y_unitaria() -> None:
    pista = Pista()
    tx, ty = pista.tangente(0.3)
    nx, ny = pista.normal(0.3)
    assert abs(tx * nx + ty * ny) < 1e-9
    assert math.isclose(math.hypot(nx, ny), 1.0, rel_tol=1e-9)


def test_chicane_es_dura_pero_seguible() -> None:
    """La chicane debe encadenar curvas cerradas... sin pasarse del límite.

    El carrito gira con radio ~59px (PRESETS: 95px/s ÷ 1.6rad/s). Con curvas
    de ese radio exacto no completa ni una vuelta (medido), así que se exige
    margen: radio mínimo entre 70 y 110px, y tramos vecinos separados lo
    bastante para que se distingan en la máscara aunque se vean a la vez.
    """
    puntos = Pista(PISTAS["chicane"]).puntos
    n = len(puntos)

    k = 6
    a, b, c = np.roll(puntos, k, axis=0), puntos, np.roll(puntos, -k, axis=0)
    lados = [
        np.linalg.norm(b - a, axis=1),
        np.linalg.norm(c - b, axis=1),
        np.linalg.norm(a - c, axis=1),
    ]
    area = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]) / 2.0
    radio = lados[0] * lados[1] * lados[2] / (4.0 * np.maximum(area, 1e-9))
    assert 70.0 <= radio.min() <= 110.0

    distancias = np.linalg.norm(puntos[:, None, :] - puntos[None, :, :], axis=2)
    indices = np.arange(n)
    separacion = np.minimum(
        np.abs(indices[:, None] - indices[None, :]),
        n - np.abs(indices[:, None] - indices[None, :]),
    )
    assert distancias[separacion > n // 12].min() >= 170.0


def test_cacahuate_tiene_cintura() -> None:
    # Era una elipse, indistinguible del óvalo: ahora debe estrecharse en el
    # medio (dos lóbulos) y ser más ancha que alta.
    puntos = Pista(PISTAS["cacahuate"]).puntos
    ancho_total = puntos[:, 0].max() - puntos[:, 0].min()
    alto_total = puntos[:, 1].max() - puntos[:, 1].min()
    centrales = np.abs(puntos[:, 0] - 800.0) < 60.0
    alto_cintura = puntos[centrales, 1].max() - puntos[centrales, 1].min()
    assert ancho_total > alto_total
    assert alto_cintura < 0.75 * alto_total  # cintura clara respecto a los lóbulos
