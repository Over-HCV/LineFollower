"""Tests de la geometría de la pista (proyección de puntos sobre el trazo)."""

from __future__ import annotations

import math

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
