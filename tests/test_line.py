"""Tests de la detección de línea con doble ROI."""

from __future__ import annotations

from first.core.vision.line import ParamsLinea, detectar_linea

from conftest import frame_con_linea


def test_linea_centrada_error_cero() -> None:
    linea = detectar_linea(frame_con_linea(x_linea=160))
    assert linea.presente
    assert abs(linea.error_px) <= 3
    assert linea.mascara is not None


def test_linea_a_la_derecha_error_positivo() -> None:
    linea = detectar_linea(frame_con_linea(x_linea=240))
    assert linea.presente
    assert linea.error_px > 60


def test_linea_a_la_izquierda_error_negativo() -> None:
    linea = detectar_linea(frame_con_linea(x_linea=80))
    assert linea.presente
    assert linea.error_px < -60


def test_sin_linea_no_hay_deteccion() -> None:
    from conftest import FONDO
    import numpy as np

    frame = np.full((240, 320, 3), FONDO, dtype=np.uint8)
    linea = detectar_linea(frame)
    assert not linea.presente


def test_modo_umbral_como_fallback() -> None:
    linea = detectar_linea(
        frame_con_linea(x_linea=200), ParamsLinea(modo="umbral")
    )
    assert linea.presente
    assert 30 < linea.error_px < 60
