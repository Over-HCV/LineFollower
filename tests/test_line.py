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


def test_cortina_oculta_la_linea_del_borde() -> None:
    # Línea pegada al borde izquierdo: con 60px de cortina deja de existir
    # para el detector (es el tramo vecino de la pista, no el que se sigue).
    frame = frame_con_linea(x_linea=40)
    assert detectar_linea(frame).presente
    assert not detectar_linea(frame, ParamsLinea(margen_lateral=60)).presente


def test_cortina_no_desplaza_el_error() -> None:
    # Las coordenadas vuelven al frame completo: el error no cambia de origen.
    frame = frame_con_linea(x_linea=200)
    sin_cortina = detectar_linea(frame)
    con_cortina = detectar_linea(frame, ParamsLinea(margen_lateral=40))
    assert con_cortina.presente
    assert abs(con_cortina.error_px - sin_cortina.error_px) <= 3
    assert abs(con_cortina.cx - sin_cortina.cx) <= 3
