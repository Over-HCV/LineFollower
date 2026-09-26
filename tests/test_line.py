"""Tests de la detección de línea con doble ROI."""

from __future__ import annotations

from first.core.vision.line import ParamsLinea, detectar_linea

from conftest import frame_con_dos_lineas, frame_con_linea


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
    assert con_cortina.mascara is not None
    # la máscara se reporta en coords del frame completo, alineada con cx
    assert con_cortina.mascara.shape[1] == frame.shape[1]


def test_atencion_elige_la_linea_central_y_no_la_vecina() -> None:
    # La vecina del borde es MÁS gruesa (más masa) y cruza toda la franja:
    # sin atención central ganaría, y el robot se iría tras ella.
    linea = detectar_linea(frame_con_dos_lineas(x_principal=160, x_vecina=40))
    assert linea.presente
    assert abs(linea.error_px) <= 5


def test_atencion_sigue_a_la_prediccion() -> None:
    # Si el frame anterior dejó la línea a la izquierda, ahí mira ahora.
    frame = frame_con_dos_lineas(x_principal=160, x_vecina=40)
    linea = detectar_linea(frame, centro_esperado=40.0)
    assert linea.presente
    assert abs(linea.cx - 40) <= 8


def test_sigma_grande_deja_de_discriminar() -> None:
    # Con una atención muy ancha vuelve a mandar la masa: es el control del
    # experimento (y lo que hace el slider al subirlo del todo).
    frame = frame_con_dos_lineas(x_principal=160, x_vecina=40)
    ancha = detectar_linea(frame, ParamsLinea(sigma_atencion=5.0))
    assert ancha.presente
    assert ancha.cx < 100  # se va con la vecina


def test_senal_de_color_no_entra_en_la_mascara() -> None:
    # El PARE rojo (BGR 0,0,210) pesa 63 en gris: sin filtro de saturación
    # el modo umbral lo tomaba por línea y la detección se rompía.
    import cv2
    import numpy as np

    from conftest import ROJO, frame_con_linea

    frame = frame_con_linea(x_linea=160)
    cv2.fillPoly(
        frame,
        [np.array([[60, 200], [100, 180], [140, 200], [140, 235], [60, 235]], np.int32)],
        ROJO,
    )
    for modo in ("kmeans", "umbral"):
        linea = detectar_linea(frame, ParamsLinea(modo=modo))
        assert linea.presente, modo
        assert abs(linea.error_px) <= 5, modo
        assert linea.mascara is not None
        zona_senal = linea.mascara[linea.mascara.shape[0] // 2 :, 60:140]
        assert float((zona_senal > 0).mean()) < 0.05, modo
