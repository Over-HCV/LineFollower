"""Tests del reconocimiento de octágonos PARE/SIGA."""

from __future__ import annotations

import cv2
import numpy as np

from first.core.types import Senal
from first.core.vision.sign import detectar_senal

from conftest import ROJO, VERDE, frame_con_octagono


def test_octagono_rojo_es_pare() -> None:
    deteccion = detectar_senal(frame_con_octagono(ROJO))
    assert deteccion.senal is Senal.PARE
    assert deteccion.caja is not None
    assert deteccion.area > 500


def test_octagono_verde_es_siga() -> None:
    deteccion = detectar_senal(frame_con_octagono(VERDE))
    assert deteccion.senal is Senal.SIGA


def test_cuadrado_rojo_no_es_senal() -> None:
    frame = frame_con_octagono(ROJO)
    frame[60:120, 100:220] = ROJO  # pinta un cuadrado encima
    deteccion = detectar_senal(frame)
    assert deteccion.senal is None


def test_octagono_pequeno_ignorado() -> None:
    deteccion = detectar_senal(frame_con_octagono(ROJO, radio=8.0))
    assert deteccion.senal is None


def test_pare_gana_a_siga_en_conflicto() -> None:
    frame = frame_con_octagono(ROJO, centro=(100, 80))
    cv2.fillPoly(frame, [np.array([[200, 40], [260, 40], [260, 120], [200, 120]])], VERDE)
    deteccion = detectar_senal(frame)
    assert deteccion.senal is Senal.PARE
