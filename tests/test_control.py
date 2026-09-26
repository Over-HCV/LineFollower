"""Tests del controlador discreto: histéresis y votación."""

from __future__ import annotations

from first.core.control import ControladorLinea
from first.core.types import Command


def test_zona_muerta_avanza() -> None:
    control = ControladorLinea(umbral_entrada=25, umbral_salida=12, ventana=5)
    assert control.decidir(0) is Command.UP
    assert control.decidir(10) is Command.UP
    assert control.decidir(-10) is Command.UP


def test_gira_hacia_la_linea() -> None:
    control = ControladorLinea(umbral_entrada=25, umbral_salida=12, ventana=5)
    assert control.decidir(40) is Command.RIGHT
    control.reiniciar()
    assert control.decidir(-40) is Command.LEFT


def test_histeresis_mantiene_el_giro() -> None:
    control = ControladorLinea(umbral_entrada=25, umbral_salida=12, ventana=5)
    control.decidir(40)
    # Baja el error pero no cruza el umbral de salida: sigue girando.
    assert control.decidir(20) is Command.RIGHT
    assert control.decidir(15) is Command.RIGHT


def test_votacion_filtra_una_lectura_ruidosa() -> None:
    control = ControladorLinea(umbral_entrada=25, umbral_salida=12, ventana=5)
    for _ in range(4):
        control.decidir(0)  # UP, UP, UP, UP
    resultado = control.decidir(40)  # una lectura aislada de RIGHT
    assert resultado is Command.UP  # la mayoría (4/5) manda


def test_reiniciar_borra_el_historial() -> None:
    control = ControladorLinea(umbral_entrada=25, umbral_salida=12, ventana=5)
    control.decidir(40)
    control.reiniciar()
    assert control.decidir(0) is Command.UP
