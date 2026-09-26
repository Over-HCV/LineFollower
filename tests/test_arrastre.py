"""Tests del arrastre: qué se agarra y dónde queda al soltar."""

from __future__ import annotations

import math

from first.adapters.sim.arrastre import Arrastre, ubicacion_signo
from first.adapters.sim.pista import PISTAS, Pista
from first.adapters.sim.world import (
    OFFSET_SIGNO_MAX,
    OFFSET_SIGNO_MIN,
    Mundo,
    Signo,
)
from first.core.types import Senal


def crear_mundo() -> Mundo:
    return Mundo(
        pista=Pista(PISTAS["ovalo"]),
        signos=[Signo(0.20, Senal.PARE, lado=1)],
    )


def test_tomar_nada_en_zona_vacia() -> None:
    mundo = crear_mundo()
    assert not Arrastre().tomar(mundo, (10.0, 10.0))


def test_tomar_y_soltar_el_carrito_lo_pega_a_la_linea() -> None:
    mundo = crear_mundo()
    arrastre = Arrastre()
    assert arrastre.tomar(mundo, (mundo.carrito.x + 20.0, mundo.carrito.y))
    assert arrastre.carrito

    destino = mundo.pista.punto(0.4)
    assert arrastre.soltar(mundo, (destino[0] + 30.0, destino[1] + 30.0))
    assert not arrastre.activo
    # Queda sobre la línea (no donde cayó el mouse) y mirando a la tangente.
    assert mundo.pista.distancia_minima((mundo.carrito.x, mundo.carrito.y)) < 5.0
    tx, ty = mundo.pista.tangente(mundo.pista.fraccion_cercana(
        (mundo.carrito.x, mundo.carrito.y)
    ))
    assert abs(mundo.carrito.theta - math.atan2(ty, tx)) < 0.2


def test_soltar_una_senal_elige_lado_y_separacion() -> None:
    mundo = crear_mundo()
    arrastre = Arrastre()
    assert arrastre.tomar(mundo, mundo.posicion_signo(mundo.signos[0]))
    assert arrastre.signo == 0

    px, py = mundo.pista.punto(0.55)
    nx, ny = mundo.pista.normal(0.55)
    arrastre.soltar(mundo, (px - nx * 70.0, py - ny * 70.0))  # lado opuesto

    signo = mundo.signos[0]
    assert abs(signo.s - 0.55) < 0.02
    assert signo.lado == -1
    assert math.isclose(signo.offset, 70.0, abs_tol=2.0)


def test_separacion_acotada_al_rango_visible() -> None:
    mundo = crear_mundo()
    px, py = mundo.pista.punto(0.1)
    nx, ny = mundo.pista.normal(0.1)
    _, _, lejos = ubicacion_signo(mundo, (px + nx * 500.0, py + ny * 500.0))
    _, _, encima = ubicacion_signo(mundo, (px, py))
    assert lejos == OFFSET_SIGNO_MAX
    assert encima == OFFSET_SIGNO_MIN == 0.0  # se puede dejar sobre la línea


def test_senal_soltada_encima_queda_sobre_la_linea() -> None:
    mundo = crear_mundo()
    arrastre = Arrastre()
    arrastre.tomar(mundo, mundo.posicion_signo(mundo.signos[0]))
    arrastre.soltar(mundo, mundo.pista.punto(0.30))

    signo = mundo.signos[0]
    assert signo.offset == 0.0
    centro = mundo.posicion_signo(signo)
    assert mundo.pista.distancia_minima(centro) < 5.0


def test_la_senal_tiene_prioridad_sobre_el_carrito() -> None:
    mundo = crear_mundo()
    mundo.reposicionar_en_pista(0.20)  # carrito justo bajo la señal
    arrastre = Arrastre()
    arrastre.tomar(mundo, mundo.posicion_signo(mundo.signos[0]))
    assert arrastre.signo == 0 and not arrastre.carrito
