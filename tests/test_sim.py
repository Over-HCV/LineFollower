"""Tests del mundo simulado y de la carrera headless end-to-end."""

from __future__ import annotations

import math

import numpy as np

from first.adapters.sim.camera import CamaraSintetica
from first.adapters.sim.carrera import simular_carrera
from first.adapters.sim.pista import PISTAS, Pista
from first.adapters.sim.realismo import REALISMOS
from first.adapters.sim.world import Mundo
from first.core.brain import Brain
from first.core.types import Command


def test_pista_cerrada_continua() -> None:
    pista = Pista()
    assert len(pista.puntos) > 100
    inicio = pista.punto(0.0)
    casi_cierre = pista.punto(0.999)
    # El punto final y el inicial están cerca (curva cerrada).
    distancia = math.dist(inicio, casi_cierre)
    assert distancia < 30.0


def test_fisica_girar_avanzando() -> None:
    mundo = Mundo()
    x0, y0, t0 = mundo.carrito.x, mundo.carrito.y, mundo.carrito.theta
    mundo.paso(Command.UP, 1.0)
    assert math.isclose(mundo.carrito.theta, t0, abs_tol=1e-9)
    assert math.dist((x0, y0), (mundo.carrito.x, mundo.carrito.y)) > 100.0

    mundo.paso(Command.LEFT, 1.0)
    assert mundo.carrito.theta < t0  # giro antihorario (izquierda)


def test_camara_sintetica_produce_frame_valido() -> None:
    mundo = Mundo()
    camara = CamaraSintetica()
    frame = camara.render(mundo, mundo.carrito)
    assert frame.shape == (240, 320, 3)
    assert frame.dtype.name == "uint8"
    # La línea oscura debe estar presente en la parte baja del frame.
    import numpy as np

    zona_inferior = frame[180:240, :]
    assert zona_inferior.mean() < 220.0


def test_carrera_completa_sigue_la_pista() -> None:
    mundo = Mundo()
    brain = Brain(segundos_pare=1.0, reloj=lambda: mundo.t, dibujar_debug=False)
    metricas = simular_carrera(brain, mundo, duracion=45.0, dt=1.0 / 60.0)
    assert metricas.descarrilamientos <= 1
    assert metricas.vueltas >= 1
    assert metricas.pares_cumplidos >= 1


def test_cambiar_realismo_conserva_pose_y_metricas() -> None:
    mundo = Mundo(pista=Pista(PISTAS["ovalo"]), realismo=REALISMOS["perfecto"])
    for _ in range(60):
        mundo.paso(Command.UP, 1.0 / 60.0)
    pose = (mundo.carrito.x, mundo.carrito.y, mundo.carrito.theta)
    t_previo, vueltas = mundo.t, mundo.vueltas
    lienzo_previo = mundo.lienzo.copy()

    mundo.cambiar_realismo(REALISMOS["alto"])

    assert (mundo.carrito.x, mundo.carrito.y, mundo.carrito.theta) == pose
    assert mundo.t == t_previo and mundo.vueltas == vueltas
    assert not np.array_equal(mundo.lienzo, lienzo_previo)  # sí se repinta


def test_mover_signo_repinta_el_lienzo() -> None:
    mundo = Mundo(pista=Pista(PISTAS["ovalo"]))
    lienzo_previo = mundo.lienzo.copy()
    signo = mundo.mover_signo(0, 0.45, -1, 80.0)
    assert signo.s == 0.45 and signo.lado == -1
    assert not np.array_equal(mundo.lienzo, lienzo_previo)


def test_reposicionar_en_pista_no_inventa_vueltas() -> None:
    mundo = Mundo(pista=Pista(PISTAS["ovalo"]))
    mundo.reposicionar_en_pista(0.95)
    for _ in range(30):
        mundo.paso(Command.UP, 1.0 / 60.0)
    assert mundo.descarrilamientos == 0
    assert mundo.vueltas <= 1  # solo la real, al cruzar la meta hacia adelante
