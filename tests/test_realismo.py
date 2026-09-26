"""Tests del realismo de la pista y del modo hueco (trazo discontinuo)."""

from __future__ import annotations

import math

import numpy as np

from first.adapters.sim.camera import CamaraSintetica
from first.adapters.sim.world import (
    PISTAS,
    REALISMOS,
    Carrito,
    Mundo,
    Pista,
    Realismo,
)
from first.core.brain import Brain
from first.core.types import Command, EstadoRobot
from first.core.vision.line import detectar_linea

from conftest import RelojFalso, frame_con_linea


def crear_brain(reloj: RelojFalso) -> Brain:
    return Brain(
        segundos_pare=2.0,
        umbral_hueco=40.0,
        tolerancia_hueco=0.5,
        reloj=reloj,
        dibujar_debug=False,
    )


def test_hueco_corto_se_cruza_recto() -> None:
    reloj = RelojFalso()
    brain = crear_brain(reloj)
    for _ in range(5):  # centrado y avanzando
        comando, _ = brain.procesar(frame_con_linea(160))
        reloj.avanzar(0.05)
    assert comando is Command.UP

    vacio = frame_con_linea(160)
    vacio[:] = 205  # la línea desaparece (hueco del trazo)
    for _ in range(6):  # 0.30s < tolerancia 0.5s: dead-reckoning recto
        comando, telemetria = brain.procesar(vacio)
        reloj.avanzar(0.05)
    assert telemetria.estado is EstadoRobot.HUECO
    assert comando is Command.UP  # sigue recto, no gira

    comando, telemetria = brain.procesar(frame_con_linea(160))  # reaparece
    assert telemetria.estado is EstadoRobot.SIGUIENDO
    assert comando is Command.UP


def test_hueco_largo_termina_recuperando() -> None:
    reloj = RelojFalso()
    brain = crear_brain(reloj)
    for _ in range(5):
        brain.procesar(frame_con_linea(160))
        reloj.avanzar(0.05)

    vacio = frame_con_linea(160)
    vacio[:] = 205
    for _ in range(16):  # 0.80s > tolerancia 0.5s
        comando, telemetria = brain.procesar(vacio)
        reloj.avanzar(0.05)
    assert telemetria.estado is EstadoRobot.RECUPERANDO
    assert comando in (Command.LEFT, Command.RIGHT)


def test_perdida_desviada_gira_de_inmediato() -> None:
    reloj = RelojFalso()
    brain = crear_brain(reloj)
    for _ in range(5):
        brain.procesar(frame_con_linea(240))  # línea a la derecha (error +80)
        reloj.avanzar(0.05)

    vacio = frame_con_linea(160)
    vacio[:] = 205
    for _ in range(4):  # apenas supera el debounce
        comando, telemetria = brain.procesar(vacio)
        reloj.avanzar(0.05)
    assert telemetria.estado is EstadoRobot.RECUPERANDO
    assert comando is Command.RIGHT  # venía desviado: gira, no cruza recto


def test_pistas_predefinidas_son_cerradas_y_visibles() -> None:
    camara = CamaraSintetica()
    for nombre, puntos in PISTAS.items():
        mundo = Mundo(pista=Pista(puntos), signos=[])
        pista = mundo.pista
        assert math.dist(pista.punto(0.0), pista.punto(0.999)) < 40.0
        # En varios puntos, con el robot bien puesto, la línea debe verse.
        detecciones = 0
        muestras = 0
        for k in range(0, 10):
            s = k / 10.0
            x, y = pista.punto(s)
            tx, ty = pista.tangente(s)
            frame = camara.render(mundo, Carrito(x, y, math.atan2(ty, tx)))
            muestras += 1
            if detectar_linea(frame).presente:
                detecciones += 1
        assert detecciones >= muestras - 1, f"pista {nombre} no detectable"


def test_pista_realista_sigue_detectandose() -> None:
    realismo = REALISMOS["realista"]
    assert realismo is not None
    mundo = Mundo(pista=Pista(PISTAS["cacahuate"]), signos=[], realismo=realismo)
    camara = CamaraSintetica()
    detecciones = 0
    muestras = 0
    for k in range(20):
        s = k / 20.0
        x, y = mundo.pista.punto(s)
        tx, ty = mundo.pista.tangente(s)
        frame = camara.render(mundo, Carrito(x, y, math.atan2(ty, tx)))
        muestras += 1
        if detectar_linea(frame).presente:
            detecciones += 1
    # Se acepta perder la muestra que caiga dentro de un hueco.
    assert detecciones >= muestras - 3


def test_reflejos_sobre_la_linea_no_rompen_deteccion() -> None:
    """Brillo blanco sobre la línea: se pierde alguna muestra pero no todo."""
    realismo = Realismo(ancho_min=20.0, ancho_max=26.0, reflejos=5, semilla=5)
    mundo = Mundo(pista=Pista(PISTAS["ovalo"]), signos=[], realismo=realismo)
    camara = CamaraSintetica()
    detecciones = 0
    errores = []
    muestras = 0
    for k in range(24):
        s = k / 24.0
        x, y = mundo.pista.punto(s)
        tx, ty = mundo.pista.tangente(s)
        frame = camara.render(mundo, Carrito(x, y, math.atan2(ty, tx)))
        muestras += 1
        linea = detectar_linea(frame)
        if linea.presente:
            detecciones += 1
            errores.append(linea.error_px)
    assert detecciones >= muestras - 4
    assert max(abs(e) for e in errores) < 60


def test_tramos_partidos_sesgo_dentro_de_banda_muerta() -> None:
    """Línea partida en paralelas: detecta y el sesgo no dispara giros."""
    realismo = Realismo(ancho_min=22.0, ancho_max=26.0, tramos_partidos=2, semilla=7)
    mundo = Mundo(pista=Pista(PISTAS["ovalo"]), signos=[], realismo=realismo)
    camara = CamaraSintetica()
    errores = []
    for k in range(24):
        s = k / 24.0
        x, y = mundo.pista.punto(s)
        tx, ty = mundo.pista.tangente(s)
        frame = camara.render(mundo, Carrito(x, y, math.atan2(ty, tx)))
        linea = detectar_linea(frame)
        if linea.presente:
            errores.append(linea.error_px)
    assert len(errores) >= 20  # la pista se ve casi siempre
    # El sesgo por elegir una de las paralelas debe caber en la banda muerta.
    assert np.mean(np.abs(errores)) < 25.0


def test_degradado_de_marker_sigue_siendo_linea() -> None:
    """Tramos con marcador descargado (color claro): K-Means los agrupa."""
    realismo = Realismo(degradado_fraccion=0.15, semilla=3)
    mundo = Mundo(pista=Pista(PISTAS["ovalo"]), signos=[], realismo=realismo)
    camara = CamaraSintetica()
    detecciones = 0
    for k in range(20):
        s = k / 20.0
        x, y = mundo.pista.punto(s)
        tx, ty = mundo.pista.tangente(s)
        frame = camara.render(mundo, Carrito(x, y, math.atan2(ty, tx)))
        if detectar_linea(frame).presente:
            detecciones += 1
    assert detecciones >= 16


def test_carrera_realista_completa_vueltas() -> None:
    """Con todas las aberraciones activas sigue completando la vuelta."""
    from first.adapters.sim.carrera import simular_carrera

    mundo = Mundo(
        pista=Pista(PISTAS["cacahuate"]),
        realismo=REALISMOS["realista"],
    )
    brain = Brain(segundos_pare=1.0, reloj=lambda: mundo.t, dibujar_debug=False)
    metricas = simular_carrera(brain, mundo, duracion=75.0, dt=1.0 / 60.0)
    assert metricas.vueltas >= 1
    # Con gaps + reflejos + partidos + manchas + degradado activos, algún
    # descarrilamiento puntual es esperable; lo imperdonable es no volver.
    assert metricas.descarrilamientos <= 5


def test_ancho_variable_extremo_detectable_en_recta() -> None:
    realismo = REALISMOS["extremo"]
    assert realismo is not None
    mundo = Mundo(pista=Pista(PISTAS["ovalo"]), signos=[], realismo=realismo)
    camara = CamaraSintetica()
    detecciones = sum(
        1
        for k in range(24)
        for s in [k / 24.0]
        for frame in [
            camara.render(
                mundo,
                Carrito(
                    *mundo.pista.punto(s),
                    math.atan2(*mundo.pista.tangente(s)[::-1]),
                ),
            )
        ]
        if detectar_linea(frame).presente
    )
    assert detecciones >= 24 - 5  # mayoría de la pista visible a pesar de todo
