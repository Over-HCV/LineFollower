"""Tests de la máquina de estados del cerebro (PARE, SIGA, recuperación)."""

from __future__ import annotations

from first.core.brain import Brain
from first.core.types import Command, EstadoRobot

from conftest import RelojFalso, frame_con_linea, frame_linea_y_octagono, ROJO, VERDE


def crear_brain(reloj: RelojFalso, segundos_pare: float = 2.0) -> Brain:
    return Brain(
        segundos_pare=segundos_pare,
        reloj=reloj,
        dibujar_debug=False,
    )


def test_sigue_linea_central_avanzando() -> None:
    reloj = RelojFalso()
    brain = crear_brain(reloj)
    frame = frame_con_linea(160)
    comando, telemetria = None, None
    for _ in range(6):
        comando, telemetria = brain.procesar(frame)
        reloj.avanzar(0.1)
    assert comando is Command.UP
    assert telemetria.estado is EstadoRobot.SIGUIENDO


def test_pare_detiene_y_temporizador_reanuda() -> None:
    reloj = RelojFalso()
    brain = crear_brain(reloj, segundos_pare=2.0)
    frame_pare = frame_linea_y_octagono(ROJO)

    for _ in range(3):  # confirmación: 3 frames consecutivos
        comando, telemetria = brain.procesar(frame_pare)
        reloj.avanzar(0.1)
    assert telemetria.estado is EstadoRobot.PARE
    assert comando is Command.DOWN

    reloj.avanzar(1.0)  # 1.3s < 2.0s: sigue detenido
    comando, _ = brain.procesar(frame_pare)
    assert comando is Command.DOWN

    reloj.avanzar(1.0)  # 2.3s > 2.0s: reanuda aunque la señal siga visible
    comando, telemetria = brain.procesar(frame_pare)
    assert telemetria.estado is EstadoRobot.SIGUIENDO
    assert comando is Command.UP


def test_siga_reanuda_antes_del_temporizador() -> None:
    reloj = RelojFalso()
    brain = crear_brain(reloj, segundos_pare=30.0)

    for _ in range(3):
        brain.procesar(frame_linea_y_octagono(ROJO))
        reloj.avanzar(0.1)
    assert brain.estado is EstadoRobot.PARE

    for _ in range(2):  # SIGA confirmado con 2 frames
        comando, telemetria = brain.procesar(frame_linea_y_octagono(VERDE))
        reloj.avanzar(0.1)
    assert telemetria.estado is EstadoRobot.SIGUIENDO
    assert comando is Command.UP


def test_linea_perdida_recupera_y_luego_se_detiene() -> None:
    reloj = RelojFalso()
    brain = crear_brain(reloj, segundos_pare=2.0)
    brain.procesar(frame_con_linea(240))  # gira a la derecha (último lado con línea)
    reloj.avanzar(0.1)

    vacio = frame_con_linea(160)
    import numpy as np

    vacio[:] = 205  # sin línea
    # Debounce: los primeros 2 frames sin línea mantienen el último comando.
    comando, telemetria = brain.procesar(vacio)
    assert telemetria.estado is EstadoRobot.SIGUIENDO
    brain.procesar(vacio)
    comando, telemetria = brain.procesar(vacio)
    assert telemetria.estado is EstadoRobot.RECUPERANDO
    assert comando is Command.RIGHT  # barre hacia el último lado con línea

    for _ in range(35):  # 3.5s > timeout de recuperación (3.0s)
        comando, telemetria = brain.procesar(vacio)
        reloj.avanzar(0.1)
    assert telemetria.estado is EstadoRobot.PERDIDO
    assert comando is Command.DOWN


def test_frame_invalido_detiene() -> None:
    reloj = RelojFalso()
    brain = crear_brain(reloj)
    comando, _ = brain.procesar(None)
    assert comando is Command.DOWN


def test_reconfigurar_tolerancia_en_vivo() -> None:
    reloj = RelojFalso()
    brain = Brain(tolerancia_entrada=30, reloj=reloj, dibujar_debug=False)
    # error +20 < tolerancia 30: no gira
    for _ in range(6):
        comando, _ = brain.procesar(frame_con_linea(180))
    assert comando is Command.UP

    brain.reconfigurar(tolerancia_entrada=12)
    assert brain.tolerancia_entrada == 12
    # el mismo error ahora supera la tolerancia: gira a la derecha
    for _ in range(6):
        comando, _ = brain.procesar(frame_con_linea(180))
    assert comando is Command.RIGHT


def test_reconfigurar_conserva_estado_y_cambia_modo() -> None:
    reloj = RelojFalso()
    brain = Brain(reloj=reloj, dibujar_debug=False)
    for _ in range(3):
        brain.procesar(frame_linea_y_octagono(ROJO))
        reloj.avanzar(0.1)
    assert brain.estado is EstadoRobot.PARE

    brain.reconfigurar(modo_linea="umbral", segundos_pare=0.2)
    assert brain.params_linea.modo == "umbral"
    assert brain.segundos_pare == 0.2
    assert brain.estado is EstadoRobot.PARE  # la FSM no se reinicia

    reloj.avanzar(0.3)  # excede el nuevo tiempo de PARE
    _, telemetria = brain.procesar(frame_con_linea(160))
    assert telemetria.estado is EstadoRobot.SIGUIENDO


def test_reconfigurar_cortina_lateral() -> None:
    reloj = RelojFalso()
    brain = Brain(reloj=reloj, dibujar_debug=False)
    # Línea pegada al borde: visible hasta que baja la cortina.
    _, telemetria = brain.procesar(frame_con_linea(40))
    assert telemetria.linea.presente

    brain.reconfigurar(margen_lateral=60)
    assert brain.params_linea.margen_lateral == 60
    assert brain.estado is EstadoRobot.SIGUIENDO  # la FSM no se reinicia
    _, telemetria = brain.procesar(frame_con_linea(40))
    assert not telemetria.linea.presente


def test_no_se_va_tras_una_linea_vecina() -> None:
    from conftest import frame_con_dos_lineas

    reloj = RelojFalso()
    brain = Brain(reloj=reloj, dibujar_debug=False)
    for _ in range(5):  # engancha la línea central
        comando, _ = brain.procesar(frame_con_linea(160))
    assert comando is Command.UP

    # aparece una vecina más gruesa por la izquierda (chicane/ocho)
    for _ in range(8):
        comando, telemetria = brain.procesar(frame_con_dos_lineas(160, 40))
    assert comando is Command.UP
    assert telemetria.estado is EstadoRobot.SIGUIENDO
    assert abs(telemetria.linea.error_px) <= 8


def test_la_prediccion_del_centro_decae_sin_linea() -> None:
    from conftest import FONDO
    import numpy as np

    reloj = RelojFalso()
    brain = Brain(reloj=reloj, dibujar_debug=False)
    for _ in range(4):
        brain.procesar(frame_con_linea(240))  # línea muy a la derecha

    vacio = np.full((240, 320, 3), FONDO, dtype=np.uint8)
    for _ in range(20):
        brain.procesar(vacio)
    # tras perderla, la atención vuelve al centro en vez de quedarse fija
    _, telemetria = brain.procesar(frame_con_linea(160))
    assert telemetria.linea.presente
    assert abs(telemetria.linea.error_px) <= 8
