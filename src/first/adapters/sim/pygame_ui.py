"""Frontend pygame: vista del mundo, cámara del carrito, telemetría y métricas.

Métricas en vivo estilo rúbrica: tiempo, vueltas, descarrilamientos,
PAREs cumplidos e intervenciones humanas (las flechas cuentan como
intervención, como en la competencia).

El layout se calcula desde la métrica de la fuente para que ningún texto
se corte sin importar el sistema o la fuente disponible.
"""

from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np
import pygame
from numpy.typing import NDArray

from ...core.brain import Brain
from ...core.types import Command, EstadoRobot
from .camera import CamaraSintetica
from .world import LIENZO_ANCHO, Mundo

MARGEN: int = 12
INTER_LINEA: int = 26
PANEL_MUNDO: tuple[int, int] = (760, 570)

TECLAS_MANUAL: dict[int, Command] = {
    pygame.K_UP: Command.UP,
    pygame.K_RIGHT: Command.RIGHT,
    pygame.K_DOWN: Command.DOWN,
    pygame.K_LEFT: Command.LEFT,
}

COLOR_TEXTO: tuple[int, int, int] = (230, 230, 230)
COLOR_ACENTO: tuple[int, int, int] = (120, 220, 120)
COLOR_ALERTA: tuple[int, int, int] = (240, 120, 120)
COLOR_HUECO: tuple[int, int, int] = (240, 190, 90)


def __a_superficie(imagen_bgr: NDArray[np.uint8]) -> pygame.Surface:
    rgb = np.ascontiguousarray(imagen_bgr[:, :, ::-1])
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))


def __mascara_a_frame(
    mascara: NDArray[np.uint8] | None, tamaño: tuple[int, int]
) -> NDArray[np.uint8]:
    """Expande la máscara de la ROI a un frame BGR del tamaño indicado."""
    if mascara is None:
        return np.zeros((tamaño[1], tamaño[0], 3), dtype=np.uint8)
    gris = cv2.resize(mascara, tamaño, interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def ejecutar_simulador(
    brain: Brain,
    mundo: Mundo | None = None,
    camara: CamaraSintetica | None = None,
    recrear: Callable[[], Mundo] | None = None,
    fps: int = 60,
) -> dict[str, float | int]:
    """Corre el simulador con la ventana pygame. Retorna métricas al cerrar."""
    pygame.init()

    fuente = pygame.font.SysFont("menlo,consolas,monaco,monospace", 16)
    fuente_chica = pygame.font.SysFont("menlo,consolas,monaco,monospace", 13)

    mundo = mundo or Mundo()
    camara = camara or CamaraSintetica()
    brain.reiniciar()

    cam_w, cam_h = camara.tamaño_frame

    lineas_referencia = [
        "modo: MANUAL   comando: right   estado: recuperando",
        "senal: siga   error: -999 px",
        "t: 99999.9s   vueltas: 99   descarrilamientos: 99",
        "PAREs cumplidos: 999   intervenciones: 999",
    ]
    ancho_texto = max(fuente.size(texto)[0] for texto in lineas_referencia)
    ayuda = "A: auto/manual   R: reiniciar   flechas: manual (cuenta intervencion)   Q: salir"
    ancho_ayuda = fuente_chica.size(ayuda)[0]

    ancho_derecha = max(cam_w, ancho_texto) + 2 * MARGEN
    ancho_ventana = MARGEN + PANEL_MUNDO[0] + MARGEN + ancho_derecha + MARGEN
    ancho_ventana = max(ancho_ventana, ancho_ayuda + 2 * MARGEN)
    alto_derecha = MARGEN + cam_h + MARGEN + cam_h + MARGEN + 4 * INTER_LINEA
    alto_ayuda = PANEL_MUNDO[1] + INTER_LINEA + fuente_chica.get_height() + MARGEN
    alto_ventana = max(alto_derecha, alto_ayuda) + MARGEN

    pos_mundo = (MARGEN, MARGEN)
    pos_ayuda = (MARGEN, MARGEN + PANEL_MUNDO[1] + INTER_LINEA // 2)
    x_derecha = MARGEN + PANEL_MUNDO[0] + MARGEN
    pos_camara = (x_derecha + (ancho_derecha - cam_w) // 2, MARGEN)
    pos_mascara = (pos_camara[0], MARGEN + cam_h + MARGEN)
    pos_texto = (x_derecha + MARGEN, pos_mascara[1] + cam_h + MARGEN)

    pantalla = pygame.display.set_mode((ancho_ventana, alto_ventana))
    pygame.display.set_caption("Reto de visión: seguidor de línea (simulador)")
    reloj = pygame.time.Clock()

    escala = PANEL_MUNDO[0] / LIENZO_ANCHO

    def mundo_nuevo() -> Mundo:
        return recrear() if recrear is not None else Mundo()

    mundo_superficie = pygame.transform.smoothscale(
        __a_superficie(mundo.lienzo), PANEL_MUNDO
    )

    automatico: bool = True
    intervenciones: int = 0
    pares_cumplidos: int = 0
    previo: EstadoRobot = EstadoRobot.SIGUIENDO
    telemetria_estado: str = EstadoRobot.SIGUIENDO.value
    comando_texto: str = Command.UP.value
    senal_texto: str = "-"
    error_texto: str = "0"

    corriendo: bool = True
    while corriendo:
        dt = reloj.tick(fps) / 1000.0
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key in (pygame.K_q, pygame.K_ESCAPE):
                    corriendo = False
                elif evento.key == pygame.K_a:
                    automatico = not automatico
                elif evento.key == pygame.K_r:
                    mundo = mundo_nuevo()
                    brain.reiniciar()
                    mundo_superficie = pygame.transform.smoothscale(
                        __a_superficie(mundo.lienzo), PANEL_MUNDO
                    )
                    intervenciones = 0
                    pares_cumplidos = 0

        teclas = pygame.key.get_pressed()
        manual = next((c for k, c in TECLAS_MANUAL.items() if teclas[k]), None)
        if manual is not None and automatico:
            automatico = False
            intervenciones += 1

        frame = camara.render(mundo, mundo.carrito)
        comando_auto, telemetria = brain.procesar(frame)
        comando = comando_auto if automatico else (manual or Command.DOWN)

        if previo is EstadoRobot.PARE and telemetria.estado is not EstadoRobot.PARE:
            pares_cumplidos += 1
        previo = telemetria.estado
        telemetria_estado = telemetria.estado.value
        comando_texto = comando.value
        senal_texto = (
            telemetria.senal_cruda.value if telemetria.senal_cruda is not None else "-"
        )
        error_texto = str(telemetria.linea.error_px)

        mundo.paso(comando, dt)

        pantalla.fill((24, 24, 28))

        panel = mundo_superficie.copy()
        cx, cy = mundo.carrito.x * escala, mundo.carrito.y * escala
        color_estado = (
            COLOR_ALERTA
            if telemetria_estado == "pare"
            else (COLOR_HUECO if telemetria_estado == "hueco" else (60, 120, 255))
        )
        pygame.draw.circle(panel, color_estado, (int(cx), int(cy)), 11)
        hx = cx + 20 * math.cos(mundo.carrito.theta)
        hy = cy + 20 * math.sin(mundo.carrito.theta)
        pygame.draw.line(panel, color_estado, (cx, cy), (hx, hy), 4)
        pantalla.blit(panel, pos_mundo)

        camara_vista = telemetria.debug if telemetria.debug is not None else frame
        pantalla.blit(__a_superficie(camara_vista), pos_camara)
        mascara_frame = __mascara_a_frame(telemetria.linea.mascara, (cam_w, cam_h))
        pantalla.blit(__a_superficie(mascara_frame), pos_mascara)

        modo = "AUTO" if automatico else "MANUAL"
        lineas = [
            (
                f"modo: {modo}   comando: {comando_texto}   estado: {telemetria_estado}",
                COLOR_ACENTO,
            ),
            (f"senal: {senal_texto}   error: {error_texto} px", COLOR_TEXTO),
            (
                f"t: {mundo.t:7.1f}s   vueltas: {mundo.vueltas}   "
                f"descarrilamientos: {mundo.descarrilamientos}",
                COLOR_TEXTO,
            ),
            (
                f"PAREs cumplidos: {pares_cumplidos}   intervenciones: {intervenciones}",
                COLOR_TEXTO,
            ),
        ]
        for i, (texto, color) in enumerate(lineas):
            pantalla.blit(
                fuente.render(texto, True, color),
                (pos_texto[0], pos_texto[1] + i * INTER_LINEA),
            )

        pantalla.blit(
            fuente_chica.render(ayuda, True, (150, 150, 150)), pos_ayuda
        )

        pygame.display.flip()

    pygame.quit()
    return {
        "duracion": mundo.t,
        "vueltas": mundo.vueltas,
        "descarrilamientos": mundo.descarrilamientos,
        "pares_cumplidos": pares_cumplidos,
        "intervenciones": intervenciones,
    }
