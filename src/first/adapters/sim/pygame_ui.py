"""Frontend pygame: vista del mundo, cámara del carrito, telemetría y métricas.

Métricas en vivo estilo rúbrica: tiempo, vueltas, descarrilamientos,
PAREs cumplidos e intervenciones humanas (las flechas cuentan como
intervención, como en la competencia).

Incluye un panel de control en vivo: sliders y selectores que ajustan la
tolerancia, el modo de línea, los segundos de PARE, la pista y el realismo
mientras la simulación corre.

El layout se calcula desde la métrica de la fuente para que ningún texto
se corte sin importar el sistema o la fuente disponible.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pygame
from numpy.typing import NDArray

from ...core.brain import Brain
from ...core.types import Command, EstadoRobot
from .camera import CamaraSintetica
from .panel import Deslizador, SelectorGrupo
from .world import LIENZO_ANCHO, PISTAS, REALISMOS, Mundo, Pista

MARGEN: int = 12
INTER_LINEA: int = 26
PANEL_MUNDO: tuple[int, int] = (760, 570)

TECLAS_MANUAL: dict[int, Command] = {
    pygame.K_UP: Command.UP,
    pygame.K_RIGHT: Command.RIGHT,
    pygame.K_DOWN: Command.DOWN,
    pygame.K_LEFT: Command.LEFT,
}

TECLAS_PISTA: dict[int, int] = {
    pygame.K_1: 0,
    pygame.K_2: 1,
    pygame.K_3: 2,
    pygame.K_4: 3,
}

COLOR_TEXTO: tuple[int, int, int] = (230, 230, 230)
COLOR_ACENTO: tuple[int, int, int] = (120, 220, 120)
COLOR_ALERTA: tuple[int, int, int] = (240, 120, 120)
COLOR_HUECO: tuple[int, int, int] = (240, 190, 90)
COLOR_BARRA: tuple[int, int, int] = (30, 30, 36)


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
    pista: str = "cacahuate",
    realismo: str = "perfecto",
    fps: int = 60,
) -> dict[str, float | int]:
    """Corre el simulador con ventana pygame. Retorna métricas al cerrar."""
    pygame.init()

    fuente = pygame.font.SysFont("menlo,consolas,monaco,monospace", 16)
    fuente_chica = pygame.font.SysFont("menlo,consolas,monaco,monospace", 13)

    pista_actual = pista if pista in PISTAS else "cacahuate"
    realismo_actual = realismo if realismo in REALISMOS else "perfecto"
    mundo = mundo or Mundo(
        pista=Pista(PISTAS[pista_actual]), realismo=REALISMOS[realismo_actual]
    )
    camara = camara or CamaraSintetica()
    brain.reiniciar()

    cam_w, cam_h = camara.tamaño_frame

    # ---------- callbacks del panel ----------
    def recrear_mundo() -> Mundo:
        return Mundo(
            pista=Pista(PISTAS[pista_actual]),
            realismo=REALISMOS[realismo_actual],
        )

    def cambiar_pista(nombre: str) -> None:
        nonlocal pista_actual, mundo, mundo_superficie, intervenciones, pares_cumplidos, previo
        pista_actual = nombre
        mundo = recrear_mundo()
        brain.reiniciar()
        mundo_superficie = pygame.transform.smoothscale(
            __a_superficie(mundo.lienzo), PANEL_MUNDO
        )
        intervenciones, pares_cumplidos, previo = 0, 0, EstadoRobot.SIGUIENDO

    def cambiar_realismo(nombre: str) -> None:
        nonlocal realismo_actual
        realismo_actual = nombre
        cambiar_pista(pista_actual)

    selector_pista = SelectorGrupo(
        "pista", list(PISTAS), list(PISTAS).index(pista_actual), cambiar_pista
    )
    selector_realismo = SelectorGrupo(
        "realismo", list(REALISMOS), list(REALISMOS).index(realismo_actual), cambiar_realismo
    )
    selector_modo = SelectorGrupo(
        "modo",
        ["kmeans", "umbral"],
        0 if brain.params_linea.modo == "kmeans" else 1,
        lambda modo: brain.reconfigurar(modo_linea=modo),
    )
    deslizador_tolerancia = Deslizador(
        "tolerancia (px)",
        6.0,
        30.0,
        float(brain.tolerancia_entrada),
        paso=1.0,
        formato="{:.0f}",
        al_cambiar=lambda v: brain.reconfigurar(tolerancia_entrada=int(v)),
    )
    deslizador_pare = Deslizador(
        "PARE (s)",
        0.5,
        10.0,
        brain.segundos_pare,
        paso=0.5,
        formato="{:.1f}",
        al_cambiar=lambda v: brain.reconfigurar(segundos_pare=v),
    )

    # ---------- layout ----------
    lineas_referencia = [
        "modo: MANUAL   comando: right   estado: recuperando",
        "senal: siga   error: -999 px",
        "t: 99999.9s   vueltas: 99   descarrilamientos: 99",
        "PAREs cumplidos: 999   intervenciones: 999",
    ]
    ancho_texto = max(fuente.size(texto)[0] for texto in lineas_referencia)
    ayuda = "A: auto/manual   R: reiniciar   flechas: manual   M: modo   +/-: tolerancia   1-4: pista   Q: sal"
    ancho_ayuda = fuente_chica.size(ayuda)[0]

    ancho_derecha = max(cam_w, ancho_texto) + 2 * MARGEN
    ancho_ventana = MARGEN + PANEL_MUNDO[0] + MARGEN + ancho_derecha + MARGEN
    ancho_ventana = max(ancho_ventana, ancho_ayuda + 2 * MARGEN)
    alto_derecha = MARGEN + cam_h + MARGEN + cam_h + MARGEN + 4 * INTER_LINEA
    alto_ayuda = PANEL_MUNDO[1] + INTER_LINEA + fuente_chica.get_height() + MARGEN
    alto_contenido = max(alto_derecha, alto_ayuda) + MARGEN

    fila_selector = fuente.get_height() + 10
    fila_slider = fuente.get_height() + 12 + 10
    barra_alto = MARGEN + fila_selector + MARGEN // 2 + fila_slider + MARGEN
    alto_ventana = alto_contenido + barra_alto

    pos_mundo = (MARGEN, MARGEN)
    pos_ayuda = (MARGEN, MARGEN + PANEL_MUNDO[1] + INTER_LINEA // 2)
    x_derecha = MARGEN + PANEL_MUNDO[0] + MARGEN
    pos_camara = (x_derecha + (ancho_derecha - cam_w) // 2, MARGEN)
    pos_mascara = (pos_camara[0], MARGEN + cam_h + MARGEN)
    pos_texto = (x_derecha + MARGEN, pos_mascara[1] + cam_h + MARGEN)
    rect_barra = pygame.Rect(0, alto_contenido, ancho_ventana, barra_alto)

    y_fila1 = alto_contenido + MARGEN
    y_fila2 = y_fila1 + fila_selector + MARGEN // 2
    x_cur = MARGEN
    ancho = selector_pista.colocar(x_cur, y_fila1, fuente)
    x_cur += ancho + 3 * MARGEN
    x_cur += selector_realismo.colocar(x_cur, y_fila1, fuente) + 3 * MARGEN
    selector_modo.colocar(x_cur, y_fila1, fuente)

    x_cur = MARGEN
    deslizador_tolerancia.colocar(x_cur, y_fila2)
    x_cur += 220 + 3 * MARGEN
    deslizador_pare.colocar(x_cur, y_fila2)

    pantalla = pygame.display.set_mode((ancho_ventana, alto_ventana))
    pygame.display.set_caption("Reto de visión: seguidor de línea (simulador)")
    reloj = pygame.time.Clock()

    escala = PANEL_MUNDO[0] / LIENZO_ANCHO

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
                elif evento.key == pygame.K_m:
                    selector_modo.ciclo()
                elif evento.key in (
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS,
                ):
                    deslizador_tolerancia.ajustar(1.0)
                elif evento.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    deslizador_tolerancia.ajustar(-1.0)
                elif evento.key in TECLAS_PISTA:
                    selector_pista.seleccionar(TECLAS_PISTA[evento.key])
                elif evento.key == pygame.K_r:
                    cambiar_pista(pista_actual)  # recrea con la misma config
            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                pos = evento.pos
                if not (
                    selector_pista.clic(pos)
                    or selector_realismo.clic(pos)
                    or selector_modo.clic(pos)
                ):
                    deslizador_tolerancia.presionar(pos)
                    deslizador_pare.presionar(pos)
            elif evento.type == pygame.MOUSEBUTTONUP:
                deslizador_tolerancia.soltar()
                deslizador_pare.soltar()
            elif evento.type == pygame.MOUSEMOTION:
                deslizador_tolerancia.mover(evento.pos)
                deslizador_pare.mover(evento.pos)

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

        pygame.draw.rect(pantalla, COLOR_BARRA, rect_barra)
        selector_pista.dibujar(pantalla, fuente)
        selector_realismo.dibujar(pantalla, fuente)
        selector_modo.dibujar(pantalla, fuente)
        deslizador_tolerancia.dibujar(pantalla, fuente)
        deslizador_pare.dibujar(pantalla, fuente)

        pygame.display.flip()

    pygame.quit()
    return {
        "duracion": mundo.t,
        "vueltas": mundo.vueltas,
        "descarrilamientos": mundo.descarrilamientos,
        "pares_cumplidos": pares_cumplidos,
        "intervenciones": intervenciones,
    }
