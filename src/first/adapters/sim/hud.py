"""Layout y utilidades de dibujo de la ventana del simulador.

El tamaño de cada zona se calcula desde la métrica de la fuente (y no con
números mágicos) para que ningún texto se corte en otra máquina o con otra
fuente disponible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import pygame
from numpy.typing import NDArray

from .lienzo import LIENZO_ANCHO

MARGEN: int = 12
INTER_LINEA: int = 26
PANEL_MUNDO: tuple[int, int] = (760, 570)

COLOR_FONDO: tuple[int, int, int] = (24, 24, 28)
COLOR_TEXTO: tuple[int, int, int] = (230, 230, 230)
COLOR_ACENTO: tuple[int, int, int] = (120, 220, 120)
COLOR_ALERTA: tuple[int, int, int] = (240, 120, 120)
COLOR_HUECO: tuple[int, int, int] = (240, 190, 90)
COLOR_BARRA: tuple[int, int, int] = (30, 30, 36)
COLOR_AYUDA: tuple[int, int, int] = (150, 150, 150)
COLOR_FANTASMA: tuple[int, int, int] = (250, 250, 120)

LINEAS_REFERENCIA: list[str] = [
    "modo: MANUAL   comando: right   estado: recuperando",
    "senal: siga   error: -999 px",
    "t: 99999.9s   vueltas: 99   descarrilamientos: 99",
    "PAREs cumplidos: 999   intervenciones: 999",
]


@dataclass(frozen=True, slots=True)
class Disposicion:
    """Geometría de la ventana: dónde va cada panel, texto y fila de widgets."""

    ventana: tuple[int, int]
    pos_mundo: tuple[int, int]
    rect_mundo: pygame.Rect
    pos_camara: tuple[int, int]
    pos_mascara: tuple[int, int]
    pos_texto: tuple[int, int]
    pos_ayuda: tuple[int, int]
    rect_barra: pygame.Rect
    filas: tuple[int, ...]  # y de cada fila de widgets de la barra
    escala: float  # mundo -> panel

    def a_mundo(self, pos: tuple[int, int]) -> tuple[float, float]:
        """Convierte un píxel de la ventana a coordenadas del mundo."""
        return (
            (pos[0] - self.pos_mundo[0]) / self.escala,
            (pos[1] - self.pos_mundo[1]) / self.escala,
        )

    def a_panel(self, punto: tuple[float, float]) -> tuple[int, int]:
        """Convierte un punto del mundo a píxel dentro del panel del mundo."""
        return (int(punto[0] * self.escala), int(punto[1] * self.escala))


def calcular(
    fuente: pygame.font.Font,
    fuente_chica: pygame.font.Font,
    tamaño_camara: tuple[int, int],
    ayuda: Sequence[str],
    filas_sliders: int = 1,
    ancho_minimo: int = 0,
) -> Disposicion:
    """Arma la disposición para la cámara, el texto y las filas de la barra.

    ``ancho_minimo`` es lo que la barra de controles necesita para no salirse
    por la derecha: la ventana nunca queda más angosta que sus widgets.
    """
    cam_w, cam_h = tamaño_camara
    ancho_texto = max(fuente.size(texto)[0] for texto in LINEAS_REFERENCIA)
    ancho_derecha = max(cam_w, ancho_texto) + 2 * MARGEN
    ancho_ventana = MARGEN + PANEL_MUNDO[0] + MARGEN + ancho_derecha + MARGEN
    ancho_ayuda = max(fuente_chica.size(linea)[0] for linea in ayuda)
    ancho_ventana = max(
        ancho_ventana, ancho_ayuda + 2 * MARGEN, ancho_minimo + 2 * MARGEN
    )

    alto_derecha = MARGEN + cam_h + MARGEN + cam_h + MARGEN + 4 * INTER_LINEA
    alto_ayuda = (
        PANEL_MUNDO[1]
        + INTER_LINEA
        + len(ayuda) * fuente_chica.get_height()
        + MARGEN
    )
    alto_contenido = max(alto_derecha, alto_ayuda) + MARGEN

    fila_selector = fuente.get_height() + 10
    fila_slider = fuente.get_height() + 22
    barra_alto = (
        MARGEN
        + fila_selector
        + filas_sliders * (MARGEN // 2 + fila_slider)
        + MARGEN
    )

    x_derecha = MARGEN + PANEL_MUNDO[0] + MARGEN
    y_fila1 = alto_contenido + MARGEN
    filas = [y_fila1]
    for i in range(filas_sliders):
        anterior = fila_selector if i == 0 else fila_slider
        filas.append(filas[-1] + anterior + MARGEN // 2)

    return Disposicion(
        ventana=(ancho_ventana, alto_contenido + barra_alto),
        pos_mundo=(MARGEN, MARGEN),
        rect_mundo=pygame.Rect((MARGEN, MARGEN), PANEL_MUNDO),
        pos_camara=(x_derecha + (ancho_derecha - cam_w) // 2, MARGEN),
        pos_mascara=(x_derecha + (ancho_derecha - cam_w) // 2, MARGEN + cam_h + MARGEN),
        pos_texto=(x_derecha + MARGEN, MARGEN + 2 * (cam_h + MARGEN)),
        pos_ayuda=(MARGEN, MARGEN + PANEL_MUNDO[1] + INTER_LINEA // 2),
        rect_barra=pygame.Rect(0, alto_contenido, ancho_ventana, barra_alto),
        filas=tuple(filas),
        escala=PANEL_MUNDO[0] / LIENZO_ANCHO,
    )


def a_superficie(imagen_bgr: NDArray[np.uint8]) -> pygame.Surface:
    rgb = np.ascontiguousarray(imagen_bgr[:, :, ::-1])
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))


def mascara_a_frame(
    mascara: NDArray[np.uint8] | None, tamaño: tuple[int, int]
) -> NDArray[np.uint8]:
    """Expande la máscara de la ROI a un frame BGR del tamaño indicado."""
    if mascara is None:
        return np.zeros((tamaño[1], tamaño[0], 3), dtype=np.uint8)
    gris = cv2.resize(mascara, tamaño, interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def color_estado(estado: str) -> tuple[int, int, int]:
    """Color del carrito según el estado de la máquina del cerebro."""
    if estado == "pare":
        return COLOR_ALERTA
    if estado == "hueco":
        return COLOR_HUECO
    return (60, 120, 255)


def panel_mundo(
    superficie: pygame.Surface,
    disposicion: Disposicion,
    pose: tuple[float, float, float],
    estado: str,
    fantasma: tuple[int, int] | None = None,
) -> pygame.Surface:
    """Copia del lienzo con el carrito, su rumbo y el fantasma del arrastre."""
    panel = superficie.copy()
    color = color_estado(estado)
    x, y, theta = pose
    cx, cy = disposicion.a_panel((x, y))
    pygame.draw.circle(panel, color, (cx, cy), 11)
    pygame.draw.line(
        panel,
        color,
        (cx, cy),
        (cx + 20 * math.cos(theta), cy + 20 * math.sin(theta)),
        4,
    )
    if fantasma is not None:
        pygame.draw.circle(
            panel,
            COLOR_FANTASMA,
            (
                fantasma[0] - disposicion.pos_mundo[0],
                fantasma[1] - disposicion.pos_mundo[1],
            ),
            14,
            width=2,
        )
    return panel


def dibujar_ayuda(
    pantalla: pygame.Surface,
    fuente: pygame.font.Font,
    disposicion: Disposicion,
    lineas: Sequence[str],
) -> None:
    """Recordatorio de atajos bajo el panel del mundo, una línea por renglón."""
    alto = fuente.get_height()
    for i, texto in enumerate(lineas):
        pantalla.blit(
            fuente.render(texto, True, COLOR_AYUDA),
            (disposicion.pos_ayuda[0], disposicion.pos_ayuda[1] + i * alto),
        )


def dibujar_lineas(
    pantalla: pygame.Surface,
    fuente: pygame.font.Font,
    disposicion: Disposicion,
    lineas: list[tuple[str, tuple[int, int, int]]],
) -> None:
    """Bloque de telemetría a la derecha, una línea por renglón."""
    for i, (texto, color) in enumerate(lineas):
        pantalla.blit(
            fuente.render(texto, True, color),
            (disposicion.pos_texto[0], disposicion.pos_texto[1] + i * INTER_LINEA),
        )
