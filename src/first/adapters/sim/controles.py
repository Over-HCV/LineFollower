"""Barra de controles del simulador: qué widget ajusta qué parámetro.

Agrupa los selectores y sliders para que el bucle de eventos solo hable de
clics y teclas, sin saber cuántos widgets hay ni en qué fila viven.
"""

from __future__ import annotations

from typing import Callable

import pygame

from ...core.brain import Brain
from .panel import Deslizador, SelectorGrupo
from .pista import PISTAS
from .realismo import REALISMOS

ANCHO_COLUMNA: int = 220  # riel + etiqueta de un slider
SLIDERS_POR_FILA: int = 5


class Controles:
    """Selectores (pista, realismo, modo) y sliders del panel en vivo."""

    def __init__(
        self,
        brain: Brain,
        pista_actual: str,
        realismo_actual: str,
        al_cambiar_pista: Callable[[str], None],
        al_cambiar_realismo: Callable[[str], None],
        al_cambiar_velocidad: Callable[[float], None],
    ) -> None:
        self.pista = SelectorGrupo(
            "pista", list(PISTAS), list(PISTAS).index(pista_actual), al_cambiar_pista
        )
        self.realismo = SelectorGrupo(
            "realismo",
            list(REALISMOS),
            list(REALISMOS).index(realismo_actual),
            al_cambiar_realismo,
        )
        self.modo = SelectorGrupo(
            "modo",
            ["kmeans", "umbral"],
            0 if brain.params_linea.modo == "kmeans" else 1,
            lambda modo: brain.reconfigurar(modo_linea=modo),
        )
        self.tolerancia = Deslizador(
            "tolerancia (px)",
            6.0,
            30.0,
            float(brain.tolerancia_entrada),
            paso=1.0,
            formato="{:.0f}",
            al_cambiar=lambda v: brain.reconfigurar(tolerancia_entrada=int(v)),
        )
        self.pare = Deslizador(
            "PARE (s)",
            0.5,
            10.0,
            brain.segundos_pare,
            paso=0.5,
            formato="{:.1f}",
            al_cambiar=lambda v: brain.reconfigurar(segundos_pare=v),
        )
        # Cortina: tapa columnas laterales del frame para que un tramo vecino
        # de la pista (chicane) no compita con la línea que se está siguiendo.
        self.cortina = Deslizador(
            "cortina (px)",
            0.0,
            80.0,
            float(brain.params_linea.margen_lateral),
            paso=5.0,
            formato="{:.0f}",
            al_cambiar=lambda v: brain.reconfigurar(margen_lateral=int(v)),
        )
        # Atención: σ de la gaussiana que pondera los candidatos a línea.
        # Bajarlo hace que el detector ignore los lados (pistas con tramos
        # paralelos); subirlo lo vuelve indiferente a dónde estaba la línea.
        self.atencion = Deslizador(
            "atencion o",
            0.05,
            0.20,
            float(brain.params_linea.sigma_atencion),
            paso=0.01,
            formato="{:.2f}",
            al_cambiar=lambda v: brain.reconfigurar(sigma_atencion=v),
        )
        self.velocidad = Deslizador(
            "velocidad x",
            0.25,
            3.0,
            1.0,
            paso=0.25,
            formato="{:.2f}",
            al_cambiar=al_cambiar_velocidad,
        )
        self.selectores = [self.pista, self.realismo, self.modo]
        self.deslizadores = [
            self.tolerancia,
            self.pare,
            self.cortina,
            self.atencion,
            self.velocidad,
        ]

    @property
    def filas_sliders(self) -> int:
        return -(-len(self.deslizadores) // SLIDERS_POR_FILA)

    def ancho_minimo(self, fuente: pygame.font.Font, margen: int) -> int:
        """Ancho que necesita la barra para que nada quede fuera de la ventana."""
        selectores = sum(s.ancho(fuente) + 3 * margen for s in self.selectores)
        columnas = min(len(self.deslizadores), SLIDERS_POR_FILA)
        sliders = columnas * (ANCHO_COLUMNA + 3 * margen)
        return max(selectores, sliders)

    def colocar(
        self, filas: tuple[int, ...], margen: int, fuente: pygame.font.Font
    ) -> None:
        """Fila 0: selectores en línea. Fila 1: los sliders, todos en una fila."""
        x = margen
        for selector in self.selectores:
            x += selector.colocar(x, filas[0], fuente) + 3 * margen
        for i, deslizador in enumerate(self.deslizadores):
            columna = i % SLIDERS_POR_FILA
            deslizador.colocar(
                margen + columna * (ANCHO_COLUMNA + 3 * margen),
                filas[1 + i // SLIDERS_POR_FILA],
            )

    def clic(self, pos: tuple[int, int]) -> bool:
        """Reparte el clic entre selectores y sliders; True si alguien lo tomó."""
        if any(selector.clic(pos) for selector in self.selectores):
            return True
        return any(deslizador.presionar(pos) for deslizador in self.deslizadores)

    def soltar(self) -> None:
        for deslizador in self.deslizadores:
            deslizador.soltar()

    def mover(self, pos: tuple[int, int]) -> None:
        for deslizador in self.deslizadores:
            deslizador.mover(pos)

    def dibujar(self, pantalla: pygame.Surface, fuente: pygame.font.Font) -> None:
        for widget in (*self.selectores, *self.deslizadores):
            widget.dibujar(pantalla, fuente)
