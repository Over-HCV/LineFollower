"""Panel de control en vivo: sliders y selectores sobre la ventana pygame.

La lógica de valor de los widgets (acotado, snap a paso, fracción, selección
excluyente) vive en métodos puros, testeables sin abrir ventana; el manejo
de mouse y el dibujado son métodos aparte.
"""

from __future__ import annotations

from typing import Callable

import pygame

COLOR_RIEL: tuple[int, int, int] = (60, 60, 72)
COLOR_PERILLA: tuple[int, int, int] = (235, 235, 235)
COLOR_BOTON: tuple[int, int, int] = (55, 55, 65)
COLOR_BOTON_ACTIVO: tuple[int, int, int] = (110, 200, 110)
COLOR_TEXTO: tuple[int, int, int] = (225, 225, 225)
COLOR_ETIQUETA: tuple[int, int, int] = (150, 150, 160)


class Deslizador:
    """Slider arrastrable para parámetros continuos (con snap a paso)."""

    def __init__(
        self,
        etiqueta: str,
        minimo: float,
        maximo: float,
        valor: float,
        paso: float = 1.0,
        formato: str = "{:.0f}",
        al_cambiar: Callable[[float], None] | None = None,
    ) -> None:
        self.etiqueta = etiqueta
        self.minimo = float(minimo)
        self.maximo = float(maximo)
        self.paso = float(paso)
        self.formato = formato
        self.al_cambiar = al_cambiar
        self.__valor = self.__aplanar(float(valor))
        self.__arrastrando = False
        self.rect = pygame.Rect(0, 0, 150, 12)
        self.__pos_etiqueta: tuple[int, int] = (0, 0)

    # ---------- lógica pura ----------

    def __acotar(self, valor: float) -> float:
        return min(self.maximo, max(self.minimo, valor))

    def __aplanar(self, valor: float) -> float:
        pasos = round((self.__acotar(valor) - self.minimo) / self.paso)
        return self.__acotar(self.minimo + pasos * self.paso)

    @property
    def valor(self) -> float:
        return self.__valor

    @property
    def fraccion(self) -> float:
        return (self.__valor - self.minimo) / (self.maximo - self.minimo)

    def fijar(self, valor: float) -> float:
        """Fija el valor (acotado y aplanado); dispara callback si cambia."""
        nuevo = self.__aplanar(valor)
        if nuevo != self.__valor:
            self.__valor = nuevo
            if self.al_cambiar is not None:
                self.al_cambiar(nuevo)
        return nuevo

    def fijar_fraccion(self, fraccion: float) -> float:
        fraccion = min(1.0, max(0.0, fraccion))
        return self.fijar(self.minimo + fraccion * (self.maximo - self.minimo))

    def ajustar(self, delta: float) -> float:
        return self.fijar(self.__valor + delta)

    # ---------- interacción ----------

    def presionar(self, pos: tuple[int, int]) -> bool:
        """Click sobre el riel: inicia arrastre y salta al punto."""
        if self.rect.collidepoint(pos):
            self.__arrastrando = True
            self.__mover(pos)
            return True
        return False

    def soltar(self) -> None:
        self.__arrastrando = False

    def mover(self, pos: tuple[int, int]) -> None:
        if self.__arrastrando:
            self.__mover(pos)

    def __mover(self, pos: tuple[int, int]) -> None:
        if self.rect.w <= 0:
            return
        fraccion = (pos[0] - self.rect.x) / self.rect.w
        self.fijar_fraccion(fraccion)

    # ---------- layout y rendering ----------

    def alto(self, fuente: pygame.font.Font) -> int:
        return fuente.get_height() + self.rect.h + 10

    def colocar(self, x: int, y: int, ancho_riel: int = 150) -> int:
        """Posiciona etiqueta y riel; retorna el ancho total ocupado."""
        self.rect.topleft = (x, y + 18)
        self.__pos_etiqueta = (x, y)
        return ancho_riel

    def dibujar(self, pantalla: pygame.Surface, fuente: pygame.font.Font) -> None:
        texto = f"{self.etiqueta}: {self.formato.format(self.__valor)}"
        pantalla.blit(
            fuente.render(texto, True, COLOR_ETIQUETA), self.__pos_etiqueta
        )
        pygame.draw.rect(pantalla, COLOR_RIEL, self.rect, border_radius=6)
        perilla_x = self.rect.x + int(self.fraccion * self.rect.w)
        pygame.draw.circle(
            pantalla,
            COLOR_PERILLA,
            (perilla_x, self.rect.centery),
            9,
        )


class SelectorGrupo:
    """Botones excluyentes para opciones discretas (la activa resaltada)."""

    def __init__(
        self,
        etiqueta: str,
        opciones: list[str],
        indice: int = 0,
        al_cambiar: Callable[[str], None] | None = None,
    ) -> None:
        self.etiqueta = etiqueta
        self.opciones = opciones
        self.__indice = indice if 0 <= indice < len(opciones) else 0
        self.al_cambiar = al_cambiar
        self.rects: list[pygame.Rect] = [
            pygame.Rect(0, 0, 0, 0) for _ in opciones
        ]
        self.__pos_etiqueta: tuple[int, int] = (0, 0)

    # ---------- lógica pura ----------

    @property
    def indice(self) -> int:
        return self.__indice

    @property
    def activa(self) -> str:
        return self.opciones[self.__indice]

    def seleccionar(self, indice: int) -> str:
        """Selecciona por índice (con wrap); dispara callback si cambia."""
        indice = indice % len(self.opciones)
        if indice != self.__indice:
            self.__indice = indice
            if self.al_cambiar is not None:
                self.al_cambiar(self.opciones[indice])
        return self.opciones[self.__indice]

    def ciclo(self) -> str:
        return self.seleccionar(self.__indice + 1)

    def clic(self, pos: tuple[int, int]) -> bool:
        """Procesa un click; True si algún botón del grupo lo consumió."""
        for i, rect in enumerate(self.rects):
            if rect.collidepoint(pos):
                self.seleccionar(i)
                return True
        return False

    # ---------- layout y rendering ----------

    def ancho_boton(self, fuente: pygame.font.Font) -> int:
        return max(fuente.size(opcion)[0] for opcion in self.opciones) + 18

    def alto_boton(self, fuente: pygame.font.Font) -> int:
        return fuente.get_height() + 8

    def ancho(self, fuente: pygame.font.Font) -> int:
        """Ancho total del grupo: etiqueta + botones con su separación."""
        return fuente.size(self.etiqueta)[0] + 10 + sum(
            fuente.size(opcion)[0] + 18 + 6 for opcion in self.opciones
        )

    def colocar(self, x: int, y: int, fuente: pygame.font.Font) -> int:
        """Posiciona etiqueta y botones; retorna el ancho total ocupado."""
        self.__pos_etiqueta = (x, y + 6)
        alto = self.alto_boton(fuente)
        ancho = fuente.size(self.etiqueta)[0] + 10
        bx = x + ancho
        for i, opcion in enumerate(self.opciones):
            w = fuente.size(opcion)[0] + 18
            self.rects[i] = pygame.Rect(bx, y, w, alto)
            bx += w + 6
        return bx - x

    def dibujar(self, pantalla: pygame.Surface, fuente: pygame.font.Font) -> None:
        pantalla.blit(
            fuente.render(self.etiqueta, True, COLOR_ETIQUETA),
            self.__pos_etiqueta,
        )
        for i, (rect, opcion) in enumerate(zip(self.rects, self.opciones)):
            activo = i == self.__indice
            pygame.draw.rect(
                pantalla,
                COLOR_BOTON_ACTIVO if activo else COLOR_BOTON,
                rect,
                border_radius=6,
            )
            texto = fuente.render(opcion, True, (20, 30, 20) if activo else COLOR_TEXTO)
            pantalla.blit(
                texto,
                (
                    rect.x + (rect.w - texto.get_width()) // 2,
                    rect.y + (rect.h - texto.get_height()) // 2,
                ),
            )
