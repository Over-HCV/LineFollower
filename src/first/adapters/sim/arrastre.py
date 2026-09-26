"""Arrastre de la escena: tomar y soltar el carrito o una señal en el mundo.

Todo en coordenadas del mundo y sin pygame: la UI solo convierte el píxel de
pantalla a mundo y delega aquí, así la regla de "a dónde va lo que suelto"
(pegarse a la línea, elegir lado y separación) se puede probar sin ventana.
"""

from __future__ import annotations

import math

from .world import OFFSET_SIGNO_MAX, OFFSET_SIGNO_MIN, Mundo

RADIO_CARRITO: float = 45.0  # zona de agarre del carrito (px de mundo)

Punto = tuple[float, float]


class Arrastre:
    """Estado del arrastre en curso: nada, el carrito, o el índice de una señal."""

    def __init__(self) -> None:
        self.carrito: bool = False
        self.signo: int | None = None

    @property
    def activo(self) -> bool:
        return self.carrito or self.signo is not None

    def tomar(self, mundo: Mundo, punto: Punto) -> bool:
        """Agarra lo que haya bajo el punto (la señal tiene prioridad)."""
        for i, signo in enumerate(mundo.signos):
            if math.dist(punto, mundo.posicion_signo(signo)) <= signo.radio:
                self.signo = i
                return True
        if math.dist(punto, (mundo.carrito.x, mundo.carrito.y)) <= RADIO_CARRITO:
            self.carrito = True
            return True
        return False

    def cancelar(self) -> None:
        self.carrito = False
        self.signo = None

    def soltar(self, mundo: Mundo, punto: Punto) -> bool:
        """Confirma la nueva posición; retorna True si algo cambió."""
        if self.carrito:
            mundo.reposicionar_en_pista(mundo.pista.fraccion_cercana(punto))
        elif self.signo is not None:
            s, lado, offset = ubicacion_signo(mundo, punto)
            mundo.mover_signo(self.signo, s, lado, offset)
        else:
            return False
        self.cancelar()
        return True


def ubicacion_signo(mundo: Mundo, punto: Punto) -> tuple[float, int, float]:
    """Traduce un punto suelto a (fracción de pista, lado, separación).

    El lado sale del signo de la proyección sobre la normal de la pista y la
    separación de su magnitud, acotada al rango en que la señal sigue siendo
    visible junto a la línea sin taparla.
    """
    s = mundo.pista.fraccion_cercana(punto)
    px, py = mundo.pista.punto(s)
    nx, ny = mundo.pista.normal(s)
    proyeccion = (punto[0] - px) * nx + (punto[1] - py) * ny
    lado = 1 if proyeccion >= 0 else -1
    offset = min(OFFSET_SIGNO_MAX, max(OFFSET_SIGNO_MIN, abs(proyeccion)))
    return s, lado, offset
