"""Control discreto: banda de histéresis + votación por mayoría.

Problema: el actuador solo acepta un comando a la vez (up/right/down/left),
por lo que un umbral simple produce oscilación izq/der. Solución:
- Histéresis (tipo Schmitt): se entra al giro con |error| >= umbral_entrada
  y solo se sale cuando |error| < umbral_salida.
- Votación: el comando publicado es la mayoría de una deque deslizante
  (estructura de datos FIFO acotada) que filtra detecciones ruidosas.
"""

from __future__ import annotations

from collections import Counter, deque

from .types import Command


class ControladorLinea:
    """Convierte el error lateral en píxeles a un comando discreto estable."""

    def __init__(
        self,
        umbral_entrada: int = 25,
        umbral_salida: int = 12,
        ventana: int = 5,
    ) -> None:
        self.umbral_entrada = umbral_entrada
        self.umbral_salida = umbral_salida
        self.__historial: deque[Command] = deque(maxlen=ventana)
        self.__giro_activo: Command | None = None

    def reiniciar(self) -> None:
        self.__historial.clear()
        self.__giro_activo = None

    def decidir(self, error_px: int) -> Command:
        """Retorna el comando votado para el error observado."""
        crudo = self.__crudo(error_px)
        self.__historial.append(crudo)
        return self.__votar(crudo)

    def __crudo(self, error_px: int) -> Command:
        """Decisión instantánea con banda de histéresis."""
        if self.__giro_activo is None:
            if error_px <= -self.umbral_entrada:
                self.__giro_activo = Command.LEFT
            elif error_px >= self.umbral_entrada:
                self.__giro_activo = Command.RIGHT
        elif abs(error_px) < self.umbral_salida:
            self.__giro_activo = None
        if self.__giro_activo is not None:
            return self.__giro_activo
        return Command.UP

    def __votar(self, crudo: Command) -> Command:
        """Mayoría de la ventana; en empate gana el comando más reciente."""
        conteo = Counter(self.__historial)
        maximo = max(conteo.values())
        ganadores = [comando for comando, n in conteo.items() if n == maximo]
        return ganadores[0] if len(ganadores) == 1 else crudo
