"""Puertos (interfaces) que desacoplan el cerebro de simulador y hardware."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .core.types import Command


class FrameSource(ABC):
    """Fuente de frames BGR: webcam, video de ensayo o cámara sintética."""

    @abstractmethod
    def read(self) -> NDArray[np.uint8] | None:
        """Retorna el siguiente frame BGR, o None si no hay disponible."""

    @abstractmethod
    def release(self) -> None:
        """Libera los recursos de la fuente."""


class CommandSink(ABC):
    """Destino de comandos: simulador, ventana o serial hacia el Arduino."""

    @abstractmethod
    def send(self, command: Command) -> None:
        """Envía un comando discreto al actuador."""

    @abstractmethod
    def close(self) -> None:
        """Cierra el canal de comandos."""
