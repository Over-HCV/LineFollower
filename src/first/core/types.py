"""Tipos fundamentales del cerebro del robot seguidor de línea."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class Command(str, Enum):
    """Las cuatro señales discretas que acepta el carrito (una a la vez)."""

    UP = "up"
    RIGHT = "right"
    DOWN = "down"
    LEFT = "left"


class EstadoRobot(str, Enum):
    """Estados de la máquina de estados del cerebro."""

    SIGUIENDO = "siguiendo"
    PARE = "pare"
    RECUPERANDO = "recuperando"
    PERDIDO = "perdido"


class Senal(str, Enum):
    """Señales de tránsito que el robot debe reconocer."""

    PARE = "pare"
    SIGA = "siga"


@dataclass(frozen=True, slots=True)
class LineInfo:
    """Observación de la línea guía dentro de un frame."""

    presente: bool
    error_px: int = 0
    error_mirada_px: int = 0
    objetivo_px: int = 0
    cx: int = 0
    cy: int = 0
    area: float = 0.0
    mascara: NDArray[np.uint8] | None = None


@dataclass(frozen=True, slots=True)
class DeteccionSenal:
    """Resultado de la etapa de reconocimiento de señales."""

    senal: Senal | None
    caja: tuple[int, int, int, int] | None
    area: float = 0.0


@dataclass(frozen=True, slots=True)
class Telemetry:
    """Salida extendida del cerebro para depuración y métricas."""

    comando: Command
    estado: EstadoRobot
    senal_confirmada: Senal | None
    senal_cruda: Senal | None
    linea: LineInfo
    debug: NDArray[np.uint8] | None = None
