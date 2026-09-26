"""Carrera headless (sin UI): el mismo loop del simulador, útil para tests y tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter

from ...core.brain import Brain
from ...core.types import Command, EstadoRobot
from .camera import CamaraSintetica
from .world import Mundo


@dataclass(frozen=True, slots=True)
class MetricasCarrera:
    """Métricas estilo rúbrica tras una corrida simulada."""

    duracion: float
    vueltas: int
    descarrilamientos: int
    pares_cumplidos: int
    tiempo_primera_vuelta: float | None
    comandos: dict[str, int] = field(default_factory=dict)


def simular_carrera(
    brain: Brain,
    mundo: Mundo | None = None,
    camara: CamaraSintetica | None = None,
    duracion: float = 60.0,
    dt: float = 1.0 / 60.0,
) -> MetricasCarrera:
    """Ejecuta el ciclo cerrado cámara -> cerebro -> física sin render de UI."""
    mundo = mundo or Mundo()
    camara = camara or CamaraSintetica()
    brain.reiniciar()

    previo: EstadoRobot = EstadoRobot.SIGUIENDO
    pares_cumplidos = 0
    tiempo_vuelta: float | None = None
    vueltas_iniciales = mundo.vueltas
    conteo: Counter[Command] = Counter()

    pasos = int(duracion / dt)
    for _ in range(pasos):
        frame = camara.render(mundo, mundo.carrito)
        comando, telemetria = brain.procesar(frame)
        conteo[comando] += 1
        mundo.paso(comando, dt)
        if previo is EstadoRobot.PARE and telemetria.estado is not EstadoRobot.PARE:
            pares_cumplidos += 1
        previo = telemetria.estado
        if tiempo_vuelta is None and mundo.vueltas > vueltas_iniciales:
            tiempo_vuelta = mundo.t

    return MetricasCarrera(
        duracion=mundo.t,
        vueltas=mundo.vueltas - vueltas_iniciales,
        descarrilamientos=mundo.descarrilamientos,
        pares_cumplidos=pares_cumplidos,
        tiempo_primera_vuelta=tiempo_vuelta,
        comandos={c.value: n for c, n in conteo.items()},
    )
