"""Adaptador futuro de hardware: Command -> tramas por serial hacia el Arduino.

Protocolo mínimo acordado: un byte + newline por comando (U/R/D/L). El
firmware del carrito mapea cada trama a su preset de motoles. Requiere
`uv add pyserial` cuando llegue el hardware.
"""

from __future__ import annotations

from ..core.types import Command
from ..ports import CommandSink

TRAMAS: dict[Command, bytes] = {
    Command.UP: b"U\n",
    Command.RIGHT: b"R\n",
    Command.DOWN: b"D\n",
    Command.LEFT: b"L\n",
}


class SerialArduinoSink(CommandSink):
    """Envía los comandos discretos al carrito real por puerto serial."""

    def __init__(self, puerto: str = "/dev/ttyUSB0", baudios: int = 9600) -> None:
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError("Falta pyserial: ejecuta 'uv add pyserial'") from exc
        self.__ser = serial.Serial(puerto, baudios)

    def send(self, command: Command) -> None:
        self.__ser.write(TRAMAS[command])

    def close(self) -> None:
        self.__ser.close()
