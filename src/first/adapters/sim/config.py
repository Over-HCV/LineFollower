"""Persistencia de la sesión del simulador (posiciones y selección).

Guarda en un JSON la última disposición: pista y realismo activos, pose del
carrito y lay-out de señales. Se escribe al soltar un arrastre y al salir;
se lee al arrancar (los argumentos explícitos del CLI tienen prioridad).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...core.types import Senal
from .world import Mundo, Signo

RUTA_PREDETERMINADA: Path = Path("sim_config.json")


def guardar(ruta: Path, mundo: Mundo, pista: str, realismo: str) -> None:
    """Escribe la configuración actual de la sesión."""
    datos: dict[str, Any] = {
        "pista": pista,
        "realismo": realismo,
        "carrito": {
            "x": mundo.carrito.x,
            "y": mundo.carrito.y,
            "theta": mundo.carrito.theta,
        },
        "signos": [
            {
                "s": signo.s,
                "tipo": signo.tipo.value,
                "lado": signo.lado,
                "offset": signo.offset,
            }
            for signo in mundo.signos
        ],
    }
    ruta.write_text(json.dumps(datos, indent=2), encoding="utf-8")


def cargar(ruta: Path) -> dict[str, Any] | None:
    """Lee la configuración guardada; None si no existe o está corrupta."""
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return datos if isinstance(datos, dict) else None


def aplicar(mundo: Mundo, datos: dict[str, Any]) -> bool:
    """Restaura señales y carrito sobre un mundo ya construido.

    Retorna True si se aplicó algo. Ante cualquier campo inválido se ignora
    esa parte (el simulador arranca con los valores por defecto).
    """
    aplicado = False
    signos_datos = datos.get("signos")
    if isinstance(signos_datos, list) and signos_datos:
        signos: list[Signo] = []
        valido = True
        for dato in signos_datos:
            if not isinstance(dato, dict):
                valido = False
                break
            try:
                signos.append(
                    Signo(
                        s=float(dato["s"]) % 1.0,
                        tipo=Senal(str(dato["tipo"])),
                        lado=-1 if int(dato["lado"]) < 0 else 1,
                        offset=float(dato.get("offset", 70.0)),
                    )
                )
            except (KeyError, TypeError, ValueError):
                valido = False
                break
        if valido:
            mundo.establecer_signos(signos)
            aplicado = True

    carrito = datos.get("carrito")
    if isinstance(carrito, dict):
        try:
            mundo.reposicionar(
                float(carrito["x"]), float(carrito["y"]), float(carrito["theta"])
            )
            aplicado = True
        except (KeyError, TypeError, ValueError):
            pass
    return aplicado
