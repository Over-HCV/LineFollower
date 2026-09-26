"""Tests de la persistencia de la sesión del simulador (JSON en disco)."""

from __future__ import annotations

from pathlib import Path

from first.adapters.sim import config
from first.adapters.sim.pista import PISTAS, Pista
from first.adapters.sim.world import Mundo, Signo
from first.core.types import Command, Senal


def crear_mundo() -> Mundo:
    return Mundo(
        pista=Pista(PISTAS["ovalo"]),
        signos=[Signo(0.20, Senal.PARE, lado=1), Signo(0.60, Senal.SIGA, lado=-1)],
    )


def test_ida_y_vuelta_de_la_disposicion(tmp_path: Path) -> None:
    ruta = tmp_path / "sim_config.json"
    original = crear_mundo()
    original.reposicionar_en_pista(0.42)
    original.mover_signo(0, 0.33, -1, 90.0)
    config.guardar(ruta, original, "ovalo", "alto")

    datos = config.cargar(ruta)
    assert datos is not None
    assert datos["pista"] == "ovalo" and datos["realismo"] == "alto"

    restaurado = crear_mundo()
    assert config.aplicar(restaurado, datos)
    assert restaurado.signos == original.signos
    assert restaurado.carrito == original.carrito


def test_cargar_sin_archivo_o_corrupto(tmp_path: Path) -> None:
    assert config.cargar(tmp_path / "no_existe.json") is None
    roto = tmp_path / "roto.json"
    roto.write_text("{esto no es json", encoding="utf-8")
    assert config.cargar(roto) is None
    lista = tmp_path / "lista.json"
    lista.write_text("[1, 2, 3]", encoding="utf-8")
    assert config.cargar(lista) is None


def test_datos_invalidos_no_rompen_el_mundo() -> None:
    mundo = crear_mundo()
    signos_previos = list(mundo.signos)
    pose_previa = (mundo.carrito.x, mundo.carrito.y, mundo.carrito.theta)

    aplicado = config.aplicar(
        mundo,
        {"signos": [{"s": 0.1, "tipo": "inexistente", "lado": 1}], "carrito": {"x": 1}},
    )
    assert not aplicado
    assert mundo.signos == signos_previos
    assert (mundo.carrito.x, mundo.carrito.y, mundo.carrito.theta) == pose_previa


def test_aplicar_resincroniza_el_progreso() -> None:
    mundo = crear_mundo()
    destino = mundo.pista.punto(0.75)
    config.aplicar(
        mundo, {"carrito": {"x": destino[0], "y": destino[1], "theta": 0.0}}
    )
    # El salto no debe contarse como vuelta ni como descarrilamiento.
    assert abs(mundo.progreso() - 0.75) < 0.02
    mundo.paso(Command.UP, 0.016)
    assert mundo.vueltas == 0
    assert mundo.descarrilamientos == 0
