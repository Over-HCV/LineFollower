"""Tests de la lógica pura de los widgets del panel en vivo."""

from __future__ import annotations

from first.adapters.sim.panel import Deslizador, SelectorGrupo


def test_deslizador_acota_a_los_extremos() -> None:
    d = Deslizador("t", 6.0, 30.0, 99.0, paso=1.0)
    assert d.valor == 30.0
    d.fijar(-5.0)
    assert d.valor == 6.0


def test_deslizador_aplana_al_paso() -> None:
    d = Deslizador("t", 0.5, 10.0, 1.0, paso=0.5)
    assert d.fijar(3.3) == 3.5
    assert d.fijar(3.2) == 3.0


def test_deslizador_fraccion_ida_y_vuelta() -> None:
    d = Deslizador("t", 0.0, 100.0, 50.0, paso=1.0)
    assert d.fraccion == 0.5
    d.fijar_fraccion(0.0)
    assert d.valor == 0.0
    d.fijar_fraccion(2.0)  # fuera de rango se acota
    assert d.valor == 100.0


def test_deslizador_callback_solo_al_cambiar() -> None:
    valores: list[float] = []
    d = Deslizador("t", 0.0, 10.0, 5.0, al_cambiar=valores.append)
    d.fijar(5.0)  # sin cambio: no dispara
    assert valores == []
    d.fijar(7.0)
    d.fijar(7.0)
    assert valores == [7.0]


def test_deslizador_ajuste_relativo() -> None:
    d = Deslizador("t", 0.0, 10.0, 4.0, paso=1.0)
    d.ajustar(3.0)
    assert d.valor == 7.0
    d.ajustar(-10.0)
    assert d.valor == 0.0


def test_selector_elige_y_avisa() -> None:
    elegidas: list[str] = []
    sel = SelectorGrupo("p", ["a", "b", "c"], 0, elegidas.append)
    assert sel.activa == "a"
    sel.seleccionar(2)
    assert sel.activa == "c"
    assert elegidas == ["c"]
    sel.seleccionar(2)  # repetir no dispara
    assert elegidas == ["c"]


def test_selector_cicla_con_wrap() -> None:
    sel = SelectorGrupo("p", ["a", "b", "c"], 1)
    assert sel.ciclo() == "c"
    assert sel.ciclo() == "a"


def test_selector_indice_envuelto() -> None:
    sel = SelectorGrupo("p", ["a", "b", "c"], 0)
    assert sel.seleccionar(99) == "a"  # 99 % 3 == 0
