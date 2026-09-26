"""Tests del K-Means implementado desde cero."""

from __future__ import annotations

import numpy as np

from first.core.vision.kmeans import asignar, kmeans


def test_separa_tres_grupos_de_luminancia() -> None:
    rng = np.random.default_rng(3)
    oscuros = rng.normal([40.0, 120.0, 35.0], 6.0, size=(100, 3))
    medios = rng.normal([60.0, 110.0, 120.0], 6.0, size=(100, 3))
    claros = rng.normal([20.0, 100.0, 215.0], 6.0, size=(100, 3))
    puntos = np.vstack([oscuros, medios, claros])

    centroides, etiquetas = kmeans(puntos, k=3)

    assert len(np.unique(etiquetas)) == 3
    v_por_cluster = centroides[:, 2]
    assert v_por_cluster.min() < 60.0  # el cluster oscuro existe
    assert v_por_cluster.max() > 170.0  # el cluster claro existe


def test_es_determinista() -> None:
    puntos = np.linspace([0, 0, 0], [179, 255, 255], num=300)
    c1, e1 = kmeans(puntos, k=3)
    c2, e2 = kmeans(puntos, k=3)
    assert np.allclose(c1, c2)
    assert np.array_equal(e1, e2)


def test_asignar_reutiliza_el_modelo() -> None:
    puntos = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [0.0, 0.0, 200.0]])
    centroides, _ = kmeans(puntos, k=2)
    etiquetas = asignar(puntos, centroides)
    assert etiquetas[0] == etiquetas[1]
    assert etiquetas[2] != etiquetas[0]
