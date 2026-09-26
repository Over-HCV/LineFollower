"""K-Means básico (Lloyd) implementado desde cero con numpy.

Inicialización determinista por cuantiles de la componente V (luminancia
en HSV): reproducible entre ejecuciones y sin semillas aleatorias.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Matriz = NDArray[np.float64]
Etiquetas = NDArray[np.int64]


def __centroides_iniciales(puntos: Matriz, k: int) -> Matriz:
    """Ordena por V (última columna) y promedia k bloques equiprobables."""
    orden = np.argsort(puntos[:, -1])
    bloques = np.array_split(puntos[orden], k)
    return np.stack([bloque.mean(axis=0) for bloque in bloques])


def __reasignar_vacios(puntos: Matriz, centroides: Matriz, etiquetas: Etiquetas) -> None:
    """Cluster vacío -> se le entrega el punto más lejano a su centro."""
    distancias = ((puntos[:, None, :] - centroides[None, :, :]) ** 2).sum(axis=2)
    for i in range(len(centroides)):
        if np.count_nonzero(etiquetas == i) == 0:
            centroides[i] = puntos[distancias.min(axis=1).argmax()]


def kmeans(puntos: Matriz, k: int = 3, iteraciones: int = 8) -> tuple[Matriz, Etiquetas]:
    """Ajusta k centroides sobre puntos (N, C). Retorna (centroides, etiquetas)."""
    centroides = __centroides_iniciales(puntos, k)
    etiquetas = np.full(len(puntos), -1, dtype=np.int64)
    for _ in range(iteraciones):
        distancias = ((puntos[:, None, :] - centroides[None, :, :]) ** 2).sum(axis=2)
        nuevas = distancias.argmin(axis=1)
        if np.array_equal(nuevas, etiquetas):
            break
        etiquetas = nuevas
        __reasignar_vacios(puntos, centroides, etiquetas)
        for i in range(k):
            grupo = puntos[etiquetas == i]
            if len(grupo) > 0:
                centroides[i] = grupo.mean(axis=0)
    return centroides, etiquetas


def asignar(puntos: Matriz, centroides: Matriz) -> Etiquetas:
    """Etiqueta cada punto con el centroide más cercano (modelo ya ajustado)."""
    distancias = ((puntos[:, None, :] - centroides[None, :, :]) ** 2).sum(axis=2)
    return distancias.argmin(axis=1)
