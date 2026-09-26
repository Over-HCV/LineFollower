"""Binarizado de la ROI: qué píxeles son línea y cuáles no.

Dos caminos con el mismo criterio de fondo —la línea es oscura Y desaturada—:
K-Means propio sobre HSV (se adapta a la iluminación) y umbral fijo sobre el
gris (rápido y predecible, y el plan B cuando el K-Means no encuentra nada).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .kmeans import asignar, kmeans
from .params import ParamsLinea

Mascara = NDArray[np.uint8]


def __limpiar(mascara: Mascara) -> Mascara:
    # Kernel 3x3: un kernel mayor borra trazos delgados reales (ancho variable).
    kernel = np.ones((3, 3), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)


def __hsv_puntos(roi: NDArray[np.uint8]) -> NDArray[np.float64]:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float64)
    return hsv.reshape(-1, 3)


def ajustar_modelo(
    roi: NDArray[np.uint8], params: ParamsLinea
) -> tuple[NDArray[np.float64], int]:
    """Ajusta K-Means sobre una muestra y elige el cluster de la línea.

    La línea es oscura Y desaturada (S baja); las señales de color son
    oscuras en V pero saturadas, así que se excluyen con el criterio S.
    """
    puntos = __hsv_puntos(roi)
    muestra = puntos[:: params.submuestreo]
    centroides, _ = kmeans(muestra, k=params.k_kmeans)
    candidatos = [
        i for i in range(len(centroides)) if centroides[i, 1] < params.saturacion_maxima
    ]
    if not candidatos:
        return centroides, -1
    cluster_linea = min(candidatos, key=lambda i: centroides[i, 2])
    return centroides, cluster_linea


def mascara_kmeans(
    roi: NDArray[np.uint8], centroides: NDArray[np.float64], cluster_linea: int
) -> Mascara:
    if cluster_linea < 0:
        return np.zeros(roi.shape[:2], dtype=np.uint8)
    alto, ancho = roi.shape[:2]
    etiquetas = asignar(__hsv_puntos(roi), centroides)
    binaria = (etiquetas == cluster_linea).astype(np.uint8) * 255
    return __limpiar(binaria.reshape(alto, ancho))


def mascara_umbral(
    roi: NDArray[np.uint8], params: ParamsLinea
) -> Mascara:
    """Umbral fijo sobre el gris, descartando lo saturado.

    El filtro de saturación no es un adorno: un PARE rojo (BGR 0,0,210) pesa
    63 en gris, por debajo del umbral, y sin él la señal entera entra en la
    máscara como si fuera línea. La línea real es oscura Y desaturada, el
    mismo criterio que usa el modo K-Means.
    """
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaria = cv2.threshold(blur, params.umbral_gris, 255, cv2.THRESH_BINARY_INV)
    saturacion = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
    binaria[saturacion >= params.saturacion_maxima] = 0
    return __limpiar(binaria)
