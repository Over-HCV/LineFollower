"""Detección de la línea guía: histograma de columnas + anticipación opcional.

Franja cercana (la línea justo delante del robot): se binariza con K-Means
(cluster más oscuro) y se calcula el centroide con un histograma de columnas
ponderado por fila — las filas inferiores pesan más porque es la línea que
el robot está pisando. Es inmune a codos y cruces: siempre apunta al tramo
que sale de debajo del robot.

Franja lejana (anticipación): solo se usa si la línea detectada 'corre'
verticalmente (bbox angosto). En curvas cerradas la línea lejana aparece
horizontal y su centroide dejaría de ser una anticipación válida.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from ..types import LineInfo
from .kmeans import asignar, kmeans

Mascara = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ParamsLinea:
    """Parámetros de la etapa de detección de línea."""

    roi_cerca: tuple[float, float] = (0.72, 1.0)
    roi_lejos: tuple[float, float] = (0.40, 0.72)
    umbral_gris: int = 70
    k_kmeans: int = 3
    submuestreo: int = 6
    masa_minima: float = 350.0  # masa mínima del segmento (trazos delgados)
    masa_maxima: float = 5000.0  # demasiada línea visible = no es línea
    peso_mirada: float = 0.45
    anticipacion_maxima: float = 30.0
    ancho_far_maximo: float = 90.0  # bbox más ancho = línea cruzando
    margen_lateral: int = 0  # cortina: columnas ignoradas a cada lado (px)
    modo: str = "kmeans"  # "kmeans" | "umbral"


def __recortar(
    frame: NDArray[np.uint8], franja: tuple[float, float]
) -> NDArray[np.uint8]:
    alto = frame.shape[0]
    y0 = int(alto * franja[0])
    y1 = int(alto * franja[1])
    return frame[y0:y1, :]


def __limpiar(mascara: Mascara) -> Mascara:
    # Kernel 3x3: un kernel mayor borra trazos delgados reales (ancho variable).
    kernel = np.ones((3, 3), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)


def __hsv_puntos(roi: NDArray[np.uint8]) -> NDArray[np.float64]:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float64)
    return hsv.reshape(-1, 3)


def __ajustar_modelo(
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
        i for i in range(len(centroides)) if centroides[i, 1] < 120.0
    ]
    if not candidatos:
        return centroides, -1
    cluster_linea = min(candidatos, key=lambda i: centroides[i, 2])
    return centroides, cluster_linea


def __mascara_kmeans(
    roi: NDArray[np.uint8], centroides: NDArray[np.float64], cluster_linea: int
) -> Mascara:
    if cluster_linea < 0:
        return np.zeros(roi.shape[:2], dtype=np.uint8)
    alto, ancho = roi.shape[:2]
    etiquetas = asignar(__hsv_puntos(roi), centroides)
    binaria = (etiquetas == cluster_linea).astype(np.uint8) * 255
    return __limpiar(binaria.reshape(alto, ancho))


def __mascara_umbral(
    roi: NDArray[np.uint8], params: ParamsLinea
) -> Mascara:
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaria = cv2.threshold(blur, params.umbral_gris, 255, cv2.THRESH_BINARY_INV)
    return __limpiar(binaria)


def __segmento_dominante(
    mascara: Mascara, params: ParamsLinea
) -> tuple[int, float, float, int] | None:
    """Centroide del segmento de línea dominante, al estilo sensor IR.

    1. Escanea las filas inferiores (la línea que el robot pisa) y arma el
       histograma de columnas; los grupos contiguos son segmentos candidatos.
    2. Elige por (cobertura vertical, masa, cercanía al centro): una línea
       real atraviesa toda la franja de arriba a abajo, mientras que una
       mancha de suciedad es compacta y no la cubre.
    3. Calcula el centroide ponderado por fila solo en ese segmento.

    Así, en un codo con dos segmentos visibles nunca promedia entre ambos,
    y las manchas oscuras del piso no suplantan a la línea.
    """
    alto, ancho = mascara.shape[:2]
    banda = mascara[alto - max(4, alto // 4) :, :]  # ~25% inferior
    columnas_fondo = (banda.astype(np.float32) / 255.0).sum(axis=0)

    umbral = 0.5 * columnas_fondo.max() if columnas_fondo.max() > 0 else 0.0
    if umbral <= 0:
        return None
    encima = columnas_fondo >= umbral
    grupos: list[tuple[int, int]] = []
    inicio = None
    for x in range(ancho):
        if encima[x] and inicio is None:
            inicio = x
        elif not encima[x] and inicio is not None:
            grupos.append((inicio, x - 1))
            inicio = None
    if inicio is not None:
        grupos.append((inicio, ancho - 1))
    if not grupos:
        return None

    pesos_filas = np.linspace(1.0, 3.0, alto, dtype=np.float32).reshape(alto, 1)
    histograma = (mascara.astype(np.float32) / 255.0) * pesos_filas
    columnas = histograma.sum(axis=0)
    indices = np.arange(ancho, dtype=np.float64)

    centro = ancho // 2

    def criterio(x0: int, x1: int) -> tuple[int, float, float]:
        masa = float(columnas[x0 : x1 + 1].sum())
        cobertura = int(np.count_nonzero((mascara[:, x0 : x1 + 1] > 0).any(axis=1)))
        cercania = -abs((x0 + x1) / 2.0 - centro)
        return cobertura, masa, cercania

    x0, x1 = max(grupos, key=lambda g: criterio(g[0], g[1]))
    masa = float(columnas[x0 : x1 + 1].sum())
    if masa < params.masa_minima or masa > params.masa_maxima:
        return None
    filas_cubiertas = np.count_nonzero((mascara[:, x0 : x1 + 1] > 0).any(axis=1))
    cobertura = float(filas_cubiertas) / float(alto)
    cx = float((columnas[x0 : x1 + 1] * indices[x0 : x1 + 1]).sum() / masa)
    return int(cx), masa, cobertura, int(x1 - x0 + 1)


def __centroide_far(
    mascara: Mascara, params: ParamsLinea
) -> int | None:
    """Centroide x de la línea lejana, solo si corre vertical (bbox angosto)."""
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contornos:
        return None
    mayor = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(mayor) < params.masa_minima * 0.5:
        return None
    _, _, w, _ = cv2.boundingRect(mayor)
    if w > params.ancho_far_maximo:
        return None
    momentos = cv2.moments(mayor)
    if momentos["m00"] == 0:
        return None
    return int(momentos["m10"] / momentos["m00"])


def detectar_linea(
    frame: NDArray[np.uint8], params: ParamsLinea | None = None
) -> LineInfo:
    """Localiza la línea bajo el robot y calcula el objetivo de dirección.

    La cortina lateral (``margen_lateral``) recorta columnas simétricas a
    cada lado ANTES de procesar: una línea "fantasma" en el borde (p. ej. el
    tramo anterior de la pista en una chicane) deja de existir para el
    detector. Las coordenadas devueltas se trasladan de vuelta al espacio
    del frame completo para que el error quede referido al mismo centro.
    """
    p = params or ParamsLinea()
    margen = max(0, min(int(p.margen_lateral), frame.shape[1] // 2 - 20))
    if margen:
        frame = frame[:, margen : frame.shape[1] - margen]
    ancho = frame.shape[1]
    centro_frame = ancho // 2

    roi_cerca = __recortar(frame, p.roi_cerca)
    if p.modo == "kmeans":
        centroides, cluster_linea = __ajustar_modelo(roi_cerca, p)
        mascara_cerca = __mascara_kmeans(roi_cerca, centroides, cluster_linea)
    else:
        mascara_cerca = __mascara_umbral(roi_cerca, p)

    cerca = __segmento_dominante(mascara_cerca, p)
    if cerca is None and p.modo == "kmeans":
        mascara_cerca = __mascara_umbral(roi_cerca, p)
        cerca = __segmento_dominante(mascara_cerca, p)
    if cerca is None:
        return LineInfo(presente=False, mascara=mascara_cerca)

    cx, _masa, cobertura, ancho_segmento = cerca

    roi_lejos = __recortar(frame, p.roi_lejos)
    if p.modo == "kmeans":
        mascara_lejos = __mascara_kmeans(roi_lejos, centroides, cluster_linea)
    else:
        mascara_lejos = __mascara_umbral(roi_lejos, p)
    cx_far = __centroide_far(mascara_lejos, p)

    error = cx - centro_frame
    if cx_far is None:
        objetivo, error_mirada = cx, 0
    else:
        anticipacion = max(
            -p.anticipacion_maxima,
            min(p.anticipacion_maxima, p.peso_mirada * (cx_far - cx)),
        )
        objetivo = int(cx + anticipacion)
        error_mirada = cx_far - centro_frame

    return LineInfo(
        presente=True,
        error_px=error,
        error_mirada_px=error_mirada,
        objetivo_px=objetivo + margen,
        cx=cx + margen,
        cy=0,
        area=float(np.count_nonzero(mascara_cerca)),
        cobertura=cobertura,
        ancho_segmento=ancho_segmento,
        mascara=mascara_cerca,
    )
