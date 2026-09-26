"""Render del trazo imperfecto: ancho variable, temblor, huecos y tramos partidos.

Es el dibujo de la línea "hecha a mano": el ancho y el desvío se modulan con
dos senoidales de fase aleatoria (nada de ruido blanco, que daría un borde
serrucho irreal) y los huecos y tramos partidos se colocan solo en zonas
rectas y lejos de las señales y de la meta.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .defectos import dibujar_manchas, dibujar_reflejos
from .lienzo import COLOR_LINEA, COLOR_LINEA_CLARA
from .pista import Pista
from .realismo import Realismo


def renderizar_pista(
    lienzo: NDArray[np.uint8],
    pista: Pista,
    r: Realismo,
    indices_signos: Sequence[int],
    rng: np.random.Generator,
) -> None:
    """Pinta la pista imperfecta: manchas, luego el trazo, luego los reflejos."""
    dibujar_manchas(lienzo, pista, r, indices_signos, rng)
    __dibujar_trazo(lienzo, pista, r, indices_signos, rng)
    dibujar_reflejos(lienzo, pista, r, indices_signos, rng)


def __dibujar_trazo(
    lienzo: NDArray[np.uint8],
    pista: Pista,
    r: Realismo,
    indices_signos: Sequence[int],
    rng: np.random.Generator,
) -> None:
    """Dibuja la línea por tramos: ancho variable, temblor y huecos."""
    p = pista.puntos
    n = len(p)
    cerrado = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(cerrado, axis=0), axis=1)
    paso = float(seg.mean())
    s_arc = np.concatenate([[0.0], np.cumsum(seg)])[:n]
    largo_total = float(seg.sum())
    t_norm = s_arc / largo_total

    f1, f2 = rng.uniform(2.0, 4.0), rng.uniform(5.0, 9.0)
    p1, p2 = rng.uniform(0.0, 2.0 * np.pi, 2)
    modulacion = (
        0.5
        + 0.35 * np.sin(2.0 * np.pi * f1 * t_norm + p1)
        + 0.15 * np.sin(2.0 * np.pi * f2 * t_norm + p2)
    )
    ancho = r.ancho_min + (r.ancho_max - r.ancho_min) * np.clip(modulacion, 0, 1)

    f3, f4 = rng.uniform(3.0, 6.0), rng.uniform(8.0, 14.0)
    p3, p4 = rng.uniform(0.0, 2.0 * np.pi, 2)
    desvio = r.jitter * (
        0.6 * np.sin(2.0 * np.pi * f3 * t_norm + p3)
        + 0.4 * np.sin(2.0 * np.pi * f4 * t_norm + p4)
    )

    tangentes = cerrado[1:] - cerrado[:-1]
    normas = np.linalg.norm(tangentes, axis=1, keepdims=True)
    normas[normas == 0.0] = 1.0
    tangentes = tangentes / normas
    normales = np.stack([-tangentes[:, 1], tangentes[:, 0]], axis=1)
    puntos_mod = p + normales * desvio[:, None]

    hueco = __generar_huecos(pista, r, indices_signos, rng, paso, largo_total)
    partido = __tramos_aleatorios(
        pista,
        indices_signos,
        rng,
        r.tramos_partidos,
        r.partido_largo_min,
        r.partido_largo_max,
        paso,
    )
    if r.degradado_fraccion > 0.0:
        cantidad_degradada = max(
            1, int(round(r.degradado_fraccion * largo_total / 175.0))
        )
    else:
        cantidad_degradada = 0
    degradado = __tramos_aleatorios(
        pista, indices_signos, rng, cantidad_degradada, 100.0, 250.0, paso
    )

    for i in range(n):
        if hueco[i]:
            continue
        j = (i + 1) % n
        if hueco[j]:
            continue
        a = puntos_mod[i]
        b = puntos_mod[j]
        color = COLOR_LINEA_CLARA if degradado[i] else COLOR_LINEA
        if partido[i]:
            # Línea partida en dos trazos paralelos (marcador que raya):
            # el segmento dominante debe elegir uno y el sesgo queda
            # dentro de la banda muerta del control.
            separacion = ancho[i] * 0.30
            sub_ancho = max(5, int(round(ancho[i] * 0.38)))
            for lado in (-1.0, 1.0):
                cv2.line(
                    lienzo,
                    (
                        int(round(a[0] + normales[i, 0] * lado * separacion)),
                        int(round(a[1] + normales[i, 1] * lado * separacion)),
                    ),
                    (
                        int(round(b[0] + normales[j, 0] * lado * separacion)),
                        int(round(b[1] + normales[j, 1] * lado * separacion)),
                    ),
                    color,
                    sub_ancho,
                )
        else:
            cv2.line(
                lienzo,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                color,
                thickness=max(1, int(round(ancho[i]))),
            )


def __zonas_excluidas(
    pista: Pista, indices_signos: Sequence[int], paso: float
) -> NDArray[np.bool_]:
    """Vecindad de las señales y zona de meta: ahí el trazo se deja limpio."""
    n = len(pista.puntos)
    excluidos = np.zeros(n, dtype=np.bool_)
    radio = max(1, int(90.0 / paso))
    for centro in indices_signos:
        for k in range(-radio, radio + 1):
            excluidos[(centro + k) % n] = True
    for k in range(-40, 41):
        excluidos[k % n] = True  # zona de meta
    return excluidos


def __generar_huecos(
    pista: Pista,
    r: Realismo,
    indices_signos: Sequence[int],
    rng: np.random.Generator,
    paso: float,
    largo_total: float,
) -> NDArray[np.bool_]:
    """Huecos solo en zonas rectas, lejos de señales y de la meta."""
    p = pista.puntos
    n = len(p)
    hueco = np.zeros(n, dtype=np.bool_)
    if r.gap_fraccion <= 0.0:
        return hueco

    d1 = p - np.roll(p, 1, axis=0)
    d2 = np.roll(p, -1, axis=0) - p
    cos = (d1 * d2).sum(axis=1) / (
        np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1)
    )
    angulo = np.arccos(np.clip(cos, -1.0, 1.0))
    curvatura = np.convolve(angulo, np.ones(31) / 31.0, mode="same")
    permitido = curvatura < (paso / 420.0)  # solo zonas rectas (r >= ~420px)
    permitido &= ~__zonas_excluidas(pista, indices_signos, paso)

    objetivo = r.gap_fraccion * largo_total
    colocado = 0.0
    candidatos = np.where(permitido)[0]
    margen = max(1, int(120.0 / paso))
    while colocado < objetivo and len(candidatos) > 0:
        i0 = int(rng.choice(candidatos))
        largo = float(rng.uniform(r.gap_largo_min, r.gap_largo_max))
        extension = min(int(largo / paso) + 1, n // 6)
        hueco[i0 : i0 + extension] = True
        colocado += largo
        candidatos = candidatos[
            (candidatos < i0 - margen) | (candidatos > i0 + extension + margen)
        ]
    return hueco


def __tramos_aleatorios(
    pista: Pista,
    indices_signos: Sequence[int],
    rng: np.random.Generator,
    cantidad: int,
    largo_min: float,
    largo_max: float,
    paso: float,
) -> NDArray[np.bool_]:
    """Tramos aleatorios de la pista, lejos de señales y de la meta."""
    n = len(pista.puntos)
    mascara = np.zeros(n, dtype=np.bool_)
    if cantidad <= 0:
        return mascara
    candidatos = np.where(~__zonas_excluidas(pista, indices_signos, paso))[0]
    margen = max(1, int(100.0 / paso))
    colocados = 0
    intentos = 0
    while colocados < cantidad and intentos < cantidad * 20 and len(candidatos) > 0:
        intentos += 1
        i0 = int(rng.choice(candidatos))
        extension = int(rng.uniform(largo_min, largo_max) / paso) + 1
        mascara[i0 : i0 + extension] = True
        colocados += 1
        candidatos = candidatos[
            (candidatos < i0 - margen) | (candidatos > i0 + extension + margen)
        ]
    return mascara


