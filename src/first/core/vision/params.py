"""Parámetros de la detección de línea (datos, sin lógica).

Viven aparte porque los comparten el binarizado, la selección de segmento y
la superposición de depuración, que son módulos distintos.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParamsLinea:
    """Parámetros de la etapa de detección de línea."""

    roi_cerca: tuple[float, float] = (0.72, 1.0)
    roi_lejos: tuple[float, float] = (0.40, 0.72)
    umbral_gris: int = 70
    saturacion_maxima: float = 120.0  # más saturado = señal de color, no línea
    k_kmeans: int = 3
    submuestreo: int = 6
    masa_minima: float = 350.0  # masa mínima del segmento (trazos delgados)
    masa_maxima: float = 5000.0  # demasiada línea visible = no es línea
    peso_mirada: float = 0.45
    anticipacion_maxima: float = 30.0
    ancho_far_maximo: float = 90.0  # bbox más ancho = línea cruzando
    salto_far_maximo: float = 160.0  # anticipación descartada si salta más (px)
    margen_lateral: int = 20  # cortina: columnas ignoradas a cada lado (px)
    sigma_atencion: float = 0.18  # σ de la atención horizontal (fracción del ancho)
    peso_cobertura: float = 1.5  # cuánto pesa cruzar la franja entera
    bono_fondo: float = 1.6  # premio al segmento que toca el borde inferior
    peso_fila: float = 3.0  # peso de la fila inferior frente a la superior
    modo: str = "kmeans"  # "kmeans" | "umbral"
