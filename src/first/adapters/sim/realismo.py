"""Preajustes de realismo: cuánta imperfección tiene el trazo de la pista.

Solo datos; el render vive en ``trazo.py`` y ``defectos.py``, que consumen
estos parámetros.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Realismo:
    """Imperfecciones del trazo real aplicadas al render de la pista.

    - ancho variable y jitter: el trazo humano no es uniforme ni recto.
    - huecos: trazo discontinuo (el marcador se corta).
    - manchas: suciedad oscura cerca de la línea (falsos candidatos).
    - tramos partidos: la línea se divide en 2 trazos paralelos (marcador
      que raya o cinta rasgada).
    - reflejos: brillo especular blanco SOBRE la línea (la rompe visualmente).
    - degradado: tramos con marcador descargado (color más claro).
    """

    ancho_min: float = 26.0
    ancho_max: float = 26.0
    jitter: float = 0.0  # desvío perpendicular suave del trazo (px)
    gap_fraccion: float = 0.0  # fracción de la longitud total con huecos
    gap_largo_min: float = 15.0
    gap_largo_max: float = 40.0
    manchas: int = 0
    mancha_radio_min: float = 10.0
    mancha_radio_max: float = 26.0
    mancha_lateral_min: float = 32.0
    reflejos: int = 0  # brillos blancos sobre la línea
    reflejo_largo_min: float = 25.0
    reflejo_largo_max: float = 70.0
    reflejo_ancho_max: float = 14.0
    tramos_partidos: int = 0  # tramos con la línea partida en paralelo
    partido_largo_min: float = 80.0
    partido_largo_max: float = 200.0
    degradado_fraccion: float = 0.0  # fracción con marcador descargado
    semilla: int = 11


REALISMOS: dict[str, Realismo | None] = {
    "perfecto": None,
    "medio": Realismo(
        ancho_min=18.0,
        ancho_max=26.0,
        jitter=3.0,
        reflejos=2,
    ),
    "alto": Realismo(
        ancho_min=15.0,
        ancho_max=28.0,
        jitter=6.0,
        gap_fraccion=0.04,
        gap_largo_min=15.0,
        gap_largo_max=35.0,
        manchas=6,
        mancha_radio_min=8.0,
        mancha_radio_max=20.0,
        mancha_lateral_min=45.0,
        reflejos=3,
        reflejo_largo_max=55.0,
        tramos_partidos=1,
        degradado_fraccion=0.08,
    ),
    "extremo": Realismo(
        ancho_min=12.0,
        ancho_max=30.0,
        jitter=10.0,
        gap_fraccion=0.10,
        gap_largo_min=20.0,
        gap_largo_max=60.0,
        manchas=15,
        reflejos=6,
        tramos_partidos=3,
        degradado_fraccion=0.15,
    ),
}
