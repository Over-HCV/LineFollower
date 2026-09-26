"""Mundo simulado: pistas cerradas (spline Catmull-Rom), señales y carrito.

La física es deliberadamente simple: cada comando discreto es un preset
(velocidad, giro) de un modelo tipo diferencial que gira mientras avanza.
La misma tabla de presets es la que se mapearía al Arduino real.

El realismo de la pista modela imperfecciones del trazo real: ancho
variable, temblor del trazo, huecos (trazo discontinuo) y manchas de
suciedad oscuras cerca de la línea.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from ...core.types import Command, Senal

LIENZO_ANCHO: int = 1600
LIENZO_ALTO: int = 1200

COLOR_SUELO: tuple[int, int, int] = (205, 205, 205)
COLOR_LINEA: tuple[int, int, int] = (40, 40, 40)
COLOR_LINEA_CLARA: tuple[int, int, int] = (98, 94, 90)  # marcador descargado
COLOR_SENAL: dict[Senal, tuple[int, int, int]] = {
    Senal.PARE: (0, 0, 210),  # BGR
    Senal.SIGA: (0, 190, 0),
}

PUNTOS_CONTROL: list[tuple[float, float]] = [
    (1320.0, 600.0),
    (1200.0, 1000.0),
    (800.0, 1120.0),
    (464.0, 936.0),
    (280.0, 600.0),
    (400.0, 200.0),
    (800.0, 80.0),
    (1136.0, 264.0),
]


def __pista_ovalo() -> list[tuple[float, float]]:
    return [
        (
            800.0 + 520.0 * math.cos(k * 2.0 * math.pi / 12.0),
            600.0 + 470.0 * math.sin(k * 2.0 * math.pi / 12.0),
        )
        for k in range(12)
    ]


def __pista_ocho() -> list[tuple[float, float]]:
    """Lemniscata: la línea se cruza a sí misma en el centro (como el ocho)."""
    return [
        (
            800.0 + 560.0 * math.sin(k * 2.0 * math.pi / 16.0),
            600.0 + 430.0 * math.sin(2.0 * k * 2.0 * math.pi / 16.0),
        )
        for k in range(16)
    ]


PISTAS: dict[str, list[tuple[float, float]]] = {
    "cacahuate": PUNTOS_CONTROL,
    "ovalo": __pista_ovalo(),
    "ocho": __pista_ocho(),
    "chicane": [
        (260.0, 260.0),
        (1340.0, 260.0),
        (1340.0, 940.0),
        (1010.0, 940.0),
        (1010.0, 640.0),
        (790.0, 640.0),
        (790.0, 940.0),
        (260.0, 940.0),
    ],
}

# (velocidad en px/s, giro en rad/s): girar avanzando.
# Radio de giro ~59px: calibrado para curvas suaves sin sobreoscilar; las
# esquinas de 90° cerradas (pista 'chicane') son el caso límite conocido.
PRESETS: dict[Command, tuple[float, float]] = {
    Command.UP: (115.0, 0.0),
    Command.LEFT: (95.0, -1.6),
    Command.RIGHT: (95.0, 1.6),
    Command.DOWN: (0.0, 0.0),
}


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
    "realista": Realismo(
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


@dataclass(frozen=True, slots=True)
class Signo:
    """Señal pintada sobre el suelo junto a la línea."""

    s: float  # posición a lo largo de la pista (fracción 0-1)
    tipo: Senal
    lado: int  # +1 o -1 (a qué lado de la línea)
    offset: float = 70.0
    radio: float = 42.0


SIGNOS_BASE: list[Signo] = [
    Signo(0.22, Senal.PARE, lado=1),
    Signo(0.30, Senal.SIGA, lado=1),
    Signo(0.60, Senal.PARE, lado=-1),
    Signo(0.68, Senal.SIGA, lado=-1),
]


@dataclass(slots=True)
class Carrito:
    """Pose del carrito en coordenadas del mundo (y hacia abajo)."""

    x: float
    y: float
    theta: float


def vertices_octagono(
    centro: tuple[float, float], radio: float
) -> NDArray[np.int32]:
    """Octágono regular (vértices desfasados para quedar 'plano' arriba)."""
    angulos = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False) + np.pi / 8.0
    xs = centro[0] + radio * np.cos(angulos)
    ys = centro[1] + radio * np.sin(angulos)
    return np.stack([xs, ys], axis=1).astype(np.int32)


class Pista:
    """Poli-línea cerrada muestreada desde la spline de control."""

    def __init__(
        self,
        puntos_control: list[tuple[float, float]] | None = None,
        muestras: int = 560,
        ancho_linea: int = 26,
    ) -> None:
        control = np.asarray(puntos_control or PUNTOS_CONTROL, dtype=np.float64)
        self.puntos: NDArray[np.float64] = self.__catmull_rom_cerrada(control, muestras)
        self.ancho_linea = ancho_linea

    @staticmethod
    def __catmull_rom_cerrada(
        control: NDArray[np.float64], muestras: int
    ) -> NDArray[np.float64]:
        """Spline cerrada Catmull-Rom (Hermite con tangentes cardinales)."""
        n = len(control)
        por_tramo = max(1, muestras // n)
        ts = np.linspace(0.0, 1.0, por_tramo, endpoint=False)
        t2, t3 = ts**2, ts**3
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + ts
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        trozos: list[NDArray[np.float64]] = []
        for i in range(n):
            p0, p1, p2, p3 = (
                control[(i - 1) % n],
                control[i],
                control[(i + 1) % n],
                control[(i + 2) % n],
            )
            m1 = 0.5 * (p2 - p0)
            m2 = 0.5 * (p3 - p1)
            q = (
                p1[None, :] * h00[:, None]
                + m1[None, :] * h10[:, None]
                + p2[None, :] * h01[:, None]
                + m2[None, :] * h11[:, None]
            )
            trozos.append(q)
        return np.concatenate(trozos)

    def indice(self, s: float) -> int:
        return int((s % 1.0) * len(self.puntos))

    def punto(self, s: float) -> tuple[float, float]:
        x, y = self.puntos[self.indice(s)]
        return (float(x), float(y))

    def tangente(self, s: float) -> tuple[float, float]:
        i = self.indice(s)
        anterior = self.puntos[i - 1]
        siguiente = self.puntos[(i + 1) % len(self.puntos)]
        dx, dy = siguiente - anterior
        norma = math.hypot(dx, dy)
        if norma == 0:
            return (1.0, 0.0)
        return (dx / norma, dy / norma)

    def distancia_minima(self, p: tuple[float, float]) -> float:
        return float(
            np.sqrt(
                ((self.puntos[:, 0] - p[0]) ** 2 + (self.puntos[:, 1] - p[1]) ** 2).min()
            )
        )

    def dibujar(self, lienzo: NDArray[np.uint8]) -> None:
        cv2.polylines(
            lienzo,
            [self.puntos.astype(np.int32)],
            isClosed=True,
            color=COLOR_LINEA,
            thickness=self.ancho_linea,
        )


class Mundo:
    """Pista + señales + carrito + métricas que espejan la rúbrica."""

    def __init__(
        self,
        pista: Pista | None = None,
        signos: list[Signo] | None = None,
        realismo: Realismo | None = None,
        umbral_descarrilamiento: float = 42.0,
    ) -> None:
        self.pista = pista or Pista()
        self.signos = signos if signos is not None else list(SIGNOS_BASE)
        self.realismo = realismo
        self.umbral_descarrilamiento = umbral_descarrilamiento

        x0, y0 = self.pista.punto(0.0)
        tx, ty = self.pista.tangente(0.0)
        self.carrito = Carrito(x0, y0, math.atan2(ty, tx))
        self.__idx_prev: int = 0
        self.lienzo = self.__renderizar()

        self.t: float = 0.0
        self.vueltas: int = 0
        self.descarrilamientos: int = 0
        self.__s_prev: float = self.progreso()
        self.__descarrilado: bool = False
        self.__direccion: float = 1.0

    @staticmethod
    def __normalizar(angulo: float) -> float:
        return (angulo + math.pi) % (2.0 * math.pi) - math.pi

    def __posicion_signo(self, signo: Signo) -> tuple[float, float]:
        px, py = self.pista.punto(signo.s)
        tx, ty = self.pista.tangente(signo.s)
        nx, ny = -ty, tx  # normal a la línea
        return (
            px + nx * (signo.lado * signo.offset),
            py + ny * (signo.lado * signo.offset),
        )

    def __renderizar(self) -> NDArray[np.uint8]:
        lienzo = np.full((LIENZO_ALTO, LIENZO_ANCHO, 3), COLOR_SUELO, dtype=np.uint8)
        rng = np.random.default_rng(self.realismo.semilla if self.realismo else 0)
        if self.realismo is None:
            self.pista.dibujar(lienzo)
        else:
            self.__dibujar_manchas(lienzo, rng)
            self.__dibujar_trazo_realista(lienzo, rng)
            self.__dibujar_reflejos(lienzo, rng)
        for signo in self.signos:
            centro = self.__posicion_signo(signo)
            cv2.fillPoly(
                lienzo,
                [vertices_octagono(centro, signo.radio)],
                color=COLOR_SENAL[signo.tipo],
            )
        return lienzo

    def __dibujar_trazo_realista(
        self, lienzo: NDArray[np.uint8], rng: np.random.Generator
    ) -> None:
        """Dibuja la línea por tramos: ancho variable, temblor y huecos."""
        r = self.realismo
        assert r is not None
        p = self.pista.puntos
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

        hueco = self.__generar_huecos(rng, paso, largo_total)
        partido = self.__tramos_aleatorios(
            rng, r.tramos_partidos, r.partido_largo_min, r.partido_largo_max, paso
        )
        if r.degradado_fraccion > 0.0:
            cantidad_degradada = max(
                1, int(round(r.degradado_fraccion * largo_total / 175.0))
            )
        else:
            cantidad_degradada = 0
        degradado = self.__tramos_aleatorios(rng, cantidad_degradada, 100.0, 250.0, paso)

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

    def __generar_huecos(
        self, rng: np.random.Generator, paso: float, largo_total: float
    ) -> NDArray[np.bool_]:
        """Huecos solo en zonas rectas, lejos de señales y de la meta."""
        r = self.realismo
        assert r is not None
        p = self.pista.puntos
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

        radio_idx = max(1, int(90.0 / paso))
        for signo in self.signos:
            centro = self.pista.indice(signo.s)
            for k in range(-radio_idx, radio_idx + 1):
                permitido[(centro + k) % n] = False
        for k in range(-40, 41):
            permitido[k % n] = False  # zona de meta

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
        self,
        rng: np.random.Generator,
        cantidad: int,
        largo_min: float,
        largo_max: float,
        paso: float,
    ) -> NDArray[np.bool_]:
        """Tramos aleatorios de la pista, lejos de señales y de la meta."""
        n = len(self.pista.puntos)
        mascara = np.zeros(n, dtype=np.bool_)
        if cantidad <= 0:
            return mascara
        excluidos = np.zeros(n, dtype=np.bool_)
        for signo in self.signos:
            centro = self.pista.indice(signo.s)
            radio = max(1, int(90.0 / paso))
            for k in range(-radio, radio + 1):
                excluidos[(centro + k) % n] = True
        for k in range(-40, 41):
            excluidos[k % n] = True  # zona de meta
        candidatos = np.where(~excluidos)[0]
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

    def __dibujar_reflejos(
        self, lienzo: NDArray[np.uint8], rng: np.random.Generator
    ) -> None:
        """Brillo especular blanco sobre la línea (reflejo de la luz).

        El reflejo no borra la geometría: la línea sigue ahí, pero la cámara
        la ve rota en pedazos. El modo hueco del cerebro debe cruzarla recto.
        """
        r = self.realismo
        assert r is not None
        if r.reflejos <= 0:
            return
        p = self.pista.puntos
        n = len(p)
        idx_signos = [self.pista.indice(sg.s) for sg in self.signos]
        colocados = 0
        intentos = 0
        while colocados < r.reflejos and intentos < r.reflejos * 20:
            intentos += 1
            i = int(rng.integers(0, n))
            if any(abs((i - i0 + n // 2) % n - n // 2) < 30 for i0 in idx_signos):
                continue
            tx, ty = self.pista.tangente(i / n)
            largo = float(rng.uniform(r.reflejo_largo_min, r.reflejo_largo_max))
            ancho_r = float(rng.uniform(6.0, r.reflejo_ancho_max))
            brillo = int(rng.uniform(215.0, 242.0))
            color = (brillo, brillo - 2, brillo - 4)  # blanco grisáceo, S baja
            cv2.ellipse(
                lienzo,
                (int(round(p[i, 0])), int(round(p[i, 1]))),
                (int(round(largo / 2.0)), int(round(ancho_r / 2.0))),
                math.degrees(math.atan2(ty, tx)),
                0.0,
                360.0,
                color,
                -1,
            )
            colocados += 1

    def __dibujar_manchas(
        self, lienzo: NDArray[np.uint8], rng: np.random.Generator
    ) -> None:
        """Manchas de suciedad oscuras y desaturadas junto a la línea."""
        r = self.realismo
        assert r is not None
        if r.manchas <= 0:
            return
        p = self.pista.puntos
        n = len(p)
        idx_signos = [self.pista.indice(sg.s) for sg in self.signos]
        colocadas = 0
        intentos = 0
        while colocadas < r.manchas and intentos < r.manchas * 20:
            intentos += 1
            i = int(rng.integers(0, n))
            if any(
                abs((i - i0 + n // 2) % n - n // 2) < 30 for i0 in idx_signos
            ):
                continue
            lado = 1 if rng.random() < 0.5 else -1
            lateral = lado * float(rng.uniform(r.mancha_lateral_min, 120.0))
            tx, ty = self.pista.tangente(i / n)
            cx = p[i, 0] - ty * lateral
            cy = p[i, 1] + tx * lateral
            eje_a = float(rng.uniform(r.mancha_radio_min, r.mancha_radio_max))
            eje_b = eje_a * float(rng.uniform(0.5, 0.9))
            gris = int(rng.uniform(60.0, 110.0))
            color = (gris, gris - int(rng.uniform(0, 8)), gris - int(rng.uniform(0, 8)))
            cv2.ellipse(
                lienzo,
                (int(round(cx)), int(round(cy))),
                (int(round(eje_a)), int(round(eje_b))),
                float(rng.uniform(0.0, 180.0)),
                0.0,
                360.0,
                color,
                -1,
            )
            colocadas += 1

    def progreso(self) -> float:
        """Fracción de pista (0-1) más cercana al carrito.

        Con histéresis de rama: en pistas que se acercan a sí mismas (el
        ocho) el índice global más cercano puede saltar de lóbulo; se
        restringe la búsqueda a una ventana del último índice visitado.
        """
        p = self.pista.puntos
        n = len(p)
        d2 = (p[:, 0] - self.carrito.x) ** 2 + (p[:, 1] - self.carrito.y) ** 2
        ventana = 60
        indices = (self.__idx_prev + np.arange(-ventana, ventana + 1)) % n
        mejor_local = int(indices[int(np.argmin(d2[indices]))])
        if d2[mejor_local] <= 200.0**2:
            self.__idx_prev = mejor_local
        else:
            self.__idx_prev = int(d2.argmin())
        return self.__idx_prev / n

    def paso(self, comando: Command, dt: float) -> None:
        """Integra el preset del comando durante dt y actualiza métricas."""
        velocidad, giro = PRESETS[comando]
        self.carrito.theta = self.__normalizar(self.carrito.theta + giro * dt)
        self.carrito.x += velocidad * math.cos(self.carrito.theta) * dt
        self.carrito.y += velocidad * math.sin(self.carrito.theta) * dt
        self.t += dt

        s = self.progreso()
        cerca_de_pista = self.pista.distancia_minima(
            (self.carrito.x, self.carrito.y)
        ) < 150.0
        # Solo cuenta vueltas recorridas hacia adelante (evita contar el
        # ping-pong de un robot que media-vuelta).
        delta = s - self.__s_prev
        if delta < -0.5:
            delta += 1.0
        elif delta > 0.5:
            delta -= 1.0
        if abs(delta) > 1e-9:
            self.__direccion = delta
        if self.__s_prev > 0.9 and s < 0.1 and cerca_de_pista and self.__direccion > 0:
            self.vueltas += 1
        self.__s_prev = s

        fuera = self.pista.distancia_minima(
            (self.carrito.x, self.carrito.y)
        ) > self.umbral_descarrilamiento
        if fuera and not self.__descarrilado:
            self.descarrilamientos += 1
        self.__descarrilado = fuera
