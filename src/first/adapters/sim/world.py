"""Mundo simulado: pista + señales + carrito + métricas de la rúbrica.

La física es deliberadamente simple: cada comando discreto es un preset
(velocidad, giro) de un modelo tipo diferencial que gira mientras avanza.
La misma tabla de presets es la que se mapearía al Arduino real.

El mundo es mutable en caliente (mover señales, reposicionar el carrito,
cambiar el realismo) sin perder pose ni métricas: el panel del simulador lo
usa para editar la escena mientras la simulación corre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np
from numpy.typing import NDArray

from ...core.types import Command, Senal
from .lienzo import COLOR_SUELO, LIENZO_ALTO, LIENZO_ANCHO
from .pista import Pista
from .realismo import Realismo
from .trazo import renderizar_pista

COLOR_SENAL: dict[Senal, tuple[int, int, int]] = {
    Senal.PARE: (0, 0, 210),  # BGR
    Senal.SIGA: (0, 190, 0),
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

# Una señal puede quedar centrada sobre la línea (offset 0): el cerebro tapa
# su caja con el color del piso antes de buscar la línea, así que ese tramo
# desaparece y el robot lo cruza en modo hueco. Es una prueba válida y dura.
OFFSET_SIGNO_MIN: float = 0.0
OFFSET_SIGNO_MAX: float = 160.0


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

    def posicion_signo(self, signo: Signo) -> tuple[float, float]:
        """Centro del octágono: punto de la pista desplazado por su normal."""
        px, py = self.pista.punto(signo.s)
        nx, ny = self.pista.normal(signo.s)
        return (
            px + nx * (signo.lado * signo.offset),
            py + ny * (signo.lado * signo.offset),
        )

    def __renderizar(self) -> NDArray[np.uint8]:
        lienzo = np.full((LIENZO_ALTO, LIENZO_ANCHO, 3), COLOR_SUELO, dtype=np.uint8)
        if self.realismo is None:
            self.pista.dibujar(lienzo)
        else:
            renderizar_pista(
                lienzo,
                self.pista,
                self.realismo,
                [self.pista.indice(sg.s) for sg in self.signos],
                np.random.default_rng(self.realismo.semilla),
            )
        for signo in self.signos:
            cv2.fillPoly(
                lienzo,
                [vertices_octagono(self.posicion_signo(signo), signo.radio)],
                color=COLOR_SENAL[signo.tipo],
            )
        return lienzo

    # ---------- edición en caliente de la escena ----------

    def redibujar(self) -> None:
        """Vuelve a pintar el lienzo tras cambiar señales o realismo."""
        self.lienzo = self.__renderizar()

    def establecer_signos(self, signos: list[Signo]) -> None:
        self.signos = list(signos)
        self.redibujar()

    def mover_signo(self, indice: int, s: float, lado: int, offset: float) -> Signo:
        """Reubica una señal sobre la pista (offset acotado a lo dibujable)."""
        signo = replace(
            self.signos[indice],
            s=s % 1.0,
            lado=1 if lado >= 0 else -1,
            offset=min(OFFSET_SIGNO_MAX, max(OFFSET_SIGNO_MIN, offset)),
        )
        self.signos[indice] = signo
        self.redibujar()
        return signo

    def cambiar_realismo(self, realismo: Realismo | None) -> None:
        """Cambia las imperfecciones SIN tocar pose, métricas ni progreso.

        La geometría de la pista es la misma: solo cambia cómo está pintada,
        así que reiniciar la carrera aquí sería gratuito y molesto.
        """
        self.realismo = realismo
        self.redibujar()

    def reposicionar(self, x: float, y: float, theta: float) -> None:
        """Teletransporta el carrito y resincroniza el estado de progreso.

        Sin esta resincronización el salto se leería como una vuelta (o un
        descarrilamiento) espurio en el siguiente paso.
        """
        self.carrito.x = x
        self.carrito.y = y
        self.carrito.theta = self.__normalizar(theta)
        self.__idx_prev = self.pista.indice_cercano((x, y))
        self.__s_prev = self.progreso()
        self.__descarrilado = (
            self.pista.distancia_minima((x, y)) > self.umbral_descarrilamiento
        )

    def reposicionar_en_pista(self, s: float) -> None:
        """Pega el carrito a la línea en la fracción s, mirando a la tangente."""
        x, y = self.pista.punto(s)
        tx, ty = self.pista.tangente(s)
        self.reposicionar(x, y, math.atan2(ty, tx))

    # ---------- física y métricas ----------

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
        distancia = self.pista.distancia_minima((self.carrito.x, self.carrito.y))
        # Solo cuenta vueltas recorridas hacia adelante (evita contar el
        # ping-pong de un robot que media-vuelta).
        delta = s - self.__s_prev
        if delta < -0.5:
            delta += 1.0
        elif delta > 0.5:
            delta -= 1.0
        if abs(delta) > 1e-9:
            self.__direccion = delta
        if (
            self.__s_prev > 0.9
            and s < 0.1
            and distancia < 150.0
            and self.__direccion > 0
        ):
            self.vueltas += 1
        self.__s_prev = s

        fuera = distancia > self.umbral_descarrilamiento
        if fuera and not self.__descarrilado:
            self.descarrilamientos += 1
        self.__descarrilado = fuera
