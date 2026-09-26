"""Cerebro del robot: orquesta visión y control sin ninguna E/S.

Contrato: ``procesar(frame_bgr) -> (Command, Telemetry)``. El mismo objeto
funciona contra el simulador, la webcam, los videos de ensayo o el carrito
real con Arduino (a través de los adaptadores).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from .control import ControladorLinea
from .types import Command, EstadoRobot, LineInfo, Senal, Telemetry
from .vision.line import ParamsLinea, detectar_linea
from .vision.sign import DeteccionSenal, detectar_senal

Reloj = Callable[[], float]


class Brain:
    """Máquina de estados SIGUIENDO/PARE/RECUPERANDO/PERDIDO + pipeline."""

    def __init__(
        self,
        segundos_pare: float = 3.0,
        tolerancia_entrada: int = 12,
        tolerancia_salida: int = 6,
        ventana_votacion: int = 5,
        confirmar_pare: int = 3,
        confirmar_siga: int = 2,
        cooldown_pare: float = 4.0,
        area_minima_senal: float = 900.0,
        timeout_recuperacion: float = 3.0,
        params_linea: ParamsLinea | None = None,
        reloj: Reloj = time.monotonic,
        dibujar_debug: bool = True,
    ) -> None:
        self.segundos_pare = segundos_pare
        self.timeout_recuperacion = timeout_recuperacion
        self.area_minima_senal = area_minima_senal
        self.confirmar_pare = confirmar_pare
        self.confirmar_siga = confirmar_siga
        self.cooldown_pare = cooldown_pare
        self.params_linea = params_linea or ParamsLinea()
        self.dibujar_debug = dibujar_debug

        self.__reloj = reloj
        self.__control = ControladorLinea(
            umbral_entrada=tolerancia_entrada,
            umbral_salida=tolerancia_salida,
            ventana=ventana_votacion,
        )
        self.estado: EstadoRobot = EstadoRobot.SIGUIENDO
        self.__historial_senales: deque[Senal | None] = deque(maxlen=5)
        self.__signo_ultimo_error: int = -1  # hacia dónde barrer si se pierde
        self.__contador_recuperacion: int = 0
        self.__ausencias: int = 0  # frames seguidos sin línea (debounce)
        self.__ultimo_comando: Command = Command.UP
        self.__t_pare: float | None = None
        self.__t_perdida: float | None = None
        self.__bloqueo_senal: Senal | None = None
        self.__cooldown_hasta: float = 0.0

    def reiniciar(self) -> None:
        """Vuelve al estado inicial (por ejemplo, al reiniciar la carrera)."""
        self.estado = EstadoRobot.SIGUIENDO
        self.__control.reiniciar()
        self.__historial_senales.clear()
        self.__t_pare = None
        self.__t_perdida = None
        self.__bloqueo_senal = None

    def procesar(self, frame: NDArray[np.uint8] | None) -> tuple[Command, Telemetry]:
        """Procesa un frame BGR y retorna el comando único más la telemetría."""
        if frame is None or frame.size == 0:
            telemetria = Telemetry(
                comando=Command.DOWN,
                estado=self.estado,
                senal_confirmada=None,
                senal_cruda=None,
                linea=LineInfo(presente=False),
            )
            return Command.DOWN, telemetria

        deteccion = detectar_senal(frame, self.area_minima_senal)
        self.__historial_senales.append(deteccion.senal)
        comando, linea = self.__transicionar(frame, deteccion)
        telemetria = Telemetry(
            comando=comando,
            estado=self.estado,
            senal_confirmada=self.__senal_confirmada(),
            senal_cruda=deteccion.senal,
            linea=linea,
            debug=self.__dibujar(frame, linea, deteccion, comando),
        )
        return comando, telemetria

    def __ocultar_senal(
        self, frame: NDArray[np.uint8], deteccion: DeteccionSenal
    ) -> NDArray[np.uint8]:
        """Tapa la señal detectada con el color mediano del suelo.

        Evita que el octágono de color contamine el K-Means de la línea
        (el matiz del rojo envuelve 0/179 y distorsiona los clusters).
        """
        if deteccion.caja is None:
            return frame
        frame = frame.copy()
        x, y, w, h = deteccion.caja
        limite_y = min(y + h, frame.shape[0])
        limite_x = min(x + w, frame.shape[1])
        frame[y:limite_y, x:limite_x] = tuple(
            int(v) for v in np.median(frame.reshape(-1, 3), axis=0)
        )
        return frame

    def __senal_confirmada(self) -> Senal | None:
        """Señal con confirmación por mayoría; bloquea re-disparos de la misma."""
        pare = self.__historial_senales.count(Senal.PARE)
        siga = self.__historial_senales.count(Senal.SIGA)
        if self.estado is EstadoRobot.PARE:
            # Detenido: solo SIGA importa (reanudar antes del temporizador).
            confirmada: Senal | None = (
                Senal.SIGA if siga >= self.confirmar_siga else None
            )
        elif pare >= self.confirmar_pare:
            confirmada = Senal.PARE
        elif siga >= self.confirmar_siga:
            confirmada = Senal.SIGA
        else:
            confirmada = None

        if self.__bloqueo_senal is not None:
            if confirmada is self.__bloqueo_senal:
                return None
            if self.__historial_senales.count(self.__bloqueo_senal) == 0:
                self.__bloqueo_senal = None
        return confirmada

    def __transicionar(
        self, frame: NDArray[np.uint8], deteccion: DeteccionSenal
    ) -> tuple[Command, LineInfo]:
        ahora = self.__reloj()
        confirmada = self.__senal_confirmada()
        linea = detectar_linea(
            self.__ocultar_senal(frame, deteccion), self.params_linea
        )

        # La señal PARE solo se obedece con línea visible: un robot perdido
        # no debe detenerse por una señal que ve desde lejos de la pista.
        # El cooldown evita re-disparar por la misma señal al pasarla.
        if (
            confirmada is Senal.PARE
            and self.estado is not EstadoRobot.PARE
            and linea.presente
            and ahora >= self.__cooldown_hasta
        ):
            self.estado = EstadoRobot.PARE
            self.__t_pare = ahora
            self.__control.reiniciar()
            self.__bloqueo_senal = Senal.PARE

        if self.estado is EstadoRobot.PARE:
            assert self.__t_pare is not None
            if (
                confirmada is not Senal.SIGA
                and ahora - self.__t_pare < self.segundos_pare
            ):
                return Command.DOWN, LineInfo(presente=False)
            self.estado = EstadoRobot.SIGUIENDO
            self.__t_pare = None
            self.__cooldown_hasta = ahora + self.cooldown_pare

        if linea.presente:
            error = linea.objetivo_px - frame.shape[1] // 2
            if error != 0:
                self.__signo_ultimo_error = 1 if error > 0 else -1

            if self.estado in (EstadoRobot.RECUPERANDO, EstadoRobot.PERDIDO):
                # Recaptura: solo se devuelve el control normal cuando la
                # línea quedó aproximadamente centrada en el campo de visión.
                if abs(error) < 120:
                    self.estado = EstadoRobot.SIGUIENDO
                    self.__control.reiniciar()
                else:
                    return (
                        Command.RIGHT if error > 0 else Command.LEFT
                    ), linea

            if self.estado is EstadoRobot.SIGUIENDO:
                self.__ausencias = 0
                self.__t_perdida = None
                comando = self.__control.decidir(error)
                self.__ultimo_comando = comando
                return comando, linea

        if self.estado is EstadoRobot.SIGUIENDO:
            self.__ausencias += 1
            if self.__ausencias >= 3:
                self.estado = EstadoRobot.RECUPERANDO
                self.__t_perdida = ahora
                self.__control.reiniciar()
            else:
                # Tolerancia al parpadeo: mantener el último comando válido.
                return self.__ultimo_comando, linea
        if self.estado is EstadoRobot.RECUPERANDO:
            assert self.__t_perdida is not None
            if ahora - self.__t_perdida <= self.timeout_recuperacion:
                # Giro continuo hacia el último lado con línea: el radio de
                # giro describe una espiral que barre y recaptura la pista.
                return (
                    Command.RIGHT if self.__signo_ultimo_error > 0 else Command.LEFT
                ), linea
            self.estado = EstadoRobot.PERDIDO
        return Command.DOWN, linea

    def __dibujar(
        self,
        frame: NDArray[np.uint8],
        linea: LineInfo,
        deteccion: DeteccionSenal,
        comando: Command,
    ) -> NDArray[np.uint8]:
        """Superpone ROIs, centroide, objetivo y estado para depuración."""
        lienzo = frame.copy()
        alto, ancho = lienzo.shape[:2]
        y_cerca = int(alto * self.params_linea.roi_cerca[0])
        y_lejos = int(alto * self.params_linea.roi_lejos[0])
        cv2.line(lienzo, (0, y_cerca), (ancho, y_cerca), (255, 255, 0), 1)
        cv2.line(lienzo, (0, y_lejos), (ancho, y_lejos), (255, 0, 255), 1)
        cv2.line(lienzo, (ancho // 2, y_lejos), (ancho // 2, alto), (0, 255, 255), 1)
        if linea.presente:
            cv2.circle(lienzo, (linea.cx, y_cerca + linea.cy), 6, (0, 255, 0), 2)
            cv2.line(
                lienzo,
                (linea.objetivo_px, y_lejos),
                (linea.objetivo_px, alto),
                (0, 0, 255),
                2,
            )
        if deteccion.caja is not None:
            x, y, w, h = deteccion.caja
            cv2.rectangle(lienzo, (x, y), (x + w, y + h), (0, 255, 255), 2)
            nombre = deteccion.senal.value.upper() if deteccion.senal else ""
            cv2.putText(
                lienzo,
                nombre,
                (x, max(y - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
        cv2.putText(
            lienzo,
            f"{comando.value.upper()} | {self.estado.value}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return lienzo
