"""Cerebro del robot: orquesta visión y control sin ninguna E/S.

Contrato: ``procesar(frame_bgr) -> (Command, Telemetry)``. El mismo objeto
funciona contra el simulador, la webcam, los videos de ensayo o el carrito
real con Arduino (a través de los adaptadores).
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .control import ControladorLinea
from .types import Command, EstadoRobot, LineInfo, Senal, Telemetry
from .senales import FiltroSenales, ocultar as ocultar_senal
from .vision.debug import dibujar as superponer_debug
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
        umbral_hueco: float = 40.0,  # error máximo para asumir hueco (px)
        tolerancia_hueco: float = 0.45,  # dead-reckoning recto hasta (s)
        cobertura_recaptura: float = 0.6,  # candidato debe cruzar la franja
        suavizado_centro: float = 0.35,  # EMA de la predicción del centro
        area_minima_senal: float = 900.0,
        timeout_recuperacion: float = 3.0,
        params_linea: ParamsLinea | None = None,
        reloj: Reloj = time.monotonic,
        dibujar_debug: bool = True,
    ) -> None:
        self.segundos_pare = segundos_pare
        self.tolerancia_entrada = tolerancia_entrada
        self.tolerancia_salida = tolerancia_salida
        self.timeout_recuperacion = timeout_recuperacion
        self.area_minima_senal = area_minima_senal
        self.confirmar_pare = confirmar_pare
        self.confirmar_siga = confirmar_siga
        self.cooldown_pare = cooldown_pare
        self.umbral_hueco = umbral_hueco
        self.tolerancia_hueco = tolerancia_hueco
        self.cobertura_recaptura = cobertura_recaptura
        self.suavizado_centro = suavizado_centro
        self.params_linea = params_linea or ParamsLinea()
        self.dibujar_debug = dibujar_debug

        self.__reloj = reloj
        self.__control = ControladorLinea(
            umbral_entrada=tolerancia_entrada,
            umbral_salida=tolerancia_salida,
            ventana=ventana_votacion,
        )
        self.__ventana_votacion = ventana_votacion
        self.estado: EstadoRobot = EstadoRobot.SIGUIENDO
        self.__senales = FiltroSenales(confirmar_pare, confirmar_siga, ventana=5)
        self.__signo_ultimo_error: int = -1  # hacia dónde barrer si se pierde
        self.__ausencias: int = 0  # frames seguidos sin línea (debounce)
        self.__ultimo_comando: Command = Command.UP
        self.__ultimo_error: float = 0.0  # error del último frame con línea
        self.__centro_esperado: float | None = None  # dónde se espera la línea
        self.__t_pare: float | None = None
        self.__t_perdida: float | None = None
        self.__cooldown_hasta: float = 0.0

    def reiniciar(self) -> None:
        """Vuelve al estado inicial (por ejemplo, al reiniciar la carrera)."""
        self.estado = EstadoRobot.SIGUIENDO
        self.__control.reiniciar()
        self.__senales.reiniciar()
        self.__t_pare = None
        self.__t_perdida = None
        self.__cooldown_hasta = 0.0
        self.__ausencias = 0
        self.__ultimo_comando = Command.UP
        self.__ultimo_error = 0.0
        self.__centro_esperado = None

    def reconfigurar(
        self,
        tolerancia_entrada: int | None = None,
        tolerancia_salida: int | None = None,
        segundos_pare: float | None = None,
        modo_linea: str | None = None,
        tolerancia_hueco: float | None = None,
        margen_lateral: int | None = None,
        sigma_atencion: float | None = None,
    ) -> None:
        """Ajusta parámetros en caliente SIN reiniciar la máquina de estados.

        La ventana de votación se reconstruye limpia: al cambiar umbrales lo
        sensato es volver a votar desde cero.
        """
        if tolerancia_entrada is not None:
            self.tolerancia_entrada = max(3, tolerancia_entrada)
        if tolerancia_salida is not None:
            self.tolerancia_salida = max(2, tolerancia_salida)
        if tolerancia_entrada is not None or tolerancia_salida is not None:
            self.tolerancia_salida = min(
                self.tolerancia_salida, self.tolerancia_entrada - 2
            )
            self.__control = ControladorLinea(
                umbral_entrada=self.tolerancia_entrada,
                umbral_salida=self.tolerancia_salida,
                ventana=self.__ventana_votacion,
            )
        if segundos_pare is not None:
            self.segundos_pare = max(0.0, segundos_pare)
        cambios: dict[str, Any] = {}
        if modo_linea is not None:
            cambios["modo"] = modo_linea
        if margen_lateral is not None:
            cambios["margen_lateral"] = max(0, margen_lateral)
        if sigma_atencion is not None:
            cambios["sigma_atencion"] = max(0.05, sigma_atencion)
        if cambios:
            self.params_linea = replace(self.params_linea, **cambios)
        if tolerancia_hueco is not None:
            self.tolerancia_hueco = max(0.0, tolerancia_hueco)

    def fijar_reloj(self, reloj: Reloj) -> None:
        """Cambia el reloj en vivo (para sincronizar con tiempo simulado)."""
        self.__reloj = reloj

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
        self.__senales.observar(deteccion.senal)
        comando, linea = self.__transicionar(frame, deteccion)
        telemetria = Telemetry(
            comando=comando,
            estado=self.estado,
            senal_confirmada=self.__senales.confirmada(
                self.estado is EstadoRobot.PARE
            ),
            senal_cruda=deteccion.senal,
            linea=linea,
            debug=(
                superponer_debug(
                    frame,
                    linea,
                    deteccion,
                    comando,
                    self.estado.value,
                    self.params_linea,
                    self.__centro_esperado,
                )
                if self.dibujar_debug
                else None
            ),
        )
        return comando, telemetria

    def __actualizar_centro(self, linea: LineInfo, ancho: int) -> None:
        """Sigue con un EMA dónde se espera la línea el próximo frame.

        Es el ancla de la atención: en curva la línea correcta no está en el
        centro del frame. Sin línea la predicción decae hacia el centro, para
        no quedarse enganchada a una posición vieja mientras se recupera.
        """
        if linea.presente:
            objetivo = float(linea.objetivo_px)
            if self.__centro_esperado is None:
                self.__centro_esperado = objetivo
            else:
                self.__centro_esperado += self.suavizado_centro * (
                    objetivo - self.__centro_esperado
                )
        elif self.__centro_esperado is not None:
            self.__centro_esperado += 0.2 * (ancho / 2.0 - self.__centro_esperado)

    def __transicionar(
        self, frame: NDArray[np.uint8], deteccion: DeteccionSenal
    ) -> tuple[Command, LineInfo]:
        ahora = self.__reloj()
        confirmada = self.__senales.confirmada(self.estado is EstadoRobot.PARE)
        linea = detectar_linea(
            ocultar_senal(frame, deteccion),
            self.params_linea,
            self.__centro_esperado,
        )
        self.__actualizar_centro(linea, frame.shape[1])

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
            self.__senales.bloquear(Senal.PARE)

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
            self.__ultimo_error = float(error)

            if self.estado in (
                EstadoRobot.HUECO,
                EstadoRobot.RECUPERANDO,
                EstadoRobot.PERDIDO,
            ):
                # Recaptura: el candidato debe ser una línea de verdad: cruza
                # la franja (cobertura alta) o es un segmento ancho típico de
                # esquina. Una mancha es compacta en ambas dimensiones.
                es_linea = (
                    linea.cobertura >= self.cobertura_recaptura
                    or linea.ancho_segmento >= 55
                )
                if es_linea:
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
                # Diagnóstico de la pérdida: si venía bien centrado es un
                # hueco del trazo (seguir recto); si venía desviado, girar.
                if abs(self.__ultimo_error) < self.umbral_hueco:
                    self.estado = EstadoRobot.HUECO
                else:
                    self.estado = EstadoRobot.RECUPERANDO
                self.__t_perdida = ahora
                self.__control.reiniciar()
            else:
                # Tolerancia al parpadeo: mantener el último comando válido.
                return self.__ultimo_comando, linea
        if self.estado is EstadoRobot.HUECO:
            assert self.__t_perdida is not None
            if ahora - self.__t_perdida <= self.tolerancia_hueco:
                # Dead-reckoning: el trazo es discontinuo y se cruza RECTO.
                # Congelar un giro aquí haría girar en ciego y descarrilar.
                return Command.UP, linea
            self.estado = EstadoRobot.RECUPERANDO
            self.__t_perdida = ahora
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
