"""Filtro de señales: confirmación por mayoría, bloqueo y enmascarado.

Una detección suelta no basta para detener el robot (un frame con un reflejo
rojizo bastaría). Aquí vive la votación sobre una ventana corta, el bloqueo
que evita re-disparar con la MISMA señal mientras se pasa por delante, y el
tapado de la señal antes de buscar la línea.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from .types import DeteccionSenal, Senal


class FiltroSenales:
    """Convierte detecciones por frame en una señal confirmada y estable."""

    def __init__(
        self,
        confirmar_pare: int = 3,
        confirmar_siga: int = 2,
        ventana: int = 5,
    ) -> None:
        self.confirmar_pare = confirmar_pare
        self.confirmar_siga = confirmar_siga
        self.__historial: deque[Senal | None] = deque(maxlen=ventana)
        self.__bloqueo: Senal | None = None

    def reiniciar(self) -> None:
        self.__historial.clear()
        self.__bloqueo = None

    def observar(self, senal: Senal | None) -> None:
        self.__historial.append(senal)

    def bloquear(self, senal: Senal) -> None:
        """Ignora esa señal hasta que desaparezca del campo de visión."""
        self.__bloqueo = senal

    def confirmada(self, detenido: bool) -> Senal | None:
        """Señal con mayoría en la ventana, respetando el bloqueo activo."""
        pare = self.__historial.count(Senal.PARE)
        siga = self.__historial.count(Senal.SIGA)
        if detenido:
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

        if self.__bloqueo is not None:
            if confirmada is self.__bloqueo:
                return None
            if self.__historial.count(self.__bloqueo) == 0:
                self.__bloqueo = None
        return confirmada


def ocultar(
    frame: NDArray[np.uint8], deteccion: DeteccionSenal
) -> NDArray[np.uint8]:
    """Tapa la señal detectada con el color mediano del suelo.

    Evita que el octágono de color contamine el K-Means de la línea (el matiz
    del rojo envuelve 0/179 y distorsiona los clusters). Efecto secundario
    buscado: una señal puesta ENCIMA de la línea borra ese tramo, y el robot
    lo cruza en modo hueco, igual que si el marcador se hubiera cortado.
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
