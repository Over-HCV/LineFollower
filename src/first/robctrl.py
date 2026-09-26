"""
Cerebro de un robot seguidor de linea.

Recibe un frame BGR (OpenCV) y devuelve una sola senal:
"avance", "detener", "izquierda" o "derecha".

Tecnicas del curso (notebooks + taller.py):
- HSV: cv2.cvtColor(..., cv2.COLOR_BGR2HSV)
- Umbral y mascara de color
- Morfologia: opening + dilate (como en el taller)
- Contornos: cv2.findContours, cv2.contourArea, cv2.arcLength
- Forma: cv2.approxPolyDP (octogono = 8 lados, ver obtener_nombre_poligono)
- Linea: gris, GaussianBlur, threshold, cv2.moments (centroide)
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

# En OpenCV el matiz H va de 0 a 179 (notebook de espacios de color).
# El rojo queda en los dos extremos del circulo de color.
ROJO_BAJO_1 = np.array([0, 80, 60], dtype=np.uint8)
ROJO_ALTO_1 = np.array([12, 255, 255], dtype=np.uint8)
ROJO_BAJO_2 = np.array([165, 80, 60], dtype=np.uint8)
ROJO_ALTO_2 = np.array([179, 255, 255], dtype=np.uint8)

# Verde ≈ 60 en la escala 0-179 del notebook.
VERDE_BAJO = np.array([35, 60, 60], dtype=np.uint8)
VERDE_ALTO = np.array([90, 255, 255], dtype=np.uint8)

# Negro de la linea sobre fondo claro.
UMBRAL_LINEA = 70


def ajustar_kernel_impar(valor: int) -> int:
    """OpenCV pide kernel impar para el Gaussian Blur (igual que taller.py)."""
    if valor < 1:
        valor = 1
    if valor % 2 == 0:
        valor += 1
    return valor


def _limpiar_mascara(mascara: np.ndarray) -> np.ndarray:
    """Quita motas (apertura) y cierra huecos pequenos (dilatacion)."""
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.dilate(mascara, kernel, iterations=1)
    return mascara


def _mascara_color(hsv: np.ndarray, color: str) -> np.ndarray:
    """Segmenta rojo o verde. El rojo usa dos rangos porque el matiz da la vuelta."""
    if color == "rojo":
        m1 = cv2.inRange(hsv, ROJO_BAJO_1, ROJO_ALTO_1)
        m2 = cv2.inRange(hsv, ROJO_BAJO_2, ROJO_ALTO_2)
        mascara = cv2.bitwise_or(m1, m2)
    else:
        mascara = cv2.inRange(hsv, VERDE_BAJO, VERDE_ALTO)
    return _limpiar_mascara(mascara)


def __es_octagono(contorno, precision: float = 4.0) -> bool:
    """
    Misma aproximacion que taller.py:
    epsilon = (precision / 100) * perimetro
    Un octagono real suele salir en 7, 8 o 9 vertices.
    """
    perimetro = cv2.arcLength(contorno, True)
    if perimetro <= 0:
        return False
    epsilon = (precision / 100.0) * perimetro
    aproximacion = cv2.approxPolyDP(contorno, epsilon, True)
    lados = len(aproximacion)
    return 7 <= lados <= 9


def detectar_senales(frame: np.ndarray, area_minima: int = 800) -> Optional[str]:
    """
    Prioridad alta: busca un octagono rojo (PARE) o verde (SIGA).

    Devuelve "pare", "siga" o None.
    Si hay los dos, gana PARE (seguridad).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for nombre, color in (("pare", "rojo"), ("siga", "verde")):
        mascara = _mascara_color(hsv, color)
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contorno in contornos:
            if cv2.contourArea(contorno) < area_minima:
                continue
            if __es_octagono(contorno):
                return nombre
    return None


def procesar_linea(
    frame: np.ndarray, area_minima: int = 150
) -> Optional[Tuple[int, int]]:
    """
    Recorta el tercio inferior (ROI) y devuelve (cx, ancho_roi)
    del centroide de la linea negra. None si no hay linea.
    """
    alto, ancho = frame.shape[:2]
    roi = frame[2 * alto // 3 :, :]

    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = ajustar_kernel_impar(5)
    gauss = cv2.GaussianBlur(gris, (blur, blur), 0)

    # THRESH_BINARY_INV: lo negro (linea) queda blanco para findContours.
    _, mascara = cv2.threshold(gauss, UMBRAL_LINEA, 255, cv2.THRESH_BINARY_INV)
    mascara = _limpiar_mascara(mascara)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    linea = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(linea) < area_minima:
        return None

    momentos = cv2.moments(linea)
    if not momentos["m00"]:
        return None

    cx = int(momentos["m10"] / momentos["m00"])
    return cx, ancho


def calcular_control(
    centroide_x: Optional[int],
    ancho: int,
    tolerancia: int = 30,
) -> str:
    """
    Compara el centroide de la linea con el centro de la imagen.
    error = cx - centro. Negativo: la linea esta a la izquierda.
    """
    if not centroide_x:
        return "detener"

    centro = ancho // 2
    error = centroide_x - centro

    if abs(error) <= tolerancia:
        return "avance"
    if error < 0:
        return "izquierda"
    return "derecha"


class ControladorRobot:
    """
    Maquina de estados:
    - "movimiento": sigue la linea.
    - "detenido": ignora la linea hasta ver SIGA o hasta que pase el temporizador.
    """

    def __init__(
        self,
        tolerancia: int = 30,
        segundos_pare: float = 3.0,
        area_minima_senal: int = 800,
    ) -> None:
        self.tolerancia = tolerancia
        self.segundos_pare = segundos_pare
        self.area_minima_senal = area_minima_senal
        self.estado = "movimiento"
        self._inicio_pare: Optional[float] = None

    def reiniciar(self) -> None:
        self.estado = "movimiento"
        self._inicio_pare = None

    def procesar(self, frame: np.ndarray) -> str:
        """Un frame BGR entra; sale una de las 4 senales."""
        if not frame.any() or frame.size == 0:
            return "detener"

        senal = detectar_senales(frame, self.area_minima_senal)

        if senal == "pare":
            self.estado = "detenido"
            self._inicio_pare = time.time()
            return "detener"

        if senal == "siga" and self.estado == "detenido":
            self.estado = "movimiento"
            self._inicio_pare = None

        if self.estado == "detenido":
            if not self._inicio_pare:
                self._inicio_pare = time.time()
            if time.time() - self._inicio_pare < self.segundos_pare:
                return "detener"
            # Se acabo el tiempo de PARE y no aparecio SIGA: vuelve a la linea.
            self.estado = "movimiento"
            self._inicio_pare = None

        resultado = procesar_linea(frame)
        if not resultado:
            return calcular_control(None, frame.shape[1], self.tolerancia)

        cx, ancho = resultado
        return calcular_control(cx, ancho, self.tolerancia)


def main() -> None:
    """Prueba con la webcam. Muestra la senal en pantalla. q para salir."""
    import sys

    indice = 0

    camara = cv2.VideoCapture(
        indice,
        cv2.CAP_DSHOW if sys.platform.startswith("win") else None,
    )

    if not camara.isOpened():
        print("No se pudo abrir la camara.")
        return

    controlador = ControladorRobot(segundos_pare=3.0)
    print("q para salir. PARE detiene 3 s o hasta un SIGA.")

    loop_active = True
    while loop_active:
        ok, frame = camara.read()
        if not ok:
            break

        orden = controlador.procesar(frame)
        alto = frame.shape[0]
        cv2.line(
            frame, (0, 2 * alto // 3), (frame.shape[1], 2 * alto // 3), (255, 255, 0), 2
        )
        cv2.putText(
            frame,
            f"{orden} | estado: {controlador.estado}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Control del robot", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            loop_active = False

    camara.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
