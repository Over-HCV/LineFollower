"""Herramienta de calibración: detector de polígonos con trackbars.

Útil para afinar Canny, blur, área mínima y precisión poligonal contra la
webcam antes de fijar parámetros en el cerebro. Ejecutar con:
    uv run python -m first.calib
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def nada(x: int) -> None:
    """Función vacía requerida por los trackbars de OpenCV."""
    pass


def obtener_nombre_poligono(lados: int) -> str:
    """Retorna el nombre del polígono según la cantidad de lados."""
    nombres = {
        3: "Triangulo",
        4: "Cuadrilatero",
        5: "Pentagono",
        6: "Hexagono",
        7: "Heptagono",
        8: "Octagono",
    }
    return nombres.get(lados, f"Poligono ({lados}L)")


def ajustar_kernel_impar(valor: int) -> int:
    """OpenCV necesita que el kernel del Gaussian Blur sea impar."""
    if valor < 1:
        valor = 1
    if valor % 2 == 0:
        valor += 1
    return valor


def crear_interfaz(nombre_ventana: str) -> None:
    """Crea la ventana principal con tamaño redimensionable y los trackbars."""
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, 1080, 680)

    cv2.createTrackbar("Umbral Canny Bajo", nombre_ventana, 80, 255, nada)
    cv2.createTrackbar("Umbral Canny Alto", nombre_ventana, 180, 255, nada)
    cv2.createTrackbar("Gaussian Blur", nombre_ventana, 5, 31, nada)
    cv2.createTrackbar("Area Minima", nombre_ventana, 1000, 20000, nada)
    cv2.createTrackbar("Precision Poligono", nombre_ventana, 2, 20, nada)


def leer_controles(nombre_ventana: str) -> tuple[int, int, int, int, int]:
    """Lee los valores actuales de los trackbars."""
    canny_bajo = cv2.getTrackbarPos("Umbral Canny Bajo", nombre_ventana)
    canny_alto = cv2.getTrackbarPos("Umbral Canny Alto", nombre_ventana)
    blur = cv2.getTrackbarPos("Gaussian Blur", nombre_ventana)
    area_minima = cv2.getTrackbarPos("Area Minima", nombre_ventana)
    precision = cv2.getTrackbarPos("Precision Poligono", nombre_ventana)

    blur = ajustar_kernel_impar(blur)
    if precision < 1:
        precision = 1

    return canny_bajo, canny_alto, blur, area_minima, precision


def preprocesar_imagen(
    frame: NDArray[np.uint8], blur: int
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Convierte la imagen a escala de grises y aplica Gaussian Blur."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gris, (blur, blur), 0)
    return gris, gauss


def detectar_bordes(
    gauss: NDArray[np.uint8], canny_bajo: int, canny_alto: int
) -> NDArray[np.uint8]:
    """Aplica Canny para detectar bordes y dilata ligeramente."""
    bordes = cv2.Canny(gauss, canny_bajo, canny_alto)
    kernel = np.ones((3, 3), np.uint8)
    bordes = cv2.dilate(bordes, kernel, iterations=1)
    return bordes


def encontrar_contornos(
    bordes: NDArray[np.uint8],
) -> list[NDArray[np.int32]]:
    """Encuentra contornos externos en la imagen de bordes."""
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos


def clasificar_cuadrilatero(w: int, h: int) -> str:
    """Diferencia entre cuadrado y rectángulo usando relación ancho/alto."""
    relacion = w / float(h)
    if 0.90 <= relacion <= 1.10:
        return "Cuadrado"
    return "Rectangulo"


def dibujar_poligonos(
    frame: NDArray[np.uint8],
    contornos: list[NDArray[np.int32]],
    area_minima: int,
    precision: int,
) -> tuple[NDArray[np.uint8], dict[str, int]]:
    """Aproxima contornos a polígonos y los dibuja sobre la imagen."""
    salida = frame.copy()
    conteo_formas: dict[str, int] = {}

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < area_minima:
            continue

        perimetro = cv2.arcLength(contorno, True)
        epsilon = (precision / 100) * perimetro
        aproximacion = cv2.approxPolyDP(contorno, epsilon, True)
        lados = len(aproximacion)

        x, y, w, h = cv2.boundingRect(aproximacion)

        forma = obtener_nombre_poligono(lados)

        if lados == 4:
            forma = clasificar_cuadrilatero(w, h)

        conteo_formas[forma] = conteo_formas.get(forma, 0) + 1

        cv2.drawContours(salida, [aproximacion], 0, (0, 255, 0), 3)

        for punto in aproximacion:
            px, py = punto[0]
            cv2.circle(salida, (px, py), 5, (0, 0, 255), -1)

        cv2.rectangle(salida, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(
            salida,
            f"{forma} ({lados}L)",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return salida, conteo_formas


def crear_panel_info(
    ancho: int,
    alto: int,
    conteo_formas: dict[str, int],
    canny_b: int,
    canny_a: int,
    blur: int,
    area_min: int,
    prec: int,
) -> NDArray[np.uint8]:
    """Crea un panel de información estilizado para llenar la 6ª casilla."""
    info = np.zeros((alto, ancho, 3), dtype=np.uint8)
    info[:] = (35, 35, 35)

    cv2.putText(
        info, "PANEL DE INFORMACION", (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
    )
    cv2.line(info, (15, 40), (ancho - 15, 40), (100, 100, 100), 1)

    y = 65
    cv2.putText(info, f"Canny: {canny_b} - {canny_a}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 22
    cv2.putText(info, f"Gauss Blur: {blur}x{blur}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 22
    cv2.putText(info, f"Area Minima: {area_min}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 22
    cv2.putText(info, f"Precision: {prec}%", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    y += 30
    cv2.putText(info, "DETECTADOS:", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.line(info, (15, y + 8), (ancho - 15, y + 8), (100, 100, 100), 1)

    y += 30
    total = sum(conteo_formas.values())
    if total == 0:
        cv2.putText(info, "Ningun objeto detectado", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 255), 1)
    else:
        for forma, cantidad in conteo_formas.items():
            cv2.putText(info, f"- {forma}: {cantidad}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 24

    cv2.putText(info, "Presiona 'Q' para salir", (15, alto - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    return info


def crear_mosaico(
    frame: NDArray[np.uint8],
    gris: NDArray[np.uint8],
    gauss: NDArray[np.uint8],
    bordes: NDArray[np.uint8],
    salida: NDArray[np.uint8],
    conteo_formas: dict[str, int],
    canny_b: int,
    canny_a: int,
    blur: int,
    area_min: int,
    prec: int,
) -> NDArray[np.uint8]:
    """Organiza las 5 pantallas + el panel de info en un mosaico 3x2."""
    ancho = 360
    alto = 270

    frame_r = cv2.resize(frame, (ancho, alto))
    gris_r = cv2.resize(cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR), (ancho, alto))
    gauss_r = cv2.resize(cv2.cvtColor(gauss, cv2.COLOR_GRAY2BGR), (ancho, alto))
    bordes_r = cv2.resize(cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR), (ancho, alto))
    salida_r = cv2.resize(salida, (ancho, alto))
    info_r = crear_panel_info(ancho, alto, conteo_formas, canny_b, canny_a, blur, area_min, prec)

    def titulo(img: NDArray[np.uint8], text: str) -> NDArray[np.uint8]:
        cv2.rectangle(img, (0, 0), (ancho, 30), (30, 30, 30), -1)
        cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img

    titulo(frame_r, "1. Original")
    titulo(gris_r, "2. Escala de Grises")
    titulo(gauss_r, "3. Gaussian Blur")
    titulo(bordes_r, "4. Bordes Canny")
    titulo(salida_r, "5. Poligonos Detectados")

    fila1 = np.hstack((frame_r, gris_r, gauss_r))
    fila2 = np.hstack((bordes_r, salida_r, info_r))

    panel = np.vstack((fila1, fila2))
    return panel


def main() -> None:
    nombre_ventana = "Detector de Poligonos"

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la camara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    crear_interfaz(nombre_ventana)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: No se pudo leer el fotograma.")
            break

        canny_bajo, canny_alto, blur, area_minima, precision = leer_controles(nombre_ventana)

        gris, gauss = preprocesar_imagen(frame, blur)
        bordes = detectar_bordes(gauss, canny_bajo, canny_alto)
        contornos = encontrar_contornos(bordes)
        salida, conteo_formas = dibujar_poligonos(frame, contornos, area_minima, precision)

        mosaico = crear_mosaico(
            frame, gris, gauss, bordes, salida, conteo_formas,
            canny_bajo, canny_alto, blur, area_minima, precision,
        )

        cv2.imshow(nombre_ventana, mosaico)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
