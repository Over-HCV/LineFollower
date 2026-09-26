"""Punto de entrada del proyecto.

Ejemplos:
    first                         # simulador pygame (por defecto)
    first --source webcam         # cerebro contra la webcam local
    first --source video --video ensayo.mp4
    first --pare-seconds 5 --tolerancia 20 --modo-linea umbral

El cerebro (core/) es idéntico para simulador, webcam, video y el carrito
real: solo cambia el adaptador de entrada/salida elegido aquí.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .core.brain import Brain
from .core.vision.line import ParamsLinea


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="first",
        description="Reto de visión: seguidor de línea con señales PARE/SIGA.",
    )
    parser.add_argument(
        "--source",
        choices=["sim", "webcam", "video"],
        default="sim",
        help="fuente de frames (por defecto: simulador pygame)",
    )
    parser.add_argument(
        "--pista",
        "--lap",
        dest="pista",
        choices=["cacahuate", "ovalo", "ocho", "chicane"],
        default=None,
        help="diseño de la pista del simulador (por defecto: la última usada)",
    )
    parser.add_argument(
        "--realismo",
        choices=["perfecto", "medio", "alto", "extremo"],
        default=None,
        help="imperfecciones del trazo (por defecto: el último usado)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="ruta del video de ensayo (con --source video)",
    )
    parser.add_argument(
        "--indice-camara",
        type=int,
        default=0,
        help="índice de la webcam (con --source webcam)",
    )
    parser.add_argument(
        "--pare-segundos",
        type=float,
        default=3.0,
        help="tiempo de detención ante la señal PARE (lo define el docente)",
    )
    parser.add_argument(
        "--tolerancia",
        type=int,
        default=12,
        help="umbral de entrada al giro en píxeles",
    )
    parser.add_argument(
        "--modo-linea",
        choices=["kmeans", "umbral"],
        default="kmeans",
        help="segmentación de la línea: K-Means adaptativo o umbral fijo",
    )
    return parser


def construir_brain(args: argparse.Namespace) -> Brain:
    params_linea = ParamsLinea(modo=args.modo_linea)
    return Brain(
        segundos_pare=args.pare_segundos,
        tolerancia_entrada=args.tolerancia,
        params_linea=params_linea,
    )


def main(argv: list[str] | None = None) -> None:
    args = construir_parser().parse_args(argv)
    brain = construir_brain(args)

    if args.source == "sim":
        from .adapters.sim.pygame_ui import ejecutar_simulador

        ejecutar_simulador(brain, pista=args.pista, realismo=args.realismo)
    elif args.source == "webcam":
        from .adapters.bucle_cv2 import bucle_cv2
        from .adapters.webcam import WebcamSource

        bucle_cv2(WebcamSource(args.indice_camara), brain)
    else:
        if args.video is None:
            raise SystemExit("--source video requiere --video RUTA")
        from .adapters.bucle_cv2 import bucle_cv2
        from .adapters.video_file import VideoFileSource

        bucle_cv2(VideoFileSource(args.video), brain)


if __name__ == "__main__":
    main()
