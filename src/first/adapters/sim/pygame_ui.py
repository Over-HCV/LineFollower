"""Frontend pygame: vista del mundo, cámara del carrito, telemetría y métricas.

Métricas en vivo estilo rúbrica: tiempo, vueltas, descarrilamientos,
PAREs cumplidos e intervenciones humanas (las flechas cuentan como
intervención, como en la competencia).

La escena se edita mientras corre: la barra de abajo (``controles.py``)
ajusta el cerebro y el mundo, y con el mouse se arrastran el carrito y las
señales sobre la pista; la disposición resultante se guarda en disco para la
próxima sesión.
"""

from __future__ import annotations

from pathlib import Path

import pygame

from ...core.brain import Brain
from ...core.types import Command, EstadoRobot
from . import config, hud
from .arrastre import Arrastre
from .camera import CamaraSintetica
from .controles import Controles
from .pista import PISTAS, Pista
from .realismo import REALISMOS
from .world import Mundo

# Flechas U+2190-2193: las de la zona emoji (⬆ U+2B06, 🢂 U+1F882) no están en
# Menlo/Monaco/Consolas y pygame las pinta como caja vacía. Estas sí existen en
# la fuente monoespaciada, así que se ven en cualquier máquina.
FLECHAS_COMANDOS: dict[Command, str] = {
    Command.UP: "↑",
    Command.DOWN: "↓",
    Command.RIGHT: "→",
    Command.LEFT: "←",
}

TECLAS_MANUAL: dict[int, Command] = {
    pygame.K_UP: Command.UP,
    pygame.K_RIGHT: Command.RIGHT,
    pygame.K_DOWN: Command.DOWN,
    pygame.K_LEFT: Command.LEFT,
}

TECLAS_PISTA: dict[int, int] = {
    pygame.K_1: 0,
    pygame.K_2: 1,
    pygame.K_3: 2,
    pygame.K_4: 3,
}

AYUDA: tuple[str, ...] = (
    "A: auto/manual   R: reiniciar   flechas: manual (cuenta intervencion)   Q: salir",
    "M: modo   +/-: tolerancia   1-4: pista   mouse: arrastra el carrito y las senales",
)

DT_MAXIMO: float = 0.1  # un frame lento no debe teletransportar al carrito


def __resolver(pedido: str | None, guardado: object, opciones: dict) -> str:
    """Nombre efectivo: el CLI explícito manda; si no, lo guardado; si no, el 1º."""
    if pedido is not None and pedido in opciones:
        return pedido
    if isinstance(guardado, str) and guardado in opciones:
        return guardado
    return next(iter(opciones))


def ejecutar_simulador(
    brain: Brain,
    mundo: Mundo | None = None,
    camara: CamaraSintetica | None = None,
    pista: str | None = None,
    realismo: str | None = None,
    fps: int = 60,
    ruta_config: Path = config.RUTA_PREDETERMINADA,
) -> dict[str, float | int]:
    """Corre el simulador con ventana pygame. Retorna métricas al cerrar."""
    pygame.init()

    fuente = pygame.font.SysFont("menlo,consolas,monaco,monospace", 16)
    fuente_chica = pygame.font.SysFont("menlo,consolas,monaco,monospace", 13)

    guardado = config.cargar(ruta_config) or {}
    pista_actual = __resolver(pista, guardado.get("pista"), PISTAS)
    realismo_actual = __resolver(realismo, guardado.get("realismo"), REALISMOS)

    propio = mundo is None
    mundo = mundo or Mundo(
        pista=Pista(PISTAS[pista_actual]), realismo=REALISMOS[realismo_actual]
    )
    # Las posiciones guardadas solo valen sobre la pista en que se guardaron.
    if propio and guardado.get("pista") == pista_actual:
        config.aplicar(mundo, guardado)
    camara = camara or CamaraSintetica()
    brain.reiniciar()

    cam_w, cam_h = camara.tamaño_frame

    automatico: bool = True
    intervenciones: int = 0
    pares_cumplidos: int = 0
    previo: EstadoRobot = EstadoRobot.SIGUIENDO
    escala_tiempo: float = 1.0
    tiempo_sim: float = 0.0
    arrastre = Arrastre()
    cursor: tuple[int, int] = (0, 0)

    # El cerebro mide PARE y huecos en tiempo simulado: así el slider de
    # velocidad cambia solo la rapidez con que se observa, no la conducta.
    brain.fijar_reloj(lambda: tiempo_sim)

    def refrescar_lienzo() -> None:
        nonlocal mundo_superficie
        mundo_superficie = pygame.transform.smoothscale(
            hud.a_superficie(mundo.lienzo), hud.PANEL_MUNDO
        )

    def guardar() -> None:
        config.guardar(ruta_config, mundo, pista_actual, realismo_actual)

    def cambiar_pista(nombre: str) -> None:
        # Otra geometría: la carrera empieza de cero.
        nonlocal pista_actual, mundo, intervenciones, pares_cumplidos, previo
        pista_actual = nombre
        mundo = Mundo(
            pista=Pista(PISTAS[pista_actual]), realismo=REALISMOS[realismo_actual]
        )
        brain.reiniciar()
        refrescar_lienzo()
        intervenciones, pares_cumplidos, previo = 0, 0, EstadoRobot.SIGUIENDO
        guardar()

    def cambiar_realismo(nombre: str) -> None:
        # Misma geometría: se conservan pose, métricas y estado del cerebro.
        nonlocal realismo_actual
        realismo_actual = nombre
        mundo.cambiar_realismo(REALISMOS[nombre])
        refrescar_lienzo()
        guardar()

    def fijar_velocidad(valor: float) -> None:
        nonlocal escala_tiempo
        escala_tiempo = valor

    controles = Controles(
        brain,
        pista_actual,
        realismo_actual,
        cambiar_pista,
        cambiar_realismo,
        fijar_velocidad,
    )
    disposicion = hud.calcular(
        fuente,
        fuente_chica,
        (cam_w, cam_h),
        AYUDA,
        controles.filas_sliders,
        controles.ancho_minimo(fuente, hud.MARGEN),
    )
    controles.colocar(disposicion.filas, hud.MARGEN, fuente)

    pantalla = pygame.display.set_mode(disposicion.ventana)
    pygame.display.set_caption("Reto de visión: seguidor de línea (simulador)")
    reloj = pygame.time.Clock()
    mundo_superficie = pygame.transform.smoothscale(
        hud.a_superficie(mundo.lienzo), hud.PANEL_MUNDO
    )

    corriendo: bool = True
    while corriendo:
        dt = min(reloj.tick(fps) / 1000.0 * escala_tiempo, DT_MAXIMO)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key in (pygame.K_q, pygame.K_ESCAPE):
                    corriendo = False
                elif evento.key == pygame.K_a:
                    automatico = not automatico
                elif evento.key == pygame.K_m:
                    controles.modo.ciclo()
                elif evento.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    controles.tolerancia.ajustar(1.0)
                elif evento.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    controles.tolerancia.ajustar(-1.0)
                elif evento.key in TECLAS_PISTA:
                    controles.pista.seleccionar(TECLAS_PISTA[evento.key])
                elif evento.key == pygame.K_r:
                    cambiar_pista(pista_actual)  # recrea con la misma config
            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                cursor = evento.pos
                if disposicion.rect_mundo.collidepoint(evento.pos):
                    arrastre.tomar(mundo, disposicion.a_mundo(evento.pos))
                else:
                    controles.clic(evento.pos)
            elif evento.type == pygame.MOUSEBUTTONUP:
                if arrastre.activo and arrastre.soltar(
                    mundo, disposicion.a_mundo(evento.pos)
                ):
                    refrescar_lienzo()
                    guardar()
                controles.soltar()
            elif evento.type == pygame.MOUSEMOTION:
                cursor = evento.pos
                controles.mover(evento.pos)

        teclas = pygame.key.get_pressed()
        manual = next((c for k, c in TECLAS_MANUAL.items() if teclas[k]), None)
        if manual is not None and automatico:
            automatico = False
            intervenciones += 1

        frame = camara.render(mundo, mundo.carrito)
        comando_auto, telemetria = brain.procesar(frame)
        comando = comando_auto if automatico else (manual or Command.DOWN)

        if previo is EstadoRobot.PARE and telemetria.estado is not EstadoRobot.PARE:
            pares_cumplidos += 1
        previo = telemetria.estado

        if not arrastre.activo:  # editar la escena congela la física
            tiempo_sim += dt
            mundo.paso(comando, dt)

        pantalla.fill(hud.COLOR_FONDO)
        pantalla.blit(
            hud.panel_mundo(
                mundo_superficie,
                disposicion,
                (mundo.carrito.x, mundo.carrito.y, mundo.carrito.theta),
                telemetria.estado.value,
                cursor if arrastre.activo else None,
            ),
            disposicion.pos_mundo,
        )
        camara_vista = telemetria.debug if telemetria.debug is not None else frame
        pantalla.blit(hud.a_superficie(camara_vista), disposicion.pos_camara)
        pantalla.blit(
            hud.a_superficie(
                hud.mascara_a_frame(telemetria.linea.mascara, (cam_w, cam_h))
            ),
            disposicion.pos_mascara,
        )

        senal = (
            telemetria.senal_cruda.value if telemetria.senal_cruda is not None else "-"
        )
        hud.dibujar_lineas(
            pantalla,
            fuente,
            disposicion,
            [
                (
                    f"modo: {'AUTO' if automatico else 'MANUAL'}   "
                    f"comando: {FLECHAS_COMANDOS[comando]}   "
                    f"estado: {telemetria.estado.value}",
                    hud.COLOR_ACENTO,
                ),
                (
                    f"senal: {senal}   error: {telemetria.linea.error_px} px",
                    hud.COLOR_TEXTO,
                ),
                (
                    f"t: {mundo.t:7.1f}s   vueltas: {mundo.vueltas}   "
                    f"descarrilamientos: {mundo.descarrilamientos}",
                    hud.COLOR_TEXTO,
                ),
                (
                    f"PAREs cumplidos: {pares_cumplidos}   "
                    f"intervenciones: {intervenciones}",
                    hud.COLOR_TEXTO,
                ),
            ],
        )
        hud.dibujar_ayuda(pantalla, fuente_chica, disposicion, AYUDA)
        pygame.draw.rect(pantalla, hud.COLOR_BARRA, disposicion.rect_barra)
        controles.dibujar(pantalla, fuente)
        pygame.display.flip()

    guardar()
    pygame.quit()
    return {
        "duracion": mundo.t,
        "vueltas": mundo.vueltas,
        "descarrilamientos": mundo.descarrilamientos,
        "pares_cumplidos": pares_cumplidos,
        "intervenciones": intervenciones,
    }
