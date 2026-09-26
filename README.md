Resumen:

Lo que quedó construido

```
src/first/
├── main.py          # CLI: --source sim|webcam|video, --pare-segundos, --tolerancia
├── ports.py         # FrameSource / CommandSink (ABCs)
├── calib.py         # tu detector de polígonos original (calibración con trackbars)
├── core/            # CEREBRO puro — sin E/S, 100% portable al Arduino
│   ├── brain.py     # FSM: SIGUIENDO/PARE/RECUPERANDO/PERDIDO
│   ├── control.py   # histéresis (Schmitt) + votación por mayoría (deque)
│   ├── senales.py   # confirmación por mayoría + bloqueo + enmascarado
│   └── vision/      # kmeans.py (desde cero), line.py, sign.py, debug.py
└── adapters/        # webcam, video_file, serial_arduino (stub), sim/
    └── sim/         # pista.py (geometría) + realismo.py/trazo.py/defectos.py (render
                     # imperfecto) + world.py (física y métricas) + camera.py +
                     # pygame_ui.py/hud.py/controles.py/arrastre.py (UI) + config.py
```

Resultados del simulador: 3 vueltas de 31.6s (tiempo teórico), distancia a línea p95 = 15.6px, 0 descarrilamientos, PAREs 100% cumplidos. 74/74 tests.
Las técnicas "ingeniosas" para diferenciarse (rúbrica)

1. K-Means propio en numpy (init determinista por cuantiles, clusters vacíos reasignados) + criterio línea = oscuro Y desaturado (inmune a señales de color)
2. Segmento dominante por histograma de columnas ponderado por fila (estilo barra de sensores IR) — no promedia en codos
3. Doble ROI con anticipación saturada ±30px y descarte de línea lejana "horizontal"
4. FSM + histéresis + votación + debounce — control discreto estable
5. PARE blindado: confirmación 3 frames, bloqueo de misma señal, cooldown, solo con línea visible
6. Atención central gaussiana anclada a la predicción: al elegir entre candidatos, la evidencia (masa x cobertura) se multiplica por una gaussiana centrada donde estaba la línea el frame anterior, más un bono al segmento que toca el borde inferior. Es lo que permite pistas con tramos paralelos (chicane) y el cruce del ocho: se ven dos líneas y hay que quedarse con la propia. Medido en chicane: de 1 vuelta / 5 descarrilamientos a 2 vueltas / 2

Cómo usarlo

uv run first                    # simulador pygame (A=auto, R=reinicia, flechas=manual cuenta intervención)
                                # barra en vivo: modo, tolerancia, PARE, cortina, atencion, velocidad x
                                # mouse: arrastra el carrito y las señales sobre la pista
                                # la disposición se guarda en sim_config.json (el CLI explícito manda)
uv run first --source webcam    # cámara real
uv run first --pare-segundos 5  # el día de la competencia
uv run pytest                   # 74 tests

El mapeo a Arduino es la tabla PRESETS en adapters/sim/world.py → serial_arduino.py envía U/R/D/L\n (solo falta uv add pyserial cuando llegue el hardware). Los warnings de SDL al abrir pygame son cosméticos (opencv+pygame duplican SDL en macOS).