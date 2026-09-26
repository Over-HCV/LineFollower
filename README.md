Resumen:

Lo que quedó construido
src/first/
├── main.py          # CLI: --source sim|webcam|video, --pare-segundos, --tolerancia
├── ports.py         # FrameSource / CommandSink (ABCs)
├── calib.py         # tu detector de polígonos original (calibración con trackbars)
├── core/            # CEREBRO puro — sin E/S, 100% portable al Arduino
│   ├── brain.py     # FSM: SIGUIENDO/PARE/RECUPERANDO/PERDIDO
│   ├── control.py   # histéresis (Schmitt) + votación por mayoría (deque)
│   └── vision/      # kmeans.py (desde cero), line.py, sign.py
└── adapters/        # webcam, video_file, serial_arduino (stub), sim/

Resultados del simulador: 3 vueltas de 31.6s (tiempo teórico), distancia a línea p95 = 15.6px, 0 descarrilamientos, PAREs 100% cumplidos. 27/27 tests.
Las técnicas "ingeniosas" para diferenciarse (rúbrica)

1. K-Means propio en numpy (init determinista por cuantiles, clusters vacíos reasignados) + criterio línea = oscuro Y desaturado (inmune a señales de color)
2. Segmento dominante por histograma de columnas ponderado por fila (estilo barra de sensores IR) — no promedia en codos
3. Doble ROI con anticipación saturada ±30px y descarte de línea lejana "horizontal"
4. FSM + histéresis + votación + debounce — control discreto estable
5. PARE blindado: confirmación 3 frames, bloqueo de misma señal, cooldown, solo con línea visible

Cómo usarlo

uv run first                    # simulador pygame (A=auto, R=reinicia, flechas=manual cuenta intervención)
uv run first --source webcam    # cámara real
uv run first --pare-segundos 5  # el día de la competencia
uv run pytest                   # 27 tests

El mapeo a Arduino es la tabla PRESETS en adapters/sim/world.py → serial_arduino.py envía U/R/D/L\n (solo falta uv add pyserial cuando llegue el hardware). Los warnings de SDL al abrir pygame son cosméticos (opencv+pygame duplican SDL en macOS).