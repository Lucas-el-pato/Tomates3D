# Tomates 3D: detección de tomates maduros y su posición (X, Y, Z)

El sistema detecta **tomates maduros** con **YOLO11-seg** y calcula la posición 3D de cada uno, en metros, con un par de **webcams estéreo** y el método **SGBM**. Todo vive en **3 notebooks Jupyter**, pensados para Google Colab. El notebook de la API sirve una **página web** desde la que se calibran las cámaras y se ven los resultados en vivo.

```
 Navegador (PC, 2 webcams)          HTTPS / WSS           Google Colab (03_api)
 ┌──────────────────────────┐   túnel cloudflared   ┌──────────────────────────────────┐
 │ Calibrar · Imagen · Vivo │ ────────────────────▶ │ FastAPI + página web             │
 │ tabla X,Y,Z + gráfico 3D │ ◀──────────────────── │ funciones de 02_ejecucion:       │
 └──────────────────────────┘                       │  rectificar → YOLO11-seg →       │
                                                    │  SGBM → máscara + disparidad →   │
                                                    │  X, Y, Z                         │
                                                    └──────────────────────────────────┘
```

## Contenido
| Archivo | Para qué |
|---|---|
| `notebooks/01_entrenamiento.ipynb` | Descarga Laboro Tomato y tomatOD, arma un dataset de **una clase (tomate maduro)** y entrena YOLO11-seg. Se corre **una sola vez**. |
| `notebooks/02_ejecucion.ipynb` | **Todas las funciones** del sistema: calibración, SGBM, detector, localización 3D, `procesar_par()`, visualización y evaluación. Trae demos: una prueba sintética sin cámaras y el procesamiento de un par real. |
| `notebooks/03_api.ipynb` | Carga las funciones del 02, levanta la API y la página web y las publica con un túnel. |
| `config/stereo_calib_generico_720p*.yml` | Calibraciones teóricas de la Logitech C270 (B = 22 cm y 10 cm) para probar sin calibrar. |
| `Docs/` | Papers de referencia. |

## Puesta en marcha (Colab)
1. **Subir el proyecto a Drive:** copiar esta carpeta a `MyDrive/tomates3d/`, de modo que exista `MyDrive/tomates3d/notebooks/`.
2. **Entrenar (una vez):** abrir `01_entrenamiento.ipynb` con GPU T4 y ejecutar todo. Deja `MyDrive/tomates3d/weights/best.pt`.
   - Sin este paso, el sistema usa un modelo COCO sustituto que toma naranjas, manzanas y pelotas como "tomate". Sirve para probar con una pelota.
3. **Levantar la API:** abrir `03_api.ipynb` con GPU T4 y ejecutar todo. Al final imprime una URL `https://….trycloudflare.com`. **Abrí esa URL en el navegador de la PC** que tiene las cámaras. No hace falta instalar ni servir nada más.
4. **Cámaras:** elegir cuál es la izquierda y cuál la derecha (⇄ las intercambia) y la resolución, y tocar **Abrir cámaras**.
5. **Calibrar** (pestaña 1):
   - Imprimir el tablero al 100 % y **medir el cuadro con una regla**.
   - Capturar 15–25 pares con el tablero quieto, variando posición, inclinación y distancia.
   - Tocar **Calcular**. Si el RMS supera 0,7 px, descartar los pares rojos y volver a calcular. La línea base debe coincidir con la distancia **centro a centro** de las lentes.
   - **Verificar:** las líneas horizontales deben pasar por los mismos puntos en ambas imágenes. Al hacer clic sobre un objeto se ve su Z, para comparar con una cinta métrica.
   - La calibración queda en `MyDrive/tomates3d/stereo_calib.yml` y se recarga sola.
6. **Detectar:** pestaña 2 con un par estático, o pestaña 3 en vivo.

Para usar el sistema otro día solo hace falta el paso 3. La URL cambia en cada ejecución. Hay que recalibrar únicamente si se mueven las cámaras o se cambia la resolución.

### Sin cámaras
`02_ejecucion.ipynb` corre una **prueba sintética** que no necesita cámaras ni modelo entrenado:
1. Calibra con un tablero simulado; la línea base calculada da ≈ 8 cm.
2. Ubica un "tomate" simulado a 30, 45 y 60 cm y mide su Z. El error es menor a 1 mm.

### En la PC (sin Colab)
Los mismos notebooks corren localmente con Jupyter, abiertos desde la carpeta `notebooks/`. Se instalan las dependencias con:
```bash
pip install ultralytics opencv-contrib-python-headless fastapi "uvicorn[standard]" python-multipart matplotlib pandas
```
Con `03_api` corriendo localmente, la página está en http://localhost:8000.

## Cómo se calcula la posición
1. **Rectificación:** con la calibración (método de Zhang, vía OpenCV) se alinean las imágenes, de modo que un mismo punto quede en la misma fila en ambas.
2. **Detección:** YOLO11-seg segmenta los **tomates maduros** en la imagen izquierda rectificada. Los verdes y semimaduros se ignoran.
3. **Disparidad:** SGBM (Hirschmüller, 2008) calcula para cada píxel cuánto se desplaza entre la imagen izquierda y la derecha.
   - La imagen se rellena a la izquierda para medir también con líneas base grandes.
   - La resolución se ajusta para que el cálculo sea rápido.
4. **Profundidad de cada tomate:** se toma la mediana de la disparidad *d* dentro de la máscara erosionada, entre los percentiles 20 y 80. Con la focal *f*, la línea base *B* y el centro óptico (cx, cy):
   `Z = f·B/d`, `X = (u−cx)·Z/f`, `Y = (v−cy)·Z/f`.
   El origen es la cámara izquierda: X a la derecha, Y hacia abajo, Z hacia adelante.
5. **Diámetro:** `2·√(área/π)·Z/f`. Para el centro del fruto se suma un radio a Z.

## API (notebook 03)
| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Página web |
| GET | `/health` | Estado del modelo y de la calibración |
| POST | `/calibration/start` · `/calibration/frame` · `/calibration/compute` | Asistente de calibración |
| DELETE | `/calibration/frame/{i}` | Descarta un par |
| GET / PUT | `/calibration` | Ver o descargar (`?format=yml`) / subir un yml |
| POST | `/calibration/verify` | Par rectificado con líneas epipolares; con `x`, `y` devuelve Z |
| POST | `/detect` | multipart `left`, `right`, `conf`, `images` → tomates con X, Y, Z |
| WS | `/ws/stream` | `uint32 BE len(left)` + JPEG izq + JPEG der → JSON por par |

Ejemplo de tomate en la respuesta:
```json
{"id": 0, "conf": 0.91, "bbox": [612, 270, 742, 398],
 "xyz_m": [0.051, -0.028, 0.601],
 "xyz_center_m": [0.054, -0.030, 0.636],
 "diameter_m": 0.070, "disparity_px": 128.2, "depth_valid_ratio": 1.0}
```
- `xyz_m` es el punto de la superficie visible.
- `xyz_center_m` es el centro estimado del fruto.

## Evaluación sugerida para el informe
- **Detector:** mAP50 y mAP50-95 de cajas y máscaras, que imprime el notebook 01.
- **Calibración:** RMS de reproyección y línea base calculada frente a la medida con regla.
- **Profundidad:**
  1. Capturar pares con un tomate a distancias medidas.
  2. Armar un CSV `left,right,z_true_m`.
  3. Completar `CSV_MEDICIONES` en `02_ejecucion` y correr la celda. Reporta MAE, RMSE y sesgo.

## Referencias
- Ultralytics YOLO11 (2024) – modelo de detección y segmentación. https://docs.ultralytics.com/models/yolo11/
- Hirschmüller, H. (2008). *Stereo Processing by Semiglobal Matching and Mutual Information*. IEEE TPAMI 30(2), 328–341 – método SGBM. https://doi.org/10.1109/TPAMI.2007.1166
- Zhang, Z. (2000). *A Flexible New Technique for Camera Calibration*. IEEE TPAMI 22(11) – calibración con tablero. https://doi.org/10.1109/34.888718
- *Tomato Recognition and Localization Method Based on Improved YOLOv5n-seg Model and Binocular Stereo Vision*. *Agronomy* (2023), 13(9), 2339 – base de la arquitectura (segmentación + disparidad dentro de la máscara). https://doi.org/10.3390/agronomy13092339
- Zhao, Bao, Mo, Li, Liu y Du (2025). *Design of tomato picking robot detection and localization system based on deep learning neural networks algorithm of Yolov5*. *Scientific Reports*, 15, 6180 – YOLOv5 + SGBM; justificación de SGBM. https://doi.org/10.1038/s41598-025-90080-6
- Laboro Tomato – dataset de entrenamiento (CC BY-NC 4.0). https://github.com/laboroai/LaboroTomato
- Tsironis et al. (2020). *TomatOD*, ISPRS Archives XLIII-B3-2020 – dataset de entrenamiento (CC BY-NC-SA 4.0). https://isprs-archives.copernicus.org/articles/XLIII-B3-2020/1077/2020/isprs-archives-XLIII-B3-2020-1077-2020.pdf

## Limitaciones
- **Latencia:** el modo en vivo depende del ancho de banda PC→Colab; se esperan unos 3–8 fps.
- **Sincronización:** dos webcams USB no están sincronizadas por hardware, así que con movimiento rápido aparece error en Z.
- **Precisión:** el error de Z crece con Z²/(f·B). Con las cámaras a 8–10 cm la precisión es de milímetros entre 0,3 y 0,6 m.
- **Zona útil:** solo se mide lo que ven **ambas** cámaras. Con líneas base grandes esa zona es angosta a corta distancia.
