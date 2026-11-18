# Tomates 3D: detección de tomates maduros y semimaduros y su posición (X, Y, Z)

El sistema detecta **tomates maduros y semimaduros** con **YOLO11-seg** y calcula la posición 3D de cada uno, en metros, con un par de **webcams estéreo** y el método **SGBM**. Todo vive en **3 notebooks Jupyter**, pensados para Google Colab. El notebook de la API sirve una **página web** desde la que se calibran las cámaras y se ven los resultados en vivo.

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
| `notebooks/01_entrenamiento.ipynb` | Prepara Laboro Tomato con **una clase `tomate` (maduros + semimaduros)**, entrena **3 modelos** para compararlos y copia el mejor a `weights/best.pt`. Se corre **una sola vez** y se puede cortar y retomar. |
| `notebooks/02_ejecucion.ipynb` | **Todas las funciones** del sistema: calibración, SGBM, detector, localización 3D, `procesar_par()`, visualización y evaluación. Trae demos: una prueba sintética sin cámaras y el procesamiento de un par real. |
| `notebooks/03_api.ipynb` | Carga las funciones del 02 y del 04, levanta la API y la página web y las publica con un túnel. |
| `notebooks/04_opencv.ipynb` | **Detector clásico con OpenCV** (color HSV + watershed), adaptado de `Open CV/VisionEsteroscopicaV3.py`. Es el método alternativo al modelo; trae una prueba sintética y la comparación YOLO vs OpenCV en un par real. |
| `config/stereo_calib_generico_720p*.yml` | Calibraciones teóricas de la Logitech C270 (B = 22 cm y 10 cm) para probar sin calibrar. |
| `Docs/` | Papers de referencia. |

## Puesta en marcha (Colab)
1. **Subir el proyecto a Drive:** copiar esta carpeta a `MyDrive/tomates3d/`, de modo que exista `MyDrive/tomates3d/notebooks/`.
2. **Entrenar (una vez):** abrir `01_entrenamiento.ipynb` con GPU T4 y ejecutar todo (~3 h en total). Deja `MyDrive/tomates3d/weights/best.pt`, con el mejor de los 3 modelos.
   - Si Colab se desconecta, volvé a ejecutar todo: no se vuelve a descargar nada, los modelos terminados se saltean y el que quedó a medias continúa.
   - Sin este paso, el sistema usa un modelo COCO sustituto que toma naranjas, manzanas y pelotas como "tomate". Sirve para probar con una pelota.
3. **Levantar la API:** abrir `03_api.ipynb` con GPU T4 y ejecutar todo. Al final imprime una URL `https://….trycloudflare.com`. **Abrí esa URL en el navegador de la PC** que tiene las cámaras. No hace falta instalar ni servir nada más.
4. **Cámaras:** elegir cuál es la izquierda y cuál la derecha (⇄ las intercambia) y la resolución, y tocar **Abrir cámaras**.
5. **Calibrar** (pestaña 1):
   - Imprimir el tablero al 100 % y **medir el cuadro con una regla**.
   - Capturar 15–25 pares con el tablero quieto, variando posición, inclinación y distancia.
   - Tocar **Calcular**. Si el RMS supera 0,7 px, descartar los pares rojos y volver a calcular. La línea base debe coincidir con la distancia **centro a centro** de las lentes.
   - **Verificar:** las líneas horizontales deben pasar por los mismos puntos en ambas imágenes. Al hacer clic sobre un objeto se ve su Z, para comparar con una cinta métrica.
   - La calibración queda en `MyDrive/tomates3d/stereo_calib.yml` y se recarga sola.
6. **Elegir el método:** en el panel izquierdo, el desplegable **Método de reconocimiento** elige entre **Modelo YOLO11-seg** y **OpenCV**. Vale para las pestañas 2 y 3, y se puede cambiar en pleno modo en vivo.
7. **Detectar:** pestaña 2 con un par estático, o pestaña 3 en vivo.

Para usar el sistema otro día solo hace falta el paso 3. La URL cambia en cada ejecución. Hay que recalibrar únicamente si se mueven las cámaras o se cambia la resolución.

### Sin cámaras
`02_ejecucion.ipynb` corre una **prueba sintética** que no necesita cámaras ni modelo entrenado:
1. Calibra con un tablero simulado; la línea base calculada da ≈ 8 cm.
2. Ubica un "tomate" simulado a 30, 45 y 60 cm y mide su Z. El error es menor a 1 mm.

### En la PC (sin Colab)
Los mismos notebooks corren localmente con Jupyter, abiertos desde la carpeta `notebooks/`. Se instalan las dependencias con:
```bash
pip install ultralytics opencv-contrib-python-headless scikit-image fastapi "uvicorn[standard]" python-multipart matplotlib pandas
```
Con `03_api` corriendo localmente, la página está en http://localhost:8000.

## Entrenamiento comparativo (notebook 01)
| Modelo | Arquitectura | Learning rate (AdamW) | Qué mide |
|---|---|---|---|
| MODELO 1 | yolo11s-seg | 0.002 | Configuración base, la que elegía Ultralytics con `optimizer="auto"` |
| MODELO 2 | yolo11s-seg | 0.0005 | Efecto del learning rate (4× menor) |
| MODELO 3 | yolo11m-seg | el mejor de 1 y 2 | Modelo más grande, con más capacidad para tomates ocluidos |

- **Dataset:** Laboro Tomato, con los tomates maduros (`*_fully_ripened`) y semimaduros (`*_half_ripened`) unidos en una clase única `tomate`; los verdes se descartan. Al principio solo se usaban los maduros y los semimaduros quedaban como fondo, pero el modelo aprendía a ignorar tomates naranjas y fallaba con luz cálida de interior, donde un tomate maduro se ve naranja. tomatOD estaba previsto como complemento, pero su servidor de descarga dejó de estar disponible (HTTP 403).
- **Partición:** el `train` de Laboro se divide 85 % train / 15 % val con semilla fija. El `test` de Laboro queda reservado. El learning rate y el modelo final se eligen con **val**; la comparación se reporta en **test**.
- **Métricas:** Precision, Recall, F1, mAP50 y mAP50-95 de caja y de máscara, más ms por imagen, parámetros y horas de entrenamiento. Quedan en `MyDrive/tomates3d/runs/clase_tomate/comparacion_test.csv`, con gráficos y curvas en el notebook.
- **Una sola descarga:** `datasets/laboro_tomato.zip` y el dataset ya convertido (`datasets/tomate_yolo.zip`) se guardan en Drive. En cada sesión se descomprimen en el disco local de Colab, que es más rápido para entrenar.

## Métodos de reconocimiento
| | Modelo YOLO11-seg (notebooks 01/02) | OpenCV (notebook 04) |
|---|---|---|
| Cómo reconoce | Red neuronal entrenada con Laboro Tomato | Umbral de rojo en HSV, morfología, transformada de distancia y watershed |
| Qué detecta | Maduros y semimaduros | Solo tomates rojos |
| Tomates que se tocan | Los separa el modelo | Los separa el watershed |
| Necesita | Entrenamiento y, para ir rápido, GPU | Nada: corre en CPU |
| Punto débil | Escenas muy distintas a las del dataset | Cambios de luz y objetos rojos del fondo |
| "Confianza" | Probabilidad del modelo | Redondez mínima: área / área del círculo que lo encierra |

Los dos métodos usan la **misma calibración y el mismo cálculo de X, Y, Z**, así que solo cambia el reconocimiento. Del script original de OpenCV se conservan la segmentación y sus parámetros, pero no la fórmula empírica `posicion()`: la posición sale de la calibración estéreo.

## Cómo se calcula la posición
1. **Rectificación:** con la calibración (método de Zhang, vía OpenCV) se alinean las imágenes, de modo que un mismo punto quede en la misma fila en ambas.
2. **Detección:** YOLO11-seg segmenta los **tomates maduros y semimaduros** en ambas imágenes rectificadas; los verdes se ignoran. Con el método OpenCV, la máscara sale de la segmentación por color.
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
| POST | `/detect` | multipart `left`, `right`, `conf`, `images`, `method` (`modelo` u `opencv`) → tomates con X, Y, Z |
| WS | `/ws/stream` | Texto `{"conf": 0.35, "method": "opencv"}` para cambiar parámetros; binario `uint32 BE len(left)` + JPEG izq + JPEG der → JSON por par |

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
- **Detector:** tabla comparativa de los 3 modelos en test (notebook 01, sección 8) y preguntas guía para la discusión (sección 10).
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

## Limitaciones
- **Latencia:** el modo en vivo depende del ancho de banda PC→Colab; se esperan unos 3–8 fps.
- **Sincronización:** dos webcams USB no están sincronizadas por hardware, así que con movimiento rápido aparece error en Z.
- **Precisión:** el error de Z crece con Z²/(f·B). Con las cámaras a 8–10 cm la precisión es de milímetros entre 0,3 y 0,6 m.
- **Zona útil:** solo se mide lo que ven **ambas** cámaras. Con líneas base grandes esa zona es angosta a corta distancia.
