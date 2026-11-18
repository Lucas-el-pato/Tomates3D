# Tomates 3D: detección de tomates maduros y semimaduros y su posición (X, Y, Z)

El sistema detecta **tomates maduros y semimaduros** con **YOLO11-seg** y calcula la posición 3D de cada uno, en metros, con un par de **webcams estéreo**. Como método alternativo incluye un detector clásico con **OpenCV**. Todo vive en **notebooks Jupyter** pensados para Google Colab. El notebook de la API sirve una **página web** desde la que se calibran las cámaras y se ven los resultados en vivo.

- **Página del proyecto (GitHub Pages):** https://lucas-el-pato.github.io/Tomates3D/
- **Reporte técnico (formato IEEE):** [`reporte/`](reporte/) (LaTeX para compilar en Overleaf)
- **Autores:** Lucas Garcia y Danis Gonzalez, Ingeniería Mecatrónica, FIUNA

```
 Navegador (PC, 2 webcams)          HTTPS / WSS           Google Colab (03_api)
 ┌──────────────────────────┐   túnel cloudflared   ┌──────────────────────────────────┐
 │ Calibrar · Imagen · Vivo │ ────────────────────▶ │ FastAPI + página web             │
 │ tabla X,Y,Z + gráfico 3D │ ◀──────────────────── │ funciones de 02_ejecucion:       │
 └──────────────────────────┘                       │  rectificar → YOLO11-seg / OpenCV│
                                                    │  → máscaras izq./der. →          │
                                                    │  disparidad (SGBM de respaldo) → │
                                                    │  X, Y, Z                         │
                                                    └──────────────────────────────────┘
```

## Estructura del repositorio
```
Tomates3D/
├── data/            Datos de entrada: cómo obtener el dataset (no se sube, 1,6 GB) y las capturas reales
├── notebooks/       01 entrenamiento · 02 ejecución · 03 API · 04 OpenCV · 05 hiperparámetros y visualización
├── results/         CSV de la búsqueda de hiperparámetros, métricas, matriz de confusión, prueba real y figuras
├── reporte/         Artículo IEEE en LaTeX para Overleaf (main.tex + figuras)
├── config/          Calibraciones genéricas de la Logitech C270
├── Docs/            Papers de referencia
├── Presentación/    Presentación del trabajo final
├── index.html       Página de GitHub Pages (CV y proyecto); imágenes en assets/
└── stereo_calib.yml Calibración estéreo real usada en las pruebas
```

## Contenido
| Archivo | Para qué |
|---|---|
| `notebooks/01_entrenamiento.ipynb` | **Preprocesamiento y entrenamiento.** Descarga Laboro Tomato y lo convierte a YOLO-seg con **una clase `tomate` (maduro + semimaduro)**; los verdes quedan como fondo. Hace la partición train/val/test y entrena y compara **3 modelos** (M1, M2, M3). Se corre **una sola vez**. |
| `notebooks/02_ejecucion.ipynb` | **Todas las funciones** del sistema: calibración, SGBM, detector, localización 3D, `procesar_par()`, visualización y evaluación. Trae demos: una prueba sintética sin cámaras y el procesamiento de un par real. |
| `notebooks/03_api.ipynb` | Carga las funciones del 02 y del 04, levanta la API y la página web y las publica con un túnel. Tiene un selector de método: modelo YOLO u OpenCV. |
| `notebooks/04_opencv.ipynb` | Detector clásico, el modelo base sin aprendizaje: umbral de rojo en HSV, morfología, transformada de distancia y watershed. |
| `notebooks/05_hiperparametros_visualizacion.ipynb` | **Búsqueda de hiperparámetros** (tabla de los 6 entrenamientos, AdamW y por qué la búsqueda fue manual) y **visualización**: curvas de entrenamiento, predicciones, matriz de confusión en test e histogramas de errores. |
| `data/` | Instrucciones para descargar el dataset y 10 pares estéreo reales con su distancia medida. Ver [`data/README.md`](data/README.md). |
| `results/` | `grid_hiperparametros.csv`, `comparacion_test_v1/v2.csv`, `matriz_confusion_test.csv`, `prueba_real_versiones.csv`, métricas por entrenamiento y figuras. |
| `config/stereo_calib_generico_720p*.yml` | Calibraciones teóricas de la Logitech C270 (B = 22 cm y 10 cm) para probar sin calibrar. |
| `Docs/` | Papers de referencia. |

## Resultados principales
| Modelo | Arquitectura | lr0 (AdamW) | Parámetros | mAP50-95 máscara (test) | F1 (test) | ms/img (T4) |
|---|---|---|---|---|---|---|
| M1 | yolo11s-seg | 0,002 | 10,1 M | 0,753 | 0,877 | 10,0 |
| M2 | yolo11s-seg | 0,0005 | 10,1 M | 0,759 | 0,888 | 9,5 |
| **M3** (elegido) | yolo11m-seg | 0,0005 | 22,4 M | **0,766** | **0,891** | 25,7 |
| OpenCV | HSV + watershed | — | 0 | — | — | 78 (CPU) |

- **Búsqueda de hiperparámetros:** manual, una variable por vez (learning rate → tamaño del modelo → definición de la clase), 6 entrenamientos en total. Detalle en `notebooks/05_*` y `results/grid_hiperparametros.csv`.
- **Prueba real (tomate a 47 cm, 10 capturas):** M3 midió **47,8 cm**, con una dispersión de 0,2 mm. OpenCV dio una Z errónea (~35 cm) en 2 de 10 capturas.
- **Calibración estéreo:** RMS de 0,37 px y línea base de 8,14 cm.

## Puesta en marcha (Colab)
1. **Subir el proyecto a Drive:** copiar esta carpeta a `MyDrive/tomates3d/`, de modo que exista `MyDrive/tomates3d/notebooks/`.
2. **Entrenar (una vez):** abrir `01_entrenamiento.ipynb` con GPU T4 y ejecutar todo. Deja `MyDrive/tomates3d/weights/best.pt`.
   - Sin este paso, el sistema usa un modelo COCO sustituto que toma naranjas, manzanas y pelotas como "tomate". Sirve para probar con una pelota.
3. **Levantar la API:** abrir `03_api.ipynb` con GPU T4 y ejecutar todo. Al final imprime una URL `https://….trycloudflare.com`. **Abrí esa URL en el navegador de la PC** que tiene las cámaras. No hace falta instalar ni servir nada más.
4. **Cámaras:** elegir cuál es la izquierda y cuál la derecha (⇄ las intercambia) y la resolución, y tocar **Abrir cámaras**. En el desplegable **Método de reconocimiento** se elige entre el modelo YOLO y OpenCV.
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
pip install ultralytics opencv-contrib-python-headless scikit-image fastapi "uvicorn[standard]" python-multipart matplotlib pandas
```
Con `03_api` corriendo localmente, la página está en http://localhost:8000.

### Reproducir los resultados
1. Descargar el dataset en `datasets/laboro_tomato.zip` (ver [`data/README.md`](data/README.md)).
2. Entrenar con `01_entrenamiento` en Colab. Las carpetas `runs/` y `weights/` no se suben al repo por tamaño.
3. Ejecutar `05_hiperparametros_visualizacion` para regenerar las tablas y figuras de `results/`.

## Cómo se calcula la posición
1. **Rectificación:** con la calibración (método de Zhang, vía OpenCV) se alinean las imágenes, de modo que un mismo punto quede en la misma fila en ambas.
2. **Detección:** YOLO11-seg (u OpenCV, según el método elegido) segmenta los **tomates maduros y semimaduros** en las imágenes izquierda y derecha rectificadas. Los verdes se ignoran.
3. **Disparidad por máscaras (método principal):** se empareja cada tomate de la izquierda con el de la derecha (misma fila, área parecida). La disparidad *d* es la diferencia horizontal entre los centroides de las dos máscaras. No depende de la textura de la piel y da directamente el centro del fruto.
4. **SGBM de respaldo:** si un tomate no tiene pareja, SGBM (Hirschmüller, 2008) calcula la disparidad píxel a píxel y se toma la mediana dentro de la máscara erosionada, entre los percentiles 20 y 80.
   - La imagen se rellena a la izquierda para medir también con líneas base grandes.
   - La resolución se ajusta para que el cálculo sea rápido.
5. **Posición:** con la focal *f*, la línea base *B* y el centro óptico (cx, cy):
   `Z = f·B/d`, `X = (u−cx)·Z/f`, `Y = (v−cy)·Z/f`.
   El origen es la cámara izquierda: X a la derecha, Y hacia abajo, Z hacia adelante.
6. **Diámetro:** `2·√(área/π)·Z/f`. Con SGBM, para el centro del fruto se suma un radio a Z.

## API (notebook 03)
| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Página web |
| GET | `/health` | Estado del modelo y de la calibración |
| POST | `/calibration/start` · `/calibration/frame` · `/calibration/compute` | Asistente de calibración |
| DELETE | `/calibration/frame/{i}` | Descarta un par |
| GET / PUT | `/calibration` | Ver o descargar (`?format=yml`) / subir un yml |
| POST | `/calibration/verify` | Par rectificado con líneas epipolares; con `x`, `y` devuelve Z |
| POST | `/calibration/scale` | Corrige la escala de la calibración con un factor o con la línea base medida |
| POST | `/detect` | multipart `left`, `right`, `conf`, `images`, `method` (`modelo` u `opencv`) → tomates con X, Y, Z |
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
- **Detector:** mAP50 y mAP50-95 de cajas y máscaras, que imprime el notebook 01. La comparación completa y la matriz de confusión están en el notebook 05.
- **Calibración:** RMS de reproyección y línea base calculada frente a la medida con regla.
- **Profundidad:**
  1. Capturar pares con un tomate a distancias medidas.
  2. Armar un CSV `left,right,z_true_m` (ejemplo: `data/capturas_reales/mediciones.csv`).
  3. Completar `CSV_MEDICIONES` en `02_ejecucion` y correr la celda. Reporta MAE, RMSE y sesgo.

## Referencias
- Ultralytics YOLO11 (2024) – modelo de detección y segmentación. https://docs.ultralytics.com/models/yolo11/
- Hirschmüller, H. (2008). *Stereo Processing by Semiglobal Matching and Mutual Information*. IEEE TPAMI 30(2), 328–341 – método SGBM. https://doi.org/10.1109/TPAMI.2007.1166
- Zhang, Z. (2000). *A Flexible New Technique for Camera Calibration*. IEEE TPAMI 22(11) – calibración con tablero. https://doi.org/10.1109/34.888718
- Zheng, S. et al. (2023). *Tomato Recognition and Localization Method Based on Improved YOLOv5n-seg Model and Binocular Stereo Vision*. *Agronomy*, 13(9), 2339 – base de la arquitectura (segmentación + disparidad dentro de la máscara). https://doi.org/10.3390/agronomy13092339
- Zhao, Bao, Mo, Li, Liu y Du (2025). *Design of tomato picking robot detection and localization system based on deep learning neural networks algorithm of Yolov5*. *Scientific Reports*, 15, 6180 – YOLOv5 + SGBM; justificación de SGBM. https://doi.org/10.1038/s41598-025-90080-6
- Laboro Tomato – dataset de entrenamiento (CC BY-NC 4.0). https://github.com/laboroai/LaboroTomato
- Loshchilov, I. y Hutter, F. (2019). *Decoupled Weight Decay Regularization* (AdamW). ICLR.
- Tsironis et al. (2020). *TomatOD*, ISPRS Archives XLIII-B3-2020 – estaba previsto como dataset complementario, pero no se usó porque su servidor de descarga dejó de estar disponible. https://isprs-archives.copernicus.org/articles/XLIII-B3-2020/1077/2020/isprs-archives-XLIII-B3-2020-1077-2020.pdf

## Limitaciones
- **Latencia:** el modo en vivo depende del ancho de banda PC→Colab; se esperan unos 3–8 fps.
- **Sincronización:** dos webcams USB no están sincronizadas por hardware, así que con movimiento rápido aparece error en Z.
- **Precisión:** el error de Z crece con Z²/(f·B). Con las cámaras a 8–10 cm la precisión es de milímetros entre 0,3 y 0,6 m. En las pruebas reales hubo un corrimiento constante de +0,7 a +1 cm, porque el centro óptico está detrás del frente de la cámara.
- **Zona útil:** solo se mide lo que ven **ambas** cámaras. Con líneas base grandes esa zona es angosta a corta distancia.
- **Cambio de dominio:** el modelo se entrenó con fotos de invernadero. En el laboratorio, las versiones con semimaduros detectaron objetos de madera anaranjada lejanos, que se descartan con un filtro de distancia.
