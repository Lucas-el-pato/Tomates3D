# data/

| Carpeta | Contenido | ¿En git? |
|---|---|---|
| `raw/` (o `../datasets/`) | Dataset en bruto **Laboro Tomato** (`laboro_tomato.zip`, 1,6 GB) | **No** (ver abajo) |
| `capturas_reales/` | 10 pares estéreo (izquierda/derecha, 1280×720) del tomate a 47 cm sobre el brazo robótico y `mediciones.csv` con la distancia real | Sí |

## Dataset en bruto: Laboro Tomato

- **Fuente:** Laboro.AI, <https://github.com/laboroai/LaboroTomato>, licencia **CC BY-NC 4.0**.
- **Contenido:** 804 fotos de invernadero (3024×4032 px), 643 de entrenamiento y 161 de test, anotadas en formato COCO con polígonos. Tiene 6 categorías: tomate grande (`b_`) y cherry (`l_`), cada uno verde, semimaduro o maduro.
- **Estructura del zip:** `annotations/train.json`, `annotations/test.json` y las imágenes (`train/`, `test/`).

### Por qué no está en el repositorio

GitHub rechaza archivos de más de **100 MB**, y Git LFS gratuito tiene un límite de 1 GB; el zip pesa **1,6 GB**. Como es un dataset público, el repo incluye cómo descargarlo y no el archivo. `datasets/`, `data/raw/` y `*.zip` están en `.gitignore`.

### Cómo descargarlo

Desde la raíz del repositorio:

```bash
mkdir -p datasets && curl -L -o datasets/laboro_tomato.zip http://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato.zip
```

Es la misma URL que usa `notebooks/01_entrenamiento.ipynb`, que en Colab lo descarga solo, lo convierte (preprocesamiento) y lo guarda en Drive. `notebooks/05_hiperparametros_visualizacion.ipynb` lee el zip directamente desde `datasets/laboro_tomato.zip`, sin descomprimirlo, para calcular la matriz de confusión en test.

## Pesos del modelo

Los pesos (`*.pt`) tampoco se suben. Se obtienen de dos formas:

1. **Reentrenando:** se ejecuta `notebooks/01_entrenamiento.ipynb` en Colab con GPU T4 (~2 h para los 3 modelos). El mejor queda en `weights/best.pt`.
2. **Pidiéndolos a los autores:** `weights/best.pt` es el MODELO 3, yolo11m-seg con lr 0,0005, unos 45 MB.

Las métricas y curvas de todos los entrenamientos sí están en `../results/entrenamiento/`.
