# Reporte técnico (formato IEEE)

Artículo en LaTeX con la clase `IEEEtran` (modo *conference*), según la plantilla del curso: [diegostaPy/CursoRedacIEEE](https://github.com/diegostaPy/CursoRedacIEEE).

| Archivo | Contenido |
|---|---|
| `main.tex` | Artículo completo, con la bibliografía al final (16 referencias en formato IEEE) |
| `figuras/` | 10 figuras cuadradas con letra grande y coma decimal (PDF para LaTeX, PNG para vista previa) |
| `Tomates3D_overleaf.zip` | `main.tex` + `figuras/*.pdf`, listo para subir a Overleaf (no se sube a git; se regenera con la carpeta) |

## Cómo compilar en Overleaf

1. En [Overleaf](https://www.overleaf.com): **New Project → Upload Project** y elegí `Tomates3D_overleaf.zip`. Si no está, comprimí `main.tex` y la carpeta `figuras/`.
2. Menú → **Compiler: pdfLaTeX**.
3. **Recompile** y descargá el PDF.
4. Guardá el PDF en esta carpeta como `Reporte_Tomates3D_IEEE.pdf`.

`IEEEtran.cls` ya viene en Overleaf. La bibliografía está escrita dentro de `main.tex` (`thebibliography`), así que no hace falta BibTeX.

## De dónde salen los datos

Todos los números de las tablas y figuras vienen de `../results/`, que regenera `notebooks/05_hiperparametros_visualizacion.ipynb`, y de la prueba real en `../data/capturas_reales/`.
