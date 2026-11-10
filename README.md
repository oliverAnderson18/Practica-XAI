# Práctica XAI: reservas de hotel

Selección de variables, modelo interpretable y explicaciones SHAP.

## Organización
- data/raw/: CSV original, sin modificar.
- data/processed/: datos derivados y particiones reproducibles.
- src/: funciones compartidas de preparación, entrenamiento y explicación.
- notebooks/: desarrollo paso a paso de la práctica.
- models/: pipelines entrenados serializados (.pickle) y sus metadatos.
- reports/: figuras, tablas y memoria.

## Entorno
Se utilizará uv. Desde la raíz del repositorio:

```powershell
uv sync
uv run jupyter lab
```

Las dependencias se añadirán a medida que se utilicen. Se debe versionar uv.lock cuando se genere.

## Reproducibilidad
- Python 3.12 y semilla 42.
- Rutas relativas al repositorio.
- Test estratificado del 25 %, reservado para la evaluación final.
- Preprocesamiento y selección de variables ajustados solo con entrenamiento.
- Guardar los pipelines completos y registrar variables, umbral y versiones.
