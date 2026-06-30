[README.md](https://github.com/user-attachments/files/29524856/README.md)
# ⚽ KNN Mundial 2026 — Match Predictor

Aplicación web de Machine Learning que predice el resultado de partidos del Mundial 2026 usando un clasificador K-Nearest Neighbors entrenado con más de 49.000 partidos internacionales.

🔗 **[Ver app en producción](https://knn-mundialista-mercadojosefina.onrender.com/)**

---

## ¿Qué hace?

Seleccionás dos selecciones nacionales, el modelo analiza sus ratings ELO históricos y predice si ganará el equipo local, el visitante, o si habrá empate. La app muestra además las probabilidades de cada resultado, los 3 partidos más similares encontrados por el algoritmo (los vecinos K=3), y un dashboard con el historial real de enfrentamientos entre los equipos seleccionados.

---

## Stack técnico

| Capa | Tecnología |
| Backend | Python · Flask · scikit-learn |
| Modelo | KNeighborsClassifier (K=3) · MinMaxScaler |
| Visualizaciones | Plotly |
| Deploy | Render |
| Datos | results.csv (+49K partidos) · elo_ratings_wc2026.csv |

---

## El modelo

### Variables de entrada (features)

| Variable | Descripción |
|---|---|
| `home_elo` | Rating ELO actual del equipo local |
| `away_elo` | Rating ELO actual del equipo visitante |
| `dif_elo` | Diferencia entre ambos ratings |
| `home_rank_avg` | Ranking FIFA promedio histórico (local) |
| `home_rating_avg` | Rating ELO promedio histórico (local) |
| `away_rank_avg` | Ranking FIFA promedio histórico (visitante) |
| `away_rating_avg` | Rating ELO promedio histórico (visitante) |

### Variable objetivo (target)
- `0` → Derrota local
- `1` → Empate
- `2` → Victoria local

### Métricas

| Set | Accuracy |
|---|---|
| Entrenamiento | 76% |
| Prueba | 61% |

```
              precision    recall  f1-score
Derrota         0.62      0.67      0.64
Empate          0.32      0.35      0.34
Victoria        0.72      0.66      0.69
accuracy                            0.61
```

### Decisiones de diseño
- Dataset filtrado desde 2014 para capturar el fútbol moderno
- Split 80/20 con `random_state=0` para reproducibilidad
- K óptimo determinado evaluando accuracy para K entre 1 y 29
- El modelo se entrena en build time (`train.py`) y se carga en runtime para optimizar memoria en el servidor

---

## Estructura del proyecto

```
├── app.py               # Backend Flask + endpoints de la API
├── train.py             # Entrenamiento y serialización del modelo
├── templates/
│   └── index.html       # Frontend (dos pantallas: selección y resultado)
├── data/
│   ├── results.csv      # Historial de partidos internacionales
│   └── elo_ratings_wc2026.csv
├── requirements.txt
├── render.yaml          # Configuración de deploy
└── Procfile
```

---

## Correr localmente

```bash
# 1. Clonar el repo
git clone https://github.com/SolJosefinaMercado/-knn-mundial.git
cd -knn-mundial

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Entrenar el modelo
python train.py

# 4. Levantar la app
python app.py
```

Abrí `http://localhost:5000` en tu navegador.

---

## API endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/teams` | Lista de equipos disponibles |
| POST | `/predict` | Predicción dado home_team y away_team |
| GET | `/history?home=X&away=Y` | Historial real de partidos entre dos equipos |

---

## Limitaciones conocidas y próximos pasos

- El dataset contiene partidos en cancha neutral y de local, lo que introduce un **sesgo de localía** en las predicciones. En un contexto mundialista (cancha neutral) esto debería corregirse reintroduciendo la variable `neutral` como feature.
- Las clases están desbalanceadas (818 victorias locales vs 451 empates en el set de entrenamiento), lo que explica el bajo F1-score del empate (0.34). Una mejora posible sería aplicar SMOTE o ajustar los pesos de clase.
- Próxima versión: incorporar forma reciente del equipo (últimos 5 partidos) como feature adicional.

---

## Datos

- **results.csv** — Historial de resultados de partidos internacionales masculinos (hasta junio 2026)
- **elo_ratings_wc2026.csv** — Ratings ELO de selecciones nacionales con snapshots anuales desde 2013

---

*Proyecto desarrollado como parte de mi formación en Data Science.*
