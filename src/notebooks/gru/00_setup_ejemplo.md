# Notebook de ejemplo: patrón de uso del código compartido

Este archivo NO es un notebook real — es una referencia rápida de cómo debe verse
la primera celda de cualquier notebook (GRU, SVM o XGBoost) para usar el código
compartido en `src/`.

## En local (Jupyter/VS Code), estando en la raíz del repo:

```python
import sys
sys.path.append("..")  # o la ruta relativa a la raíz del repo

from src.preprocessing.clean import load_and_clean, train_test_split_by_flow
from src.evaluation.metrics import evaluate_model, Timer

df = load_and_clean("../data/raw")
X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_by_flow(df)
```

## En Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<usuario-u-org>/<nombre-repo>.git
%cd <nombre-repo>
!pip install -r requirements.txt -q

from src.preprocessing.clean import load_and_clean, train_test_split_by_flow
from src.evaluation.metrics import evaluate_model, Timer

# Ajusta esta ruta a donde tengas el dataset en tu Drive
df = load_and_clean("/content/drive/MyDrive/CIC-IDS-2017/")
X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_by_flow(df)
```

## Al terminar de entrenar y evaluar

```python
with Timer() as t:
    modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]  # si aplica

resultados = evaluate_model(
    y_true=y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    model_name="GRU",  # o "SVM" / "XGBoost"
    training_time_seconds=t.elapsed,
)
```

Esto guarda automáticamente `reports/metrics/gru.json` (o `svm.json`, `xgboost.json`)
con el mismo formato para los tres, listo para la tabla comparativa de la Fase 7-8.

---

**Borra este archivo cuando tengas tu primer notebook real** (`01_eda.ipynb`,
`02_preprocesamiento.ipynb`, etc.) — es solo una guía inicial.
