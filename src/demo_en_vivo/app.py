"""
Demo en vivo del detector de ataques Low & Slow con SVM.

Carga el modelo SVM entrenado (pipeline + umbral + características) y lo aplica
sobre flujos de la partición de prueba de CIC-IDS2017. No entrena ni modifica el
modelo: solo realiza inferencia y muestra las métricas.

Ejecución, desde la carpeta src/:

    streamlit run demo_en_vivo/app.py
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # src/demo_en_vivo
PROJECT_ROOT = ROOT.parent                      # src/

# El pipeline guardado usa la transformación logarítmica de
# src/preprocessing/clean.py, así que el paquete del proyecto debe ser importable.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# CONFIGURACION GENERAL
# ============================================================

DATA_DIR = ROOT / "data"
DEFAULT_MODEL = ROOT / "models" / "svm_final.joblib"

st.set_page_config(
    page_title="Low & Slow IDS | SVM",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        max-width: 1450px;
    }

    [data-testid="stMetric"] {
        background: rgba(127,127,127,.07);
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CARGA DE DATOS Y DEL MODELO
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def csv_from_upload(file) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file.getvalue()))


@st.cache_resource(show_spinner=False)
def load_model(path: str):

    p = Path(path)

    if not p.is_file():
        raise FileNotFoundError(f"No existe el modelo: {p}")

    if p.suffix.lower() not in (".joblib", ".pkl", ".pickle"):
        raise ValueError(f"Formato de modelo no admitido: {p.suffix}")

    return joblib.load(p)


def unpack_svm(artifact):
    """Extrae el pipeline, el umbral y la lista de características del artefacto."""

    if not isinstance(artifact, dict):
        raise ValueError(
            "El archivo SVM debe contener un diccionario "
            "con pipeline, umbral y caracteristicas."
        )

    missing = [
        key for key in ("pipeline", "umbral", "caracteristicas")
        if key not in artifact
    ]

    if missing:
        raise ValueError(f"Faltan claves en el SVM: {missing}")

    pipeline = artifact["pipeline"]
    threshold = float(artifact["umbral"])
    feature_names = artifact["caracteristicas"]

    if not hasattr(pipeline, "decision_function"):
        raise ValueError("El pipeline del SVM no tiene decision_function.")

    if not np.isfinite(threshold):
        raise ValueError("El umbral guardado no es valido.")

    return pipeline, threshold, feature_names


# ============================================================
# ETIQUETAS
# ============================================================

def extract_labels(y_frame: pd.DataFrame) -> np.ndarray:

    if y_frame.shape[1] != 1:
        raise ValueError("y_test.csv debe contener una sola columna.")

    y = y_frame.iloc[:, 0]

    if y.isna().any():
        raise ValueError("Las etiquetas contienen valores nulos.")

    if pd.api.types.is_numeric_dtype(y):

        values = y.to_numpy()

        if not np.isin(values, [0, 1]).all():
            raise ValueError("Las etiquetas deben ser binarias (0/1).")

        return values.astype(int)

    normalized = y.astype(str).str.strip().str.lower()

    mapping = {
        "0": 0,
        "benign": 0,
        "benigno": 0,
        "normal": 0,
        "1": 1,
        "attack": 1,
        "ataque": 1,
        "slowloris": 1,
        "slowhttptest": 1,
        "slowhttp": 1,
    }

    if not normalized.isin(mapping).all():
        bad = normalized[~normalized.isin(mapping)].unique()[:5]
        raise ValueError(f"Etiquetas no reconocidas: {bad.tolist()}")

    return normalized.map(mapping).to_numpy(dtype=int)


# ============================================================
# PREPARACION DE CARACTERISTICAS
# ============================================================

def prepare_features(frame: pd.DataFrame, model, feature_names) -> pd.DataFrame:

    x = frame.copy()

    # Eliminar indices exportados por pandas
    x = x.loc[:, ~x.columns.astype(str).str.match(r"^Unnamed: \d+$")]

    if x.columns.duplicated().any():
        raise ValueError("X_test contiene columnas duplicadas.")

    expected = list(feature_names)

    if not expected:
        raise ValueError("La lista de caracteristicas esta vacia.")

    missing = [col for col in expected if col not in x.columns]

    if missing:
        raise ValueError(f"Faltan caracteristicas: {missing[:12]}")

    # Respetar exactamente el orden del entrenamiento
    x = x.loc[:, expected]

    non_numeric = x.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    if non_numeric:
        raise ValueError(f"Caracteristicas no numericas: {non_numeric[:10]}")

    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("X_test contiene NaN o infinitos.")

    n_expected = getattr(model, "n_features_in_", None)

    if n_expected is not None and x.shape[1] != n_expected:
        raise ValueError(
            f"El modelo espera {n_expected} columnas, pero recibio {x.shape[1]}."
        )

    return x


# ============================================================
# PREDICCIONES
# ============================================================

def predict_svm(model, x: pd.DataFrame, threshold: float):

    classes = getattr(model, "classes_", None)

    if classes is not None and list(classes) != [0, 1]:
        raise ValueError(
            f"Se esperan clases [0, 1], pero se encontraron {classes}."
        )

    scores = np.asarray(model.decision_function(x))

    if scores.ndim != 1:
        raise ValueError("Se esperaba un SVM binario.")

    predictions = (scores >= threshold).astype(int)

    return predictions, scores


# ============================================================
# METRICAS Y GRAFICAS
# ============================================================

def evaluate(y_true, y_pred, scores):

    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }

    if len(np.unique(y_true)) == 2:
        result["ROC-AUC"] = roc_auc_score(y_true, scores)
        result["AP"] = average_precision_score(y_true, scores)
    else:
        result["ROC-AUC"] = float("nan")
        result["AP"] = float("nan")

    return result


def draw_confusion(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.imshow(cm, cmap="Blues")

    for (i, j), value in np.ndenumerate(cm):
        ax.text(
            j, i, f"{value:,}",
            ha="center", va="center", fontsize=14,
            color="white" if value > cm.max() / 2 else "black",
        )

    ax.set_xticks([0, 1], ["Benigno", "Ataque"])
    ax.set_yticks([0, 1], ["Benigno", "Ataque"])
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de confusion")
    fig.tight_layout()

    return fig, cm


def draw_score_distribution(y_true, scores, threshold):
    """Histograma de la puntuacion del SVM por clase, con el umbral de decision."""

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    low, high = np.percentile(scores, [0.5, 99.5])
    bins = np.linspace(min(low, threshold), max(high, threshold), 60)

    ax.hist(scores[y_true == 0], bins=bins, alpha=0.6, label="Benigno",
            color="#9a9a9a", log=True)
    ax.hist(scores[y_true == 1], bins=bins, alpha=0.7, label="Ataque",
            color="#2a78d6", log=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
               label=f"Umbral = {threshold:.3f}")

    ax.set(xlabel="Puntuacion del SVM (decision_function)",
           ylabel="Flujos (escala log)",
           title="Distribucion de la puntuacion")
    ax.legend(fontsize=8)
    fig.tight_layout()

    return fig


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

st.title("Detección de ataques Low & Slow con SVM")

st.caption(
    "Demostración interactiva de inferencia sobre flujos de red de CIC-IDS2017 "
    "· Support Vector Machine con kernel RBF"
)


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:

    st.header("Configuración")

    model_path = st.text_input("Ruta del modelo", str(DEFAULT_MODEL))

    st.divider()

    source = st.radio("Origen de datos", ["Archivos locales", "Subir CSV"])

    if source == "Archivos locales":

        x_path = st.text_input("Ruta X_test.csv", str(DATA_DIR / "X_test.csv"))
        y_path = st.text_input("Ruta y_test.csv", str(DATA_DIR / "y_test.csv"))
        x_upload = None
        y_upload = None

    else:

        x_upload = st.file_uploader("X_test.csv", type="csv", key="x")
        y_upload = st.file_uploader("y_test.csv (opcional)", type="csv", key="y")
        x_path = ""
        y_path = ""

    st.divider()

    try:
        _artifact = load_model(str(Path(model_path).resolve()))
        _, _umbral, _features = unpack_svm(_artifact)
        _hp = _artifact.get("hiperparametros", {})

        st.success(
            f"**Modelo SVM cargado**  \n"
            f"Kernel: {_artifact.get('kernel', 'rbf')}  \n"
            f"C = {_hp.get('C', 'N/D')} · γ = {_hp.get('gamma', 'N/D')}  \n"
            f"Características: {len(_features)}  \n"
            f"Umbral (F2 en validación): {_umbral:.3f}"
        )

    except Exception as exc:
        st.error(f"No se pudo cargar el modelo: {exc}")

    run = st.button(
        "Ejecutar predicciones",
        type="primary",
        use_container_width=True,
    )


st.info(
    "Esta demostración realiza inferencia con el modelo SVM previamente "
    "entrenado. No realiza entrenamiento ni modifica los parámetros del modelo. "
    "El umbral y las características se cargan desde svm_final.joblib."
)


# ============================================================
# EJECUCION DE INFERENCIA
# ============================================================

if run:

    try:

        with st.spinner("Cargando datos y ejecutando inferencia..."):

            if source == "Archivos locales":

                if not Path(x_path).is_file():
                    raise FileNotFoundError(f"No existe X_test: {x_path}")

                x_df = read_csv(str(Path(x_path).resolve()))
                y_df = (
                    read_csv(str(Path(y_path).resolve()))
                    if Path(y_path).is_file()
                    else None
                )

            else:

                if x_upload is None:
                    raise ValueError("Debes subir X_test.csv.")

                x_df = csv_from_upload(x_upload)
                y_df = csv_from_upload(y_upload) if y_upload else None

            artifact = load_model(str(Path(model_path).resolve()))
            model, threshold, feature_names = unpack_svm(artifact)
            x = prepare_features(x_df, model, feature_names)

            y = extract_labels(y_df) if y_df is not None else None

            if y is not None and len(y) != len(x):
                raise ValueError(
                    "X_test y y_test tienen tamaños diferentes: "
                    f"{len(x)} y {len(y)}."
                )

            started = time.perf_counter()
            preds, scores = predict_svm(model, x, threshold)
            elapsed = time.perf_counter() - started

            results = pd.DataFrame({
                "fila_original": np.arange(len(x)),
                "prediccion": preds,
                "clase_predicha": np.where(preds == 1, "Ataque", "Benigno"),
                "score": scores,
            })

            if y is not None:
                results["etiqueta_real"] = y
                results["correcto"] = preds == y

            st.session_state["demo_results"] = {
                "predictions": results,
                "labels": y,
                "scores": scores,
                "elapsed": elapsed,
                "n_input": len(x),
                "threshold": threshold,
            }

    except Exception as exc:

        st.error(f"No se pudo ejecutar la inferencia: {exc}")
        st.info(
            "Revisa el modelo, las dependencias, las características "
            "y el preprocesamiento."
        )


# ============================================================
# VISUALIZACION DE RESULTADOS
# ============================================================

if "demo_results" in st.session_state:

    data = st.session_state["demo_results"]
    result = data["predictions"]
    y_true = data["labels"]
    pred = result["prediccion"].to_numpy()
    scores = data["scores"]
    threshold = data["threshold"]

    st.subheader("Resultados · SVM")

    st.caption(
        f"{len(result):,} flujos evaluados · "
        f"Inferencia: {data['elapsed']:.3f} s · "
        f"Umbral: {threshold:.4f}"
    )

    if y_true is not None:

        # ------------------------- Metricas -------------------------
        metrics = evaluate(y_true, pred, scores)

        cols = st.columns(6)

        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.2%}" if np.isfinite(value) else "N/D")

        st.divider()

        # --------------- Matriz de confusion y errores ---------------
        left, right = st.columns(2)

        with left:

            fig, cm = draw_confusion(y_true, pred)
            st.pyplot(fig)
            plt.close(fig)

            tn, fp, fn, tp = cm.ravel()

            st.caption(f"TP: {tp:,} · TN: {tn:,} · FP: {fp:,} · FN: {fn:,}")

        with right:

            st.markdown("**Distribución de clases**")

            distribution = pd.DataFrame(
                {
                    "Real": np.bincount(y_true, minlength=2),
                    "Predicho": np.bincount(pred, minlength=2),
                },
                index=["Benigno", "Ataque"],
            )

            st.bar_chart(
                distribution,
                x_label="Clase",
                y_label="Cantidad de flujos",
                stack=False,
            )

            st.markdown("**Errores de clasificación**")

            errors = pd.DataFrame(
                {"Cantidad": [fp, fn]},
                index=["Falsos positivos", "Falsos negativos"],
            )

            st.bar_chart(errors)

        st.divider()

        # ------------------- Curvas y puntuacion -------------------
        if len(np.unique(y_true)) == 2:

            c1, c2, c3 = st.columns(3)

            with c1:

                fpr, tpr, _ = roc_curve(y_true, scores)

                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                ax.plot(fpr, tpr, label=f"AUC = {metrics['ROC-AUC']:.4f}")
                ax.plot([0, 1], [0, 1], "--", color="gray")
                ax.set(
                    xlabel="False Positive Rate",
                    ylabel="True Positive Rate",
                    title="Curva ROC",
                )
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

            with c2:

                precision, recall, _ = precision_recall_curve(y_true, scores)

                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                ax.plot(recall, precision, label=f"AP = {metrics['AP']:.4f}")
                ax.set(
                    xlabel="Recall",
                    ylabel="Precision",
                    title="Curva Precision-Recall",
                    xlim=(0, 1),
                    ylim=(0, 1.05),
                )
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

            with c3:

                fig = draw_score_distribution(y_true, scores, threshold)
                st.pyplot(fig)
                plt.close(fig)

        else:

            st.warning(
                "Solo existe una clase en y_test. "
                "No se pueden calcular ROC-AUC ni AP."
            )

    else:

        st.warning(
            "No se proporcionó y_test.csv. "
            "Se muestran predicciones sin métricas."
        )

        c1, c2 = st.columns(2)
        c1.metric("Benignos predichos", f"{(pred == 0).sum():,}")
        c2.metric("Ataques predichos", f"{(pred == 1).sum():,}")

        st.bar_chart(
            pd.Series(np.bincount(pred, minlength=2), index=["Benigno", "Ataque"])
        )

    # ----------------- Predicciones individuales -----------------
    st.divider()
    st.subheader("Predicciones individuales")

    st.dataframe(result.head(1000), use_container_width=True, hide_index=True)

    st.download_button(
        "Descargar todas las predicciones (CSV)",
        result.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_svm.csv",
        mime="text/csv",
    )


# ============================================================
# INSTRUCCIONES INICIALES
# ============================================================

else:

    st.subheader("Instrucciones")

    st.markdown(
        """
        1. Por defecto se usan `data/X_test.csv` y `data/y_test.csv`
           (flujos de la partición de prueba). También puedes subir tus propios CSV.

        2. El umbral de decisión y la lista de características se recuperan
           automáticamente de `models/svm_final.joblib`. `X_test.csv` trae los
           flujos sin escalar: el pipeline aplica el log con signo y el
           escalador de entrenamiento antes del SVM.

        3. Presiona **Ejecutar predicciones**.

        4. Revisa las métricas, la matriz de confusión, las curvas ROC y
           precisión-recall y la distribución de la puntuación.
        """
    )
