"""
Módulo de clasificación de riesgo agrícola con CatBoost.

Implementa el patrón Strategy: CatBoostTrainer encapsula la estrategia
de entrenamiento y puede sustituirse por otra implementación (XGBoost,
LightGBM, etc.) sin modificar el código que lo consume.
"""

from __future__ import annotations

import unicodedata
import warnings
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _normalizar(texto: str) -> str:
    """Normaliza texto a ASCII sin acentos para comparación robusta.

    El CSV siap_procesado.csv tiene codificación mixta: algunos acentos en
    latin-1 y otros en UTF-8, por lo que "Pérdida crítica" puede llegar como
    'PÃ©rdida crÃ­tica'. Esta función elimina todos los diacríticos y
    convierte a minúsculas para que el mapeo sea independiente de la
    codificación exacta.
    """
    s = unicodedata.normalize("NFKD", str(texto))
    s = s.encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()


def _label_a_clase(valor: str) -> int:
    """Mapea el texto de nivel_riesgo a su entero ordinal (0–4).

    Usa _normalizar internamente para ser robusto a mojibake.
    """
    s = _normalizar(valor)
    if s in ("", "nan", "none"):
        return 0
    if "sin" in s:
        return 0
    if "bajo" in s:
        return 1
    if "medio" in s:
        return 2
    if "alto" in s:
        return 3
    # "Pérdida crítica" normalizada → 'pardida cratica'
    return 4


# ---------------------------------------------------------------------------
# Constantes compartidas
# ---------------------------------------------------------------------------

RIESGO_MAP = {
    "Sin siniestro": 0,
    "Riesgo bajo": 1,
    "Riesgo medio": 2,
    "Riesgo alto": 3,
    "Pérdida crítica": 4,
}
RIESGO_INV = {v: k for k, v in RIESGO_MAP.items()}

# Features disponibles ANTES de la cosecha (sin data leakage post-cosecha)
FEATURES_SIEMBRA = [
    "Nomestado",
    "Nomcicloproductivo",
    "Nommodalidad",
    "Nomcultivo",
    "Sembrada",
    "Precio",
    "proporcion_siniestro",
]

CAT_FEATURES = ["Nomestado", "Nomcicloproductivo", "Nommodalidad", "Nomcultivo"]

PALETA = {
    0: "#2ecc71",  # Sin siniestro
    1: "#f1c40f",  # Riesgo bajo
    2: "#e67e22",  # Riesgo medio
    3: "#e74c3c",  # Riesgo alto
    4: "#8e44ad",  # Pérdida crítica
}


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class CatBoostTrainer:
    """Estrategia de entrenamiento de clasificación de riesgo agrícola con CatBoost.

    ¿Qué es CatBoost?
    -----------------
    CatBoost (Categorical Boosting) es un algoritmo de gradient boosting
    desarrollado por Yandex (2017) que extiende el boosting clásico con dos
    innovaciones principales:

    1. **Ordered boosting**: En el boosting estándar, cada árbol se entrena
       sobre los residuos de todos los datos, lo que introduce un sesgo porque
       la predicción de un ejemplo i depende de ejemplos que incluyen al
       propio i en su gradiente. CatBoost resuelve esto ordenando los datos
       aleatoriamente y usando, para calcular el gradiente del ejemplo i,
       *únicamente* los ejemplos anteriores a él en ese orden. El resultado
       es un estimador sin sesgo de gradiente, lo que reduce el sobreajuste
       especialmente en datasets pequeños o muy desbalanceados.

    2. **cat_features nativo (Target Statistics)**:
       CatBoost NO necesita LabelEncoder ni OneHotEncoder para las variables
       categóricas. En cambio, convierte cada categoría usando una estadística
       del target calculada *solo sobre el subconjunto de ejemplos anteriores*
       (ordered TS), evitando que información del futuro contamine la
       codificación de cada ejemplo.

       Esto elimina el riesgo de **data leakage** que introduce un
       LabelEncoder clásico:

       | Método              | ¿Ve el target de todo el dataset? | Leakage |
       |---------------------|-----------------------------------|---------|
       | LabelEncoder global | ✗ (solo codifica frecuencias)     | Parcial |
       | TargetEncoder global| ✓ (fit en train+val+test)         | Alto    |
       | CatBoost nativo     | Solo ejemplos anteriores (ordered)| Ninguno |

       En la práctica: si se aplica un TargetEncoder o un OrdinalEncoder
       *fit sobre el dataset completo antes del split temporal*, las
       estadísticas del target de 2023-2024 se "filtran" hacia los registros
       de 2010-2020, inflando artificialmente las métricas de test. CatBoost
       evita esto por diseño.

    Parámetros
    ----------
    params : dict | None
        Hiperparámetros para CatBoostClassifier. Si es None, usa los
        valores por defecto internos (optimizados para este dataset).
    random_state : int
        Semilla aleatoria para reproducibilidad. Por defecto 42.
    """

    # Hiperparámetros por defecto (resultado de la búsqueda Optuna)
    _DEFAULT_PARAMS: dict = {
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 1.0,
        "random_strength": 1.0,
        "auto_class_weights": "Balanced",
        "verbose": 0,
    }

    def __init__(self, params: dict | None = None, random_state: int = 42):
        self.random_state = random_state
        self.params = {**self._DEFAULT_PARAMS, "random_seed": random_state}
        if params:
            self.params.update(params)

        self.model: CatBoostClassifier | None = None
        self.evals_result_: dict = {}
        self.cat_features: list[str] = CAT_FEATURES
        self.features: list[str] = FEATURES_SIEMBRA

    # ------------------------------------------------------------------
    # Preparación de datos
    # ------------------------------------------------------------------

    def preparar_datos(
        self, df: pd.DataFrame
    ) -> tuple[
        pd.DataFrame, pd.Series,
        pd.DataFrame, pd.Series,
        pd.DataFrame, pd.Series,
    ]:
        """Aplica el split temporal y devuelve (X_train, y_train, X_val, y_val, X_test, y_test).

        Split:
          - Train → 2010–2020
          - Val   → 2021–2022
          - Test  → 2023–2024

        Las variables categóricas se pasan como **strings** directamente;
        CatBoost las codifica internamente con Target Statistics ordenadas,
        por lo que no se aplica ningún LabelEncoder ni OHE aquí.

        Las variables post-cosecha (Cosechada, Volumenproduccion, Rendimiento,
        Siniestrada, Valorproduccion) se excluyen para evitar data leakage:
        esas columnas no se conocen en el momento de la siembra.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame completo cargado con DataLoader. Debe contener la
            columna 'Anio' y 'nivel_riesgo'.

        Retorna
        -------
        Tupla de 6 elementos: X_train, y_train, X_val, y_val, X_test, y_test.
        """
        df = df.copy()

        # Mapear target de forma robusta a mojibake
        df["target"] = df["nivel_riesgo"].map(_label_a_clase)

        # Convertir columnas a string para CatBoost
        for col in self.cat_features:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("Desconocido")

        # Asegurar numéricos
        for col in ["Sembrada", "Precio", "proporcion_siniestro"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        anio = pd.to_numeric(df["Anio"], errors="coerce")
        mask_train = anio <= 2020
        mask_val = (anio >= 2021) & (anio <= 2022)
        mask_test = anio >= 2023

        X_train = df.loc[mask_train, self.features].reset_index(drop=True)
        y_train = df.loc[mask_train, "target"].reset_index(drop=True)

        X_val = df.loc[mask_val, self.features].reset_index(drop=True)
        y_val = df.loc[mask_val, "target"].reset_index(drop=True)

        X_test = df.loc[mask_test, self.features].reset_index(drop=True)
        y_test = df.loc[mask_test, "target"].reset_index(drop=True)

        print(f"[CatBoostTrainer] Split temporal:")
        print(f"  Train (2010-2020): {len(X_train):>7,} registros")
        print(f"  Val   (2021-2022): {len(X_val):>7,} registros")
        print(f"  Test  (2023-2024): {len(X_test):>7,} registros")

        return X_train, y_train, X_val, y_val, X_test, y_test

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def entrenar(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "CatBoostTrainer":
        """Entrena el modelo con early stopping y monitoreo de validación.

        Parámetros
        ----------
        X_train, y_train : DataFrame y Series de entrenamiento.
        X_val, y_val     : DataFrame y Series de validación (eval_set).

        El entrenamiento usa:
        - eval_set con X_val para monitorear el logloss en cada iteración.
        - early_stopping_rounds=50: detiene si no mejora en 50 iteraciones.
        - auto_class_weights='Balanced': compensa el desbalance de clases
          (95 % Sin siniestro) sin calcular pesos manualmente.

        Guarda el historial de pérdida en self.evals_result_ con las claves
        'learn' y 'validation', cada una con {'MultiClass': [...]}.

        Retorna
        -------
        self (para encadenamiento fluido).
        """
        cat_idx = [X_train.columns.get_loc(c) for c in self.cat_features if c in X_train.columns]

        train_pool = Pool(X_train, label=y_train, cat_features=cat_idx)
        val_pool = Pool(X_val, label=y_val, cat_features=cat_idx)

        self.model = CatBoostClassifier(**self.params)
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
            use_best_model=True,
        )

        # Extraer historial de pérdida
        self.evals_result_ = self.model.get_evals_result()
        best_iter = self.model.get_best_iteration()
        print(f"[CatBoostTrainer] Entrenamiento finalizado. "
              f"Mejor iteración: {best_iter}")

        return self

    # ------------------------------------------------------------------
    # Evaluación
    # ------------------------------------------------------------------

    def evaluar(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """Evalúa el modelo sobre el conjunto de test y devuelve métricas completas.

        Parámetros
        ----------
        X_test : DataFrame con las mismas columnas usadas en entrenamiento.
        y_test : Series con las etiquetas reales (int 0–4).

        Retorna
        -------
        dict con las claves:
          accuracy, balanced_accuracy, f1_macro, f1_weighted,
          precision_macro, recall_macro, log_loss, roc_auc_macro,
          kappa_cuadratico, classification_report, confusion_matrix.
        """
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a entrenar() primero.")

        y_pred = self.model.predict(X_test).flatten().astype(int)
        y_proba = self.model.predict_proba(X_test)

        clases = sorted(y_test.unique())

        # ROC-AUC multiclase one-vs-rest
        try:
            roc_auc = roc_auc_score(
                y_test, y_proba,
                multi_class="ovr",
                average="macro",
                labels=list(range(5)),
            )
        except ValueError:
            roc_auc = float("nan")

        metricas = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "log_loss": log_loss(y_test, y_proba, labels=list(range(5))),
            "roc_auc_macro": roc_auc,
            "kappa_cuadratico": cohen_kappa_score(y_test, y_pred, weights="quadratic"),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=[RIESGO_INV[i] for i in range(5)],
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=list(range(5))),
        }

        print(f"\n[CatBoostTrainer] Métricas en test:")
        for k in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
                  "roc_auc_macro", "kappa_cuadratico"]:
            print(f"  {k:<22}: {metricas[k]:.4f}")

        return metricas

    # ------------------------------------------------------------------
    # Interpretabilidad SHAP
    # ------------------------------------------------------------------

    def explicar_shap(
        self,
        X_sample: pd.DataFrame,
        tipo: Literal["summary", "bar", "waterfall", "dependence"] = "summary",
        indice: int | None = None,
        variable: str | None = None,
        color_variable: str | None = None,
        ruta_guardado: str | Path | None = None,
        clase_objetivo: int = 4,
    ) -> None:
        """Genera visualizaciones SHAP para interpretar el modelo.

        Parámetros
        ----------
        X_sample : DataFrame con las features (sin la columna target).
        tipo : Tipo de gráfica SHAP. Opciones:
            - 'summary'    : Beeswarm con todos los features y puntos coloreados
                             por valor de la feature.
            - 'bar'        : Barras de importancia media |SHAP| por feature.
            - 'waterfall'  : Descomposición de la predicción de UN ejemplo
                             (requiere `indice`).
            - 'dependence' : Efecto de una variable continua sobre SHAP
                             (requiere `variable`).
        indice : Índice de la fila a explicar (solo para tipo='waterfall').
        variable : Nombre de la columna para tipo='dependence'.
        color_variable : Variable de color en el dependence plot.
        ruta_guardado : Si se proporciona, guarda la figura en esa ruta.
        clase_objetivo : Clase (0-4) para la que se calculan los SHAP.
                         Por defecto 4 (Pérdida crítica).
        """
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado.")

        cat_idx = [X_sample.columns.get_loc(c) for c in self.cat_features if c in X_sample.columns]
        pool = Pool(X_sample, cat_features=cat_idx)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(pool)

        # shap_values puede ser una lista de arrays (una por clase) o un array 3D
        if isinstance(shap_values, list):
            sv = shap_values[clase_objetivo]
            base_val = explainer.expected_value[clase_objetivo]
        else:
            sv = shap_values[:, :, clase_objetivo]
            base_val = explainer.expected_value[clase_objetivo]

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.close(fig)  # se reabre dentro de cada plot

        if tipo == "summary":
            shap.summary_plot(
                sv, X_sample,
                feature_names=list(X_sample.columns),
                show=False,
                plot_type="dot",
            )
        elif tipo == "bar":
            shap.summary_plot(
                sv, X_sample,
                feature_names=list(X_sample.columns),
                show=False,
                plot_type="bar",
            )
        elif tipo == "waterfall":
            if indice is None:
                raise ValueError("Para tipo='waterfall' debes proporcionar `indice`.")
            expl = shap.Explanation(
                values=sv[indice],
                base_values=base_val,
                data=X_sample.iloc[indice].values,
                feature_names=list(X_sample.columns),
            )
            shap.plots.waterfall(expl, show=False)
        elif tipo == "dependence":
            if variable is None:
                raise ValueError("Para tipo='dependence' debes proporcionar `variable`.")
            shap.dependence_plot(
                variable, sv, X_sample,
                interaction_index=color_variable,
                show=False,
            )
        else:
            raise ValueError(f"tipo='{tipo}' no reconocido. Usa: summary, bar, waterfall, dependence.")

        if ruta_guardado:
            Path(ruta_guardado).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(ruta_guardado, bbox_inches="tight", dpi=150)
            print(f"[CatBoostTrainer] Figura guardada en {ruta_guardado}")

        plt.show()

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def serializar(self, ruta: str | Path) -> None:
        """Serializa el modelo entrenado a disco con joblib.

        Guarda el objeto CatBoostTrainer completo (modelo + metadatos).
        Para cargar: CatBoostTrainer.cargar(ruta).

        Parámetros
        ----------
        ruta : Ruta del archivo de destino (p. ej. 'models/catboost_model.pkl').
        """
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado. No hay nada que serializar.")

        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ruta)
        print(f"[CatBoostTrainer] Modelo guardado en {ruta}")

    @classmethod
    def cargar(cls, ruta: str | Path) -> "CatBoostTrainer":
        """Carga un CatBoostTrainer serializado desde disco.

        Parámetros
        ----------
        ruta : Ruta del archivo .pkl generado por serializar().

        Retorna
        -------
        Instancia de CatBoostTrainer lista para predecir.
        """
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {ruta}")

        trainer = joblib.load(ruta)
        print(f"[CatBoostTrainer] Modelo cargado desde {ruta}")
        return trainer

    # ------------------------------------------------------------------
    # Acceso rápido a predicciones
    # ------------------------------------------------------------------

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve las clases predichas (int 0–4)."""
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado.")
        return self.model.predict(X).flatten().astype(int)

    def predecir_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve las probabilidades por clase (shape: n × 5)."""
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado.")
        return self.model.predict_proba(X)
