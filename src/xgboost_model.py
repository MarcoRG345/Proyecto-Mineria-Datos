import unicodedata
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def _normalizar(texto: str) -> str:
    """Normaliza una cadena para comparación robusta a acentos y mojibake.

    El CSV `siap_procesado.csv` tiene codificación MIXTA: algunos acentos están
    en latin-1 y otros en UTF-8, por lo que "Pérdida crítica" puede llegar como
    'PÃ©rdida crÃ\xadtica', 'Pérdida cr\xadtica', etc. Esta función dobla los
    caracteres a ASCII (eliminando acentos y bytes basura) y deja minúsculas,
    para poder mapear por palabra clave sin depender de la codificación exacta.
    """
    s = unicodedata.normalize("NFKD", str(texto))
    s = s.encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()

class XGBoostDataPrep:
    """
    Módulo auxiliar para el preprocesamiento y utilidades del modelo XGBoost.
    Maneja la ingeniería de características, codificación y partición de datos.
    """

    # Features que usará el modelo
    FEATURES = [
        'Anio',
        'Idestado_encoded',
        'Idmunicipio_encoded',
        'Idciclo',
        'Idmodalidad',
        'Idcultivo_encoded',
        'log_sembrada',
        'interaccion_mod_ciclo'
    ]

    # Nombres legibles para usar en gráficas (feature importance, SHAP, etc.)
    FEATURES_LEGIBLES = {
        'Anio': 'Año',
        'Idestado_encoded': 'Estado',
        'Idmunicipio_encoded': 'Municipio',
        'Idciclo': 'Ciclo productivo',
        'Idmodalidad': 'Modalidad (riego/temporal)',
        'Idcultivo_encoded': 'Cultivo',
        'log_sembrada': 'log(Sup. sembrada)',
        'interaccion_mod_ciclo': 'Modalidad × Ciclo',
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # Mapeo de la variable objetivo a enteros
        self.riesgo_map = {
            "Sin siniestro": 0,
            "Riesgo bajo": 1,
            "Riesgo medio": 2,
            "Riesgo alto": 3,
            "Pérdida crítica": 4
        }
        # Mapeo inverso para reportes
        self.riesgo_map_inv = {v: k for k, v in self.riesgo_map.items()}
        self.encoders = {}
        
    def preprocess(self):
        """Aplica las transformaciones básicas y devuelve el DataFrame listo."""
        print("Iniciando preprocesamiento de datos para XGBoost...")
        
        # 1. Ingeniería de features
        # Reducir el sesgo de 'Sembrada' usando log1p
        self.df['log_sembrada'] = np.log1p(self.df['Sembrada'])
        
        # Interacción entre modalidad y ciclo
        self.df['interaccion_mod_ciclo'] = self.df['Idmodalidad'] * self.df['Idciclo']
        
        # 2. Codificar variable objetivo de forma ROBUSTA a la codificación.
        # No se puede usar `.map(riesgo_map)` directo porque el CSV tiene
        # codificación mixta y "Pérdida crítica" no coincide exactamente con la
        # clave (la 'í' llega como mojibake). Mapeamos por PALABRA CLAVE sobre
        # el texto normalizado a ASCII (ver `_normalizar`). Así las 5 clases se
        # asignan correctamente sin importar acentos ni bytes corruptos.
        def _a_clase(valor):
            s = _normalizar(valor)
            if s in ('', 'nan', 'none'):
                return np.nan          # verdadero faltante
            if 'sin' in s:
                return 0
            if 'bajo' in s:
                return 1
            if 'medio' in s:
                return 2
            if 'alto' in s:
                return 3
            # Vocabulario cerrado de 5 categorías: cualquier valor restante
            # corresponde a "Pérdida crítica" (que por el mojibake del CSV se
            # normaliza como 'pardida cratica' y no contiene palabras claras).
            return 4

        self.df['target'] = self.df['nivel_riesgo'].map(_a_clase)

        n_nan = int(self.df['target'].isna().sum())
        if n_nan:
            print(f"  ADVERTENCIA: {n_nan} registros con nivel_riesgo no reconocido "
                  f"-> asignados a clase 0. Valores: "
                  f"{self.df.loc[self.df['target'].isna(), 'nivel_riesgo'].unique()[:5]}")
        self.df['target'] = self.df['target'].fillna(0).astype(int)
        
        # 3. Label Encoding de variables categóricas
        categoricas_a_codificar = ['Idestado', 'Idmunicipio', 'Idcultivo']
        for col in categoricas_a_codificar:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
            self.encoders[col] = le
            
        print("Preprocesamiento completado.")
        print(f"  Filas: {len(self.df):,}")
        print(f"  Distribución target: {dict(self.df['target'].value_counts().sort_index())}")
        return self.df
        
    def temporal_split(self):
        """Realiza el split temporal de los datos según los años especificados en el plan."""
        if 'target' not in self.df.columns:
            self.preprocess()
            
        features = self.FEATURES
        print(f"\nFeatures utilizadas ({len(features)}): {features}")
        
        # Split temporal
        train_mask = self.df['Anio'] <= 2020
        val_mask = (self.df['Anio'] >= 2021) & (self.df['Anio'] <= 2022)
        test_mask = self.df['Anio'] >= 2023
        
        X_train = self.df.loc[train_mask, features].copy()
        y_train = self.df.loc[train_mask, 'target'].copy()
        
        X_val = self.df.loc[val_mask, features].copy()
        y_val = self.df.loc[val_mask, 'target'].copy()
        
        X_test = self.df.loc[test_mask, features].copy()
        y_test = self.df.loc[test_mask, 'target'].copy()
        
        print(f"Entrenamiento (2010-2020): {len(X_train):,} registros ({len(X_train)/len(self.df)*100:.1f}%)")
        print(f"Validación    (2021-2022): {len(X_val):,} registros ({len(X_val)/len(self.df)*100:.1f}%)")
        print(f"Test          (2023-2024): {len(X_test):,} registros ({len(X_test)/len(self.df)*100:.1f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
        
    @staticmethod
    def get_sample_weights(y):
        """Calcula los pesos por clase para compensar el desbalance."""
        classes = np.sort(np.unique(y))
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weights_dict = dict(zip(classes, weights))
        
        print("\nPesos de clase calculados (balanced):")
        for cls, weight in class_weights_dict.items():
            print(f"  Clase {cls}: {weight:.4f}")
            
        sample_weights = y.map(class_weights_dict)
        return sample_weights
