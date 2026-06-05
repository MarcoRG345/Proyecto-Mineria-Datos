import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

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
        
        # 2. Codificar variable objetivo
        # Normalizar posibles variaciones de encoding en el texto
        nivel = self.df['nivel_riesgo'].astype(str).str.strip()
        self.df['target'] = nivel.map(self.riesgo_map)
        
        # Fallback: si hay NaN en target, intentar matching parcial
        if self.df['target'].isna().any():
            mask_na = self.df['target'].isna()
            for idx in self.df[mask_na].index:
                val = str(self.df.loc[idx, 'nivel_riesgo']).strip()
                for key in self.riesgo_map:
                    if key.lower() in val.lower() or val.lower() in key.lower():
                        self.df.loc[idx, 'target'] = self.riesgo_map[key]
                        break
        
        # Si aún hay NaN, asignar clase 0 (sin siniestro) como default
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
