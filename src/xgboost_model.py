import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

class XGBoostDataPrep:
    """
    Módulo auxiliar para el preprocesamiento y utilidades del modelo XGBoost.
    Maneja la ingeniería de características, codificación y partición de datos.
    """
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
        
    def preprocess(self):
        """Aplica las transformaciones básicas y devuelve el DataFrame listo."""
        print("Iniciando preprocesamiento de datos para XGBoost...")
        
        # 1. Ingeniería de features
        # Reducir el sesgo de 'Sembrada' usando log1p
        self.df['log_sembrada'] = np.log1p(self.df['Sembrada'])
        
        # Interacción entre modalidad y ciclo
        self.df['interaccion_mod_ciclo'] = self.df['Idmodalidad'] * self.df['Idciclo']
        
        # 2. Codificar variable objetivo
        self.df['target'] = self.df['nivel_riesgo'].map(self.riesgo_map)
        
        # 3. Label Encoding de variables categóricas (que originalmente son numéricas o texto)
        categoricas_a_codificar = ['Idestado', 'Idmunicipio', 'Idcultivo']
        self.encoders = {}
        for col in categoricas_a_codificar:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
            self.encoders[col] = le
            
        print("Preprocesamiento completado.")
        return self.df
        
    def temporal_split(self):
        """Realiza el split temporal de los datos según los años especificados en el plan."""
        if 'target' not in self.df.columns:
            self.preprocess()
            
        # Variables predictoras (features)
        features = [
            'Idestado_encoded', 
            'Idmunicipio_encoded', 
            'Idciclo', 
            'Idmodalidad', 
            'Idcultivo_encoded', 
            'log_sembrada',
            'interaccion_mod_ciclo'
        ]
        
        print(f"Features utilizadas ({len(features)}): {features}")
        
        # Split temporal
        train_mask = self.df['Anio'] <= 2020
        val_mask = (self.df['Anio'] >= 2021) & (self.df['Anio'] <= 2022)
        test_mask = self.df['Anio'] >= 2023
        
        X_train = self.df.loc[train_mask, features]
        y_train = self.df.loc[train_mask, 'target']
        
        X_val = self.df.loc[val_mask, features]
        y_val = self.df.loc[val_mask, 'target']
        
        X_test = self.df.loc[test_mask, features]
        y_test = self.df.loc[test_mask, 'target']
        
        print(f"Entrenamiento (2010-2020): {len(X_train)} registros ({len(X_train)/len(self.df)*100:.1f}%)")
        print(f"Validación (2021-2022): {len(X_val)} registros ({len(X_val)/len(self.df)*100:.1f}%)")
        print(f"Test (2023-2024): {len(X_test)} registros ({len(X_test)/len(self.df)*100:.1f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
        
    @staticmethod
    def get_sample_weights(y_train):
        """Calcula los pesos por clase para compensar el desbalance."""
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights_dict = dict(zip(classes, weights))
        
        print("Pesos de clase calculados (balanced):")
        for cls, weight in class_weights_dict.items():
            print(f"  Clase {cls}: {weight:.4f}")
            
        sample_weights = y_train.map(class_weights_dict)
        return sample_weights
