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
