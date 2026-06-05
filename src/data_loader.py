"""
Abstrae el acceso al dataset.
"""

import pandas as pd
from pathlib import Path


class DataLoader:
    """
    Repositorio de datos agrícolas SIAP (nivel municipal, 2010-2024).

    Parámetros
    ----------
    ruta : str | Path
    encoding : str
        Codificación del archivo. Por defecto 'latin-1'.
    sep : str
        Separador del CSV. Por defecto ','.
    """

    def __init__(self, ruta: str | Path, encoding: str = "latin-1", sep: str = ","):
        self.ruta = Path(ruta)
        self.encoding = encoding
        self.sep = sep
        self._df: pd.DataFrame | None = None


    def cargar(self) -> pd.DataFrame:
        """
        Carga el CSV en memoria, aplica tipos correctos y devuelve el DataFrame.
        Guarda una copia interna para llamadas subsecuentes.
        """
        if self._df is not None:
            return self._df

        if not self.ruta.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo: {self.ruta}\n"
            )

        print(f"[DataLoader] Cargando {self.ruta.name} ...")
        df = pd.read_csv(self.ruta, encoding=self.encoding, sep=self.sep, low_memory=False)

        # El archivo trae un BOM (Byte Order Mark) al inicio. Como se lee con
        # 'latin-1', el BOM aparece como los caracteres 'ï»¿' pegados al nombre
        # de la primera columna (p. ej. 'ï»¿Anio' en vez de 'Anio'). Si se lee
        # con utf-8-sig aparecería como '﻿'. Limpiamos ambos casos para que
        # las columnas tengan nombres correctos sin importar la codificación.
        df.columns = [
            str(c).replace("﻿", "").replace("ï»¿", "").strip()
            for c in df.columns
        ]

        self._df = df
        print(f"[DataLoader] {df.shape[0]} filas , {df.shape[1]} columnas cargadas.")
        return self._df


    @staticmethod
    def aplicar_tipos(
        df: pd.DataFrame,
        numericas: list[str],
        categoricas: list[str],
    ) -> pd.DataFrame:
        """
        Convierte columnas del DataFrame al tipo correcto.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame al que se le aplicarán los tipos.
        numericas : list[str]
            Nombres de columnas que deben ser float64.
        categoricas : list[str]
            Nombres de columnas que deben ser category.

        Retorna
        -------
        pd.DataFrame con los tipos aplicados (modifica en lugar).
        """
        for col in numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in categoricas:
            if col in df.columns:
                df[col] = df[col].astype("category")

        if "Anio" in df.columns:
            df["Anio"] = pd.to_numeric(df["Anio"], errors="coerce").astype("Int64")

        return df