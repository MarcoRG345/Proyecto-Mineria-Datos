"""
Encapsula toda la lógica del EDA requerida por el proyecto:
  1. Descripción general
  2. Calidad de datos
  3. Estadísticas descriptivas
  4. Detección de outliers
  5. Distribuciones
  6. Correlaciones
  7. Variables categóricas
  8. Serie de tiempo y análisis geográfico
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from IPython.display import display, Markdown
from src.data_loader import DataLoader

warnings.filterwarnings("ignore")

# ── Estilo global ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLORES_MODALIDAD = {"Temporal": "#e07b39", "Riego": "#3b7fc4"}


class EDAAnalyzer:
    """
    Analizador exploratorio para el dataset agrícola SIAP.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame ya cargado por DataLoader.
    guardar_figuras : bool
        Si True, guarda cada gráfica en la carpeta `figuras/`.
    """

    NUMERICAS = [
        "Sembrada", "Cosechada", "Siniestrada",
        "Volumenproduccion", "Rendimiento", "Precio", "Valorproduccion",
    ]
    CATEGORICAS = [
        "Nomestado", "Nomcicloproductivo", "Nommodalidad", "Nomcultivo",
    ]

    def __init__(self, df: pd.DataFrame, guardar_figuras: bool = False):
        self.df = DataLoader.aplicar_tipos(
            df.copy(),
            numericas=self.NUMERICAS,
            categoricas=self.CATEGORICAS,
        )
        self.guardar_figuras = guardar_figuras

    # 1. DESCRIPCIÓN GENERAL

    def descripcion_general(self) -> pd.DataFrame:
        """
        Imprime dimensiones, rango temporal y tipos de datos.
        Retorna un DataFrame resumen de tipos.
        """
        df = self.df

        print("DESCRIPCIÓN DEL DATASET\n")
        print(f"Filas    : {df.shape[0]}")
        print(f"Columnas : {df.shape[1]}")
        if "Anio" in df.columns:
            print(f"Periodo  : {df['Anio'].min()} – {df['Anio'].max()}\n")

        resumen = pd.DataFrame({
            "Tipo de Dato": df.dtypes,
            "No Nulos": df.notna().sum(),
            "Únicos": df.nunique(),
        })
        display(resumen)
        return resumen

    # PROPORCIÓN DE SINIESTRO

    def nueva_variable_proporcion_siniestro(self) -> pd.DataFrame:
        """
        Crea la variable derivada proporcion_siniestro.

        Definición
        ----------
        proporcion_siniestro = Siniestrada / Sembrada

        La proporción se recorta al intervalo [0, 1] para evitar valores
        imposibles debidos a inconsistencias en el reporte de hectáreas.
        Los registros donde Sembrada == 0 reciben el valor 0.

        La nueva columna se agrega directamente al DataFrame interno
        del analizador (self.df) y se muestra un resumen estadístico.

        Retorna
        -------
        pd.DataFrame
            Estadísticas descriptivas de la nueva variable.
        """
        df = self.df

        if "Sembrada" not in df.columns or "Siniestrada" not in df.columns:
            print("Las columnas 'Sembrada' y/o 'Siniestrada' no están disponibles.")
            return pd.DataFrame()

        df["proporcion_siniestro"] = (
            df["Siniestrada"]
            .div(df["Sembrada"].replace(0, np.nan))
            .clip(0, 1)
            .fillna(0)
        )

        self.df = df

        desc_proporcion = pd.DataFrame({
            "Tipo de Dato": [df['proporcion_siniestro'].dtype],
            "No Nulos": [df['proporcion_siniestro'].notna().sum()],
            "Únicos": [df['proporcion_siniestro'].nunique()],
        })
        
        display(desc_proporcion)

        return df


    # 2. CALIDAD DE DATOS

    def calidad_datos(self) -> dict:
        """
        Reporta valores faltantes, duplicados.
        """
        df = self.df
        nulos = df.isna().sum()
        nulos_pct = (df.isna().mean() * 100).round(2)
        duplicados = df.duplicated().sum()

        # Nulos
        tabla_nulos = pd.DataFrame({"Nulos": nulos, "% Nulos": nulos_pct})
        tabla_nulos = tabla_nulos[tabla_nulos["Nulos"] > 0].sort_values("% Nulos", ascending=False)
        display(tabla_nulos)

        # Duplicados
        print(f"\nRegistros duplicados: {duplicados:,}")

        return {
            "nulos_por_columna": nulos.to_dict(),
            "pct_nulos": nulos_pct.to_dict(),
            "duplicados": int(duplicados),
        }

    def analisis_nulos(self) -> None:
        """
        Analiza patrones en los datos faltantes.
        Co-ocurrencia (correlación de nulos)
        """
        df = self.df
        cols_con_nulos = df.columns[df.isna().any()].tolist()
        
        if not cols_con_nulos:
            print("No hay valores nulos en el dataset.")
            return
        
        # Correlación de nulos
        if len(cols_con_nulos) > 1:
            corr_nulos = df[cols_con_nulos].isna().corr()
            
            # Graficar heatmap
            fig, ax = plt.subplots(figsize=(max(6, len(cols_con_nulos)*1.2), max(5, len(cols_con_nulos)*1.2)))
            sns.heatmap(corr_nulos, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
            ax.set_title("Correlación de Nulos", fontsize=13)
            plt.tight_layout()
            self._guardar_o_mostrar("correlacion_nulos.png")

    def imputar_nulos_estructurales(self) -> pd.DataFrame:
        """
        Imputa con 0 los nulos en 'Precio', 'Rendimiento' y 'Valorproduccion' 
        cuando el área 'Cosechada' es 0, ya que estructuralmente no hubo producción.
        """
        df = self.df
        
        if "Cosechada" in df.columns:
            mask_cosechada_cero = df["Cosechada"] == 0
            
            for col in ["Precio", "Rendimiento", "Valorproduccion", "Volumenproduccion"]:
                if col in df.columns:
                    df.loc[mask_cosechada_cero & df[col].isna(), col] = 0
            
            self.df = df
            
        return df

    def eliminar_duplicados(self) -> pd.DataFrame:
        """
        Elimina los registros duplicados exactos del dataset y actualiza el estado interno.
        """
        df = self.df
        duplicados = df.duplicated().sum()
        
        if duplicados > 0:
            df = df.drop_duplicates()
            self.df = df
            print(f"Se eliminaron {duplicados:,} registros duplicados.")
            print(f"El dataset ahora tiene {len(df):,} filas.")
        else:
            print("No se encontraron registros duplicados.")
            
        return df


    # 3. ESTADÍSTICAS DESCRIPTIVAS

    def estadisticas_descriptivas_numericas(self) -> pd.DataFrame:
        """
        Media, mediana, desv. estándar, cuartiles y moda para variables clave.
        """
        df = self.df
        cols = [c for c in self.NUMERICAS if c in df.columns]

        desc = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
        desc["rango"] = desc["max"] - desc["min"]
        display(desc.round(2))

        return desc

    def estadisticas_descriptivas_categoricas(self) -> None:
        """
        Cardinalidad, moda y principales frecuencias para variables categóricas.
        """
        df = self.df

        for col in self.CATEGORICAS:
            if col in df.columns:
                moda = df[col].mode()[0] if not df[col].mode().empty else "N/A"
                n_unicos = df[col].nunique()
                print(f"\n{col}: {n_unicos} categorías. Moda: '{moda}'")
                
                # calculamos las frecuencias
                frecuencias = df[col].value_counts().head(3)
                print("Top 3 frecuencias (conteos):")
                for cat, count in frecuencias.items():
                    print(f"- {cat}: {count:,} registros")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. DETECCIÓN DE OUTLIERS
    # ──────────────────────────────────────────────────────────────────────────

    def detectar_outliers(self, columnas: list[str] | None = None) -> pd.DataFrame:
        """
        Usa regla IQR para identificar outliers en variables numéricas.
        Genera boxplots comparativos por modalidad hídrica.
        """
        df = self.df
        cols = columnas or [c for c in ["Sembrada", "Siniestrada", "Rendimiento"] if c in df.columns]

        resultados = []
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lim_inf = q1 - 1.5 * iqr
            lim_sup = q3 + 1.5 * iqr
            n_outliers = ((df[col] < lim_inf) | (df[col] > lim_sup)).sum()
            pct = n_outliers / len(df) * 100
            resultados.append({
                "Variable": col, "Q1": q1, "Q3": q3, "IQR": iqr,
                "Lím. inf.": lim_inf, "Lím. sup.": lim_sup,
                "Outliers": n_outliers, "% Outliers": round(pct, 2),
            })

        tabla = pd.DataFrame(resultados).set_index("Variable")
        print("=" * 60)
        print("  OUTLIERS (Regla IQR)")
        print("=" * 60)
        print(tabla.to_string())
        print("\n  Decisión: los outliers extremos en Siniestrada y Rendimiento")
        print("  corresponden a eventos reales (desastres climáticos), por lo que")
        print("  se conservan. Se aplicará transformación log1p antes del modelado.")

        # Boxplots
        fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 5))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            if "Nommodalidad" in df.columns:
                data = [
                    df.loc[df["Nommodalidad"].astype(str).str.contains(k, case=False), col].dropna()
                    for k in ["Temporal", "Riego"]
                ]
                ax.boxplot(data, labels=["Temporal", "Riego"], patch_artist=True,
                           boxprops=dict(facecolor="#e0e8f0"))
            else:
                ax.boxplot(df[col].dropna(), patch_artist=True)
            ax.set_title(f"Outliers: {col}")
            ax.set_ylabel("Hectáreas")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        plt.suptitle("Distribución y outliers por modalidad hídrica", fontsize=13, y=1.01)
        plt.tight_layout()
        self._guardar_o_mostrar("outliers_boxplot.png")

        return tabla

    # ──────────────────────────────────────────────────────────────────────────
    # 5. DISTRIBUCIONES
    # ──────────────────────────────────────────────────────────────────────────

    def visualizar_distribuciones(self, columnas: list[str] | None = None) -> None:
        """
        Histogramas con KDE para variables numéricas relevantes.
        """
        df = self.df
        cols = columnas or [c for c in self.NUMERICAS if c in df.columns]
        n = len(cols)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 9))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            data = df[col].dropna()
            # Log1p para variables con sesgo positivo severo
            if data.skew() > 2:
                data_plot = np.log1p(data)
                xlabel = f"log1p({col})"
            else:
                data_plot = data
                xlabel = col
            axes[i].hist(data_plot, bins=60, color="#5a9fd4", edgecolor="white", density=True, alpha=0.7)
            axes[i].set_title(col, fontsize=11)
            axes[i].set_xlabel(xlabel, fontsize=9)
            axes[i].set_ylabel("Densidad")

        # Ocultar ejes sobrantes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Distribuciones de variables numéricas\n(escala log1p cuando sesgo > 2)",
                     fontsize=13)
        plt.tight_layout()
        self._guardar_o_mostrar("distribuciones.png")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. CORRELACIONES
    # ──────────────────────────────────────────────────────────────────────────

    def matriz_correlaciones(self, metodo: str = "spearman") -> pd.DataFrame:
        """
        Heatmap de correlaciones. Usa Spearman por defecto (robusto a outliers).
        """
        df = self.df
        cols = [c for c in self.NUMERICAS if c in df.columns]
        corr = df[cols].corr(method=metodo)

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax,
        )
        ax.set_title(f"Matriz de correlaciones ({metodo.capitalize()})", fontsize=13)
        plt.tight_layout()
        self._guardar_o_mostrar("correlaciones.png")

        print(f"\n  Correlaciones notables con Siniestrada ({metodo}):")
        for col in cols:
            if col != "Siniestrada":
                val = corr.loc["Siniestrada", col]
                if abs(val) > 0.1:
                    print(f"  · {col}: {val:.3f}")

        return corr

    # ──────────────────────────────────────────────────────────────────────────
    # 7. VARIABLES CATEGÓRICAS
    # ──────────────────────────────────────────────────────────────────────────

    def analisis_categoricas(self, top_n: int = 15) -> None:
        """
        Barras de frecuencia para variables categóricas.
        Incluye análisis de balance de la variable objetivo (ratio siniestro).
        """
        df = self.df
        cols = [c for c in self.CATEGORICAS if c in df.columns]

        for col in cols:
            freq = df[col].value_counts().head(top_n)
            fig, ax = plt.subplots(figsize=(10, 5))
            freq.plot(kind="bar", ax=ax, color="#5a9fd4", edgecolor="white")
            ax.set_title(f"Top {top_n}: {col}", fontsize=12)
            ax.set_xlabel("")
            ax.set_ylabel("Registros")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            self._guardar_o_mostrar(f"barras_{col}.png")

        # Balance de clases (ratio siniestro)
        if "ratio_siniestro" in df.columns and "Nommodalidad" in df.columns:
            print("\n  Balance de ratio_siniestro por modalidad:")
            display(df.groupby("Nommodalidad", observed=True)["ratio_siniestro"].describe().round(4))

    # ──────────────────────────────────────────────────────────────────────────
    # 8. SERIE DE TIEMPO Y ANÁLISIS GEOGRÁFICO
    # ──────────────────────────────────────────────────────────────────────────

    def serie_temporal(self) -> pd.DataFrame:
        """
        Evolución anual del área sembrada, cosechada y siniestrada (en millones de ha).
        Desagregada por modalidad hídrica.
        """
        df = self.df
        if "Anio" not in df.columns:
            print("Columna 'Anio' no encontrada.")
            return pd.DataFrame()

        cols_area = [c for c in ["Sembrada", "Cosechada", "Siniestrada"] if c in df.columns]

        # Total nacional por año
        anual = df.groupby("Anio", observed=True)[cols_area].sum() / 1e6

        fig, ax = plt.subplots(figsize=(12, 5))
        for col in cols_area:
            ax.plot(anual.index, anual[col], marker="o", label=col)
        ax.set_title("Evolución anual: áreas sembrada, cosechada y siniestrada (millones ha)", fontsize=12)
        ax.set_xlabel("Año")
        ax.set_ylabel("Millones de hectáreas")
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.tight_layout()
        self._guardar_o_mostrar("serie_temporal_total.png")

        # Por modalidad
        if "Nommodalidad" in df.columns and "Siniestrada" in df.columns:
            modal = (df.groupby(["Anio", "Nommodalidad"], observed=True)["Siniestrada"]
                       .sum()
                       .div(1e6)
                       .reset_index())
            fig, ax = plt.subplots(figsize=(12, 5))
            for mod, grp in modal.groupby("Nommodalidad", observed=True):
                color = COLORES_MODALIDAD.get(str(mod), None)
                ax.plot(grp["Anio"], grp["Siniestrada"], marker="o", label=mod, color=color)
            ax.set_title("Área siniestrada anual por modalidad hídrica (millones ha)", fontsize=12)
            ax.set_xlabel("Año")
            ax.set_ylabel("Millones de hectáreas")
            ax.legend(title="Modalidad")
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            plt.tight_layout()
            self._guardar_o_mostrar("serie_temporal_modalidad.png")

        return anual

    def top_estados_siniestros(self, top_n: int = 15) -> pd.DataFrame:
        """
        Ranking de estados con mayor área siniestrada acumulada (2010-2024).
        """
        df = self.df
        if "Nomestado" not in df.columns or "Siniestrada" not in df.columns:
            return pd.DataFrame()

        ranking = (df.groupby("Nomestado", observed=True)["Siniestrada"]
                     .sum()
                     .sort_values(ascending=False)
                     .head(top_n)
                     .div(1e6))

        fig, ax = plt.subplots(figsize=(10, 6))
        ranking.sort_values().plot(kind="barh", ax=ax, color="#e07b39", edgecolor="white")
        ax.set_title(f"Top {top_n} estados con mayor área siniestrada (2010-2024)", fontsize=12)
        ax.set_xlabel("Millones de hectáreas")
        plt.tight_layout()
        self._guardar_o_mostrar("top_estados_siniestros.png")

        return ranking.reset_index()

    def scatter_sembrada_vs_siniestrada(self) -> None:
        """
        Scatter plot: área sembrada vs. siniestrada, coloreado por modalidad.
        """
        df = self.df
        if "Sembrada" not in df.columns or "Siniestrada" not in df.columns:
            return

        muestra = df.sample(min(50_000, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(9, 6))

        if "Nommodalidad" in muestra.columns:
            for mod, grp in muestra.groupby("Nommodalidad", observed=True):
                color = COLORES_MODALIDAD.get(str(mod), "#aaaaaa")
                ax.scatter(
                    np.log1p(grp["Sembrada"]), np.log1p(grp["Siniestrada"]),
                    alpha=0.15, s=5, color=color, label=mod,
                )
            ax.legend(title="Modalidad", markerscale=4)
        else:
            ax.scatter(np.log1p(muestra["Sembrada"]), np.log1p(muestra["Siniestrada"]),
                       alpha=0.1, s=5)

        ax.set_xlabel("log1p(Sembrada) [ha]")
        ax.set_ylabel("log1p(Siniestrada) [ha]")
        ax.set_title("Área sembrada vs. siniestrada (escala log)\nmuestra aleatoria de 50,000 registros")
        plt.tight_layout()
        self._guardar_o_mostrar("scatter_sembrada_siniestrada.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Columna auxiliar
    # ──────────────────────────────────────────────────────────────────────────

    def _preparar_columna_ratio(self) -> None:
        """Agrega ratio_siniestro = Siniestrada / Sembrada (clipeado a [0,1])."""
        if "Sembrada" in self.df.columns and "Siniestrada" in self.df.columns:
            self.df["ratio_siniestro"] = (
                self.df["Siniestrada"]
                .div(self.df["Sembrada"].replace(0, np.nan))
                .clip(0, 1)
                .fillna(0)
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _guardar_o_mostrar(self, nombre: str) -> None:
        if self.guardar_figuras:
            from pathlib import Path
            Path("figuras").mkdir(exist_ok=True)
            plt.savefig(f"figuras/{nombre}", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()