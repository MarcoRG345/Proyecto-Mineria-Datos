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
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from IPython.display import display, Markdown

from src.data_loader import DataLoader

warnings.filterwarnings("ignore")

# Estilo global
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

    # Niveles ordinales de riesgo agrícola
    NIVELES_RIESGO: list[str] = [
        "Sin siniestro",
        "Riesgo bajo",
        "Riesgo medio",
        "Riesgo alto",
        "Pérdida crítica",
    ]

    # Colores asociados a cada nivel de riesgo
    COLORES_RIESGO: dict[str, str] = {
        "Sin siniestro":   "#2ecc71",
        "Riesgo bajo":     "#f1c40f",
        "Riesgo medio":    "#e67e22",
        "Riesgo alto":     "#e74c3c",
        "Pérdida crítica": "#8e44ad",
    }

    def __init__(self, df: pd.DataFrame, guardar_figuras: bool = False) -> None:
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

        La proporción se recorta al intervalo [0, 1] para evitar valores
        imposibles debidos a inconsistencias en el reporte de hectáreas.
        Los registros donde Sembrada == 0 reciben el valor 0.
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

    def discretizar_proporcion_siniestro(self) -> pd.DataFrame:
        """
        Crea la variable categórica ordinal nivel_riesgo a partir de
        proporcion_siniestro mediante discretización por intervalos fijos.

        Criterio de cortes:
        - 0 : Sin siniestro
        - (0.00, 0.10] : Riesgo bajo
        - (0.10, 0.40] : Riesgo medio
        - (0.40, 0.75] : Riesgo alto
        - (0.75, 1.00] : Pérdida crítica
        """
        df = self.df

        if "proporcion_siniestro" not in df.columns:
            print("Primero ejecuta nueva_variable_proporcion_siniestro().")
            return df

        self.df["nivel_riesgo"] = pd.Categorical(
            self.df["proporcion_siniestro"].map(self._clasificar_riesgo),
            categories=self.NIVELES_RIESGO,
            ordered=True,
        )

        df = self.df

        # Distribución de frecuencias y porcentajes
        conteo = df["nivel_riesgo"].value_counts().reindex(self.NIVELES_RIESGO)
        pct = (conteo / len(df) * 100).round(2)
        tabla = pd.DataFrame({"Registros": conteo, "% del total": pct})
        display(tabla)
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(
            conteo.index,
            conteo.values,
            color=[self.COLORES_RIESGO[c] for c in conteo.index],
            edgecolor="white",
            width=0.6,
        )
        for bar, val, p in zip(bars, conteo.values, pct.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:,}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_title("Distribución de nivel de riesgo agrícola\n(discretización de proporcion_siniestro)",
                     fontsize=13)
        ax.set_xlabel("Nivel de riesgo")
        ax.set_ylabel("Número de registros")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        self._guardar_o_mostrar("distribucion_nivel_riesgo.png")

        return self.df


    # 2. CALIDAD DE DATOS Y PROCESAMIENTO DE DATOS

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
        Media, mediana, desv. estándar, cuartiles y moda para variables.
        """
        df = self.df
        cols = [c for c in self.NUMERICAS if c in df.columns]

        desc = self.df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
        desc["rango"] = desc["max"] - desc["min"]
        display(desc.round(4))

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
                print("Top 3 frecuencias:")
                for cat, count in frecuencias.items():
                    print(f"- {cat}: {count:,} registros")


    # 4. DETECCIÓN DE OUTLIERS

    def detectar_outliers(self, columnas: list[str] | None = None) -> pd.DataFrame:
        """
        Usa regla IQR para identificar outliers en variables numéricas.
        Genera boxplots comparativos por modalidad hídrica.
        """
        df = self.df
        cols = [c for c in self.NUMERICAS if c in df.columns]

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
                "Límite inferior": lim_inf, "Límite superior": lim_sup,
                "Outliers": n_outliers, "% Outliers": round(pct, 2),
            })

        tabla = pd.DataFrame(resultados).set_index("Variable")
        display(tabla)

        # Boxplots
        for col in cols:
            fig, ax = plt.subplots(figsize=(6, 5))
            
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
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            plt.tight_layout()

            if self.guardar_figuras:
                Path("figuras").mkdir(exist_ok=True)
                plt.savefig(f"figuras/outliers_boxplot_{col}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        return tabla


    # 5. DISTRIBUCIONES

    def visualizar_distribuciones(self, columnas: list[str] | None = None) -> None:
        """
        Histogramas para variables numéricas relevantes.
        """
        cols = [c for c in self.NUMERICAS if c in self.df.columns]
        
        if columnas is not None:
            cols = columnas

        n = len(cols)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 9))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            data = self.df[col].dropna()
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


    # 6. CORRELACIONES

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

        return corr


    # 7. VARIABLES CATEGÓRICAS

    def analisis_categoricas(self, top_n: int = 15) -> None:
        """
        Barras de frecuencia para variables categóricas.
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


    # 8. SERIE DE TIEMPO Y ANÁLISIS GEOGRÁFICO

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

    def top_estados_cosechada(self, top_n: int = 10) -> pd.DataFrame:
        """
        Ranking de estados con mayor área cosechada acumulada (millones de ha).

        Parámetros
        ----------
        top_n : int
            Número de estados a mostrar (default 10).
        """
        df = self.df

        if "Nomestado" not in df.columns or "Cosechada" not in df.columns:
            print("Se requieren las columnas 'Nomestado' y 'Cosechada'.")
            return pd.DataFrame()

        # Total cosechado por estado
        total = (
            df.groupby("Nomestado", observed=True)["Cosechada"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .div(1e6)
        )
        estados_top = total.index.tolist()

        # Desagregación por modalidad
        resultado = pd.DataFrame({"Total": total})

        if "Nommodalidad" in df.columns:
            sub = df[df["Nomestado"].isin(estados_top)]
            modal = (
                sub.groupby(["Nomestado", "Nommodalidad"], observed=True)["Cosechada"]
                .sum()
                .div(1e6)
                .unstack(fill_value=0)
            )
            # Asegurar que existan ambas columnas aunque no aparezcan en el subconjunto
            for mod in ["Riego", "Temporal"]:
                if mod not in modal.columns:
                    modal[mod] = 0.0
            modal = modal.reindex(estados_top)[["Riego", "Temporal"]]
            resultado = resultado.join(modal, how="left").fillna(0)

            # Barras apiladas por modalidad
            fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.55)))
            modal_plot = modal.sort_values("Total" if "Total" in modal.columns else "Riego", ascending=True)
            # ordenar igual que el ranking total
            modal_plot = modal.reindex(total.sort_values().index)

            modal_plot[["Temporal", "Riego"]].plot(
                kind="barh",
                stacked=True,
                ax=ax,
                color=[COLORES_MODALIDAD["Temporal"], COLORES_MODALIDAD["Riego"]],
                edgecolor="white",
                width=0.65,
            )
            ax.set_title(
                f"Área cosechada por modalidad hídrica – Top {top_n} estados",
                fontsize=12,
            )
            ax.set_xlabel("Millones de hectáreas")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.1f}"))
            ax.legend(title="Modalidad", loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            self._guardar_o_mostrar("top_estados_cosechada_modalidad.png")

        return resultado.reset_index()

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

    def top_cultivos_por_estado(
        self,
        top_estados: int = 5,
        top_cultivos: int = 5,
    ) -> pd.DataFrame:
        """
        Genera un heatmap y un gráfico de barras agrupadas con los cultivos
        más frecuentes dentro de los estados más frecuentes del dataset.

        Parámetros
        ----------
        top_estados  : int
            Número de estados con más registros a considerar (default 5).
        top_cultivos : int
            Número de cultivos más frecuentes a mostrar por estado (default 5).
        """
        df = self.df

        if "Nomestado" not in df.columns or "Nomcultivo" not in df.columns:
            print("Se requieren las columnas 'Nomestado' y 'Nomcultivo'.")
            return pd.DataFrame()

        # Seleccionar top estados y top cultivos
        estados = (
            df["Nomestado"].value_counts()
            .head(top_estados)
            .index.tolist()
        )
        cultivos = (
            df["Nomcultivo"].value_counts()
            .head(top_cultivos)
            .index.tolist()
        )

        sub = df[df["Nomestado"].isin(estados) & df["Nomcultivo"].isin(cultivos)]

        pivot = (
            sub.groupby(["Nomestado", "Nomcultivo"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(index=estados, columns=cultivos, fill_value=0)
        )

        # Heatmap
        fig, ax = plt.subplots(figsize=(max(8, top_cultivos * 1.8), max(4, top_estados * 1.1)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=",",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Número de registros"},
        )
        ax.set_title(
            f"Top {top_cultivos} cultivos en los Top {top_estados} estados\n(número de registros)",
            fontsize=13,
        )
        ax.set_xlabel("Cultivo")
        ax.set_ylabel("Estado")
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        self._guardar_o_mostrar("heatmap_cultivos_estados.png")

        # Barras agrupadas
        pivot_plot = pivot.T          # cultivos en eje X, estados como series
        x = np.arange(len(cultivos))
        ancho = 0.15
        paleta = sns.color_palette("tab10", n_colors=top_estados)

        fig, ax = plt.subplots(figsize=(max(10, top_cultivos * 2), 6))
        for i, (estado, color) in enumerate(zip(estados, paleta)):
            offset = (i - top_estados / 2 + 0.5) * ancho
            vals = pivot_plot[estado].values if estado in pivot_plot.columns else np.zeros(len(cultivos))
            bars = ax.bar(x + offset, vals, width=ancho, label=estado, color=color, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(cultivos, rotation=35, ha="right", fontsize=9)
        ax.set_title(
            f"Registros por cultivo en los Top {top_estados} estados",
            fontsize=13,
        )
        ax.set_ylabel("Número de registros")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.legend(title="Estado", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        self._guardar_o_mostrar("barras_cultivos_estados.png")

        return pivot


    # Helpers

    def exportar_dataset_limpio(
        self,
        ruta: str | None = None,
        formato: str = "csv",
    ) -> str:
        """
        Exporta el DataFrame procesado.

        Parámetros
        ----------
        ruta : str | None
            Ruta completa del archivo de destino.
        formato : str
        """
        from pathlib import Path

        formato = formato.lower().strip()
        if formato not in ("csv", "parquet"):
            raise ValueError("El formato debe ser 'csv' o 'parquet'.")

        ruta_path = (
            Path("../data/processed") / f"siap_procesado.{formato}"
            if ruta is None
            else Path(ruta)
        )
        ruta_path.parent.mkdir(parents=True, exist_ok=True)

        df_export = self.df.copy()
        # Convertir columnas Categorical a str para compatibilidad CSV/Parquet
        for col in df_export.select_dtypes(include="category").columns:
            df_export[col] = df_export[col].astype(str)

        if formato == "csv":
            df_export.to_csv(ruta_path, index=False, encoding="utf-8-sig")
        else:
            df_export.to_parquet(ruta_path, index=False)

        vars_derivadas = [c for c in ["proporcion_siniestro", "nivel_riesgo"] if c in df_export.columns]
        print("Dataset exportado correctamente.")
        print(f"  Ruta    : {ruta_path.resolve()}")
        print(f"  Formato : {formato.upper()}")
        print(f"  Filas   : {len(df_export):,}")
        print(f"  Columnas: {len(df_export.columns):,}")
        if vars_derivadas:
            print(f"  Variables derivadas incluidas: {', '.join(vars_derivadas)}")

        return str(ruta_path.resolve())

    @staticmethod
    def _clasificar_riesgo(p: float) -> str:
        """Clasifica una proporción de siniestro en su nivel de riesgo textual."""
        if p == 0:
            return "Sin siniestro"
        elif p <= 0.10:
            return "Riesgo bajo"
        elif p <= 0.40:
            return "Riesgo medio"
        elif p <= 0.75:
            return "Riesgo alto"
        return "Pérdida crítica"

    def _guardar_o_mostrar(self, nombre: str) -> None:
        """Guarda la figura activa en disco (si guardar_figuras) y la muestra."""
        if self.guardar_figuras:
            Path("figuras").mkdir(exist_ok=True)
            plt.savefig(f"figuras/{nombre}", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
