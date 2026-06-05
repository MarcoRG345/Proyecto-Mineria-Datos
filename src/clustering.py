"""
Clase y funciones para el pipeline de clustering (K-Means) de los municipios.
Implementa el patrón Pipeline para la transformación secuencial de los datos.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


NOMBRES_CLUSTERS = {
    0: "Temporal\nGran Escala",
    1: "Agricultura\nCampesina",
    2: "Riego\nComercial",
    3: "Vulnerabilidad\nCrítica"
}

NOMBRES_CLUSTERS_LARGO = {
    0: "Agricultura de Temporal de Gran Escala",
    1: "Agricultura Tradicional de Temporal y Subsistencia a Pequeña Escala",
    2: "Corredores Agrícolas Comerciales de Riego y Alta Productividad",
    3: "Zonas de Siniestralidad Crónica y Vulnerabilidad Climática Crítica"
}


class AgroClusteringPipeline:
    """
    Pipeline de Agrupamiento No Supervisado (K-Means) para identificar perfiles
    de vulnerabilidad agrícola y riesgo climático a nivel municipal en México.

    Implementa el patrón de diseño Pipeline: cada método corresponde a una etapa
    secuencial bien definida del flujo de datos. El método ejecutar() encadena
    todas las etapas automáticamente, cumpliendo con la definición formal del patrón.

    Etapas del pipeline:
      1. preprocesar_y_agregar  → Consolidación de registros a nivel municipal.
      2. escalar_datos          → Transformación logarítmica + StandardScaler.
      3. evaluar_codo_y_silueta → Selección del número óptimo de clústeres.
      4. entrenar_kmeans        → Ajuste del modelo K-Means definitivo.
      5. aplicar_pca            → Reducción de dimensionalidad para visualización.
    """

    def __init__(self, random_state: int = 42, scaler_type: str = "standard"):
        self.random_state = random_state
        self.scaler_type = scaler_type
        self.scaler = None
        self.pca = None
        self.kmeans = None
        # Variables seleccionadas para caracterizar el perfil de vulnerabilidad agrícola municipal:
        # - log_Sembrada_total: escala productiva del municipio (pequeño vs gran productor)
        # - proporcion_siniestro_total: riesgo histórico de pérdida de cultivos
        # - prop_temporal: dependencia de lluvia natural (mayor valor = mayor exposición climática)
        # - log_Rendimiento_medio: eficiencia productiva del municipio
        # - log_Precio_medio: valor comercial promedio de sus cultivos
        # - log_Diversidad_cultivos: resiliencia ecológica (monocultivo vs producción diversificada)
        self.features_cols = [
            "log_Sembrada_total",
            "proporcion_siniestro_total",
            "prop_temporal",
            "log_Rendimiento_medio",
            "log_Precio_medio",
            "log_Diversidad_cultivos"
        ]

        # Atributos de almacenamiento de datos
        self.df_mun = None    # DataFrame a nivel municipal con variables crudas y derivadas
        self.X_scaled = None  # Matriz de variables escaladas
        self.X_pca = None     # Matriz proyectada en las componentes principales

    def ejecutar(self, df_clean: pd.DataFrame, k: int = 4, n_components_pca: int = 2) -> "AgroClusteringPipeline":
        """
        Ejecuta el pipeline completo de forma secuencial y automática.
        Este método es el punto de entrada formal del patrón Pipeline:
        encadena preprocesamiento → escalado → entrenamiento → PCA
        sin que el cliente tenga que gestionar el orden de las etapas.

        Parámetros
        ----------
        df_clean      : DataFrame limpio proveniente de EDAAnalyzer.
        k             : Número de clústeres para K-Means (default 4).
        n_components_pca : Componentes para la reducción PCA (default 2).

        Retorna
        -------
        self, para permitir encadenamiento fluido si se desea.
        """
        self.preprocesar_y_agregar(df_clean)
        self.escalar_datos()
        self.entrenar_kmeans(k=k)
        self.aplicar_pca(n_components=n_components_pca)
        return self

    def preprocesar_y_agregar(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Consolida los registros de producción a nivel municipal para construir un
        perfil agrícola único por municipio, calculando métricas de escala, riesgo
        y diversidad que servirán como insumo del agrupamiento.

        Parámetros
        ----------
        df_clean : pd.DataFrame
            DataFrame de producción agrícola ya limpio.

        Retorna
        -------
        pd.DataFrame con un registro por municipio y variables agregadas.
        """
        df = df_clean.copy()

        if "proporcion_siniestro" not in df.columns:
            df["proporcion_siniestro"] = (
                df["Siniestrada"]
                .div(df["Sembrada"].replace(0, np.nan))
                .clip(0, 1)
                .fillna(0)
            )

        df["es_temporal"] = df["Nommodalidad"].astype(str).str.contains("Temporal", case=False).astype(int)
        df["area_temporal"] = df["Sembrada"] * df["es_temporal"]

        df_mun = df.groupby(["Nomestado", "Nommunicipio"], observed=True).agg(
            Sembrada_total=("Sembrada", "sum"),
            Siniestrada_total=("Siniestrada", "sum"),
            Cosechada_total=("Cosechada", "sum"),
            Volumen_total=("Volumenproduccion", "sum"),
            Valor_total=("Valorproduccion", "sum"),
            Area_temporal_total=("area_temporal", "sum"),
            Precio_medio=("Precio", "mean"),
            Rendimiento_medio=("Rendimiento", "mean"),
            Diversidad_cultivos=("Nomcultivo", "nunique")
        ).reset_index()

        df_mun = df_mun[df_mun["Sembrada_total"] >= 0.1].copy()

        df_mun["proporcion_siniestro_total"] = (
            df_mun["Siniestrada_total"]
            .div(df_mun["Sembrada_total"])
            .clip(0, 1)
        )

        df_mun["prop_temporal"] = (
            df_mun["Area_temporal_total"]
            .div(df_mun["Sembrada_total"])
            .clip(0, 1)
        )

        df_mun["Precio_medio"] = df_mun["Precio_medio"].fillna(0)
        df_mun["Rendimiento_medio"] = df_mun["Rendimiento_medio"].fillna(0)

        df_mun["log_Sembrada_total"] = np.log1p(df_mun["Sembrada_total"])
        df_mun["log_Rendimiento_medio"] = np.log1p(df_mun["Rendimiento_medio"])
        df_mun["log_Precio_medio"] = np.log1p(df_mun["Precio_medio"])
        df_mun["log_Diversidad_cultivos"] = np.log1p(df_mun["Diversidad_cultivos"])

        self.df_mun = df_mun
        print(f"{len(df_mun)} municipios procesados.")
        return df_mun

    def escalar_datos(self) -> np.ndarray:
        """
        Aplica escalamiento estándar o robusto sobre las variables seleccionadas.
        El escalamiento es obligatorio para K-Means ya que el algoritmo es sensible
        a la magnitud de las variables.
        """
        if self.df_mun is None:
            raise ValueError("Primero debes ejecutar preprocesar_y_agregar(df).")
        X = self.df_mun[self.features_cols].values
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "robust":
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaler_type debe ser 'standard' o 'robust'")
        self.X_scaled = self.scaler.fit_transform(X)
        return self.X_scaled

    def aplicar_pca(self, n_components: int = 2) -> np.ndarray:
        """
        Aplica PCA sobre las variables escaladas para reducir la dimensionalidad
        y facilitar la visualización de los grupos en 2D.

        Se recomienda usar n_components=2 para visualización y n_components mayores
        para preservar más varianza en análisis posteriores.
        """
        if self.X_scaled is None:
            raise ValueError("Primero debes ejecutar escalar_datos().")

        print(f"PCA con {n_components} componentes: ")
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.X_pca = self.pca.fit_transform(self.X_scaled)

        var_explicada = self.pca.explained_variance_ratio_
        print(f"Varianza explicada por componente: {var_explicada}")
        print(f"Varianza explicada acumulada: {sum(var_explicada):.4f}")
        return self.X_pca

    def evaluar_codo_y_silueta(self, max_k: int = 10, plot: bool = True) -> dict:
        """
        Calcula la inercia y el coeficiente de silueta para k en [2, max_k]
        para guiar la selección del número óptimo de clústeres.

        Se recomienda elegir el k donde la reducción de inercia se aplana (codo)
        y el coeficiente de silueta es máximo o cercano a su pico.
        """
        if self.X_scaled is None:
            raise ValueError("Primero debes ejecutar escalar_datos().")

        inercias = []
        siluetas = []
        rango_k = range(1, max_k + 1)

        print("Evaluando K-Means para distintos valores de k")
        for k in rango_k:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(self.X_scaled)
            inercias.append(km.inertia_)
            if k >= 2:
                score_sil = silhouette_score(self.X_scaled, labels)
                siluetas.append(score_sil)
                print(f"  k={k} | Inercia: {km.inertia_:.2f} | Silueta: {score_sil:.4f}")
            else:
                siluetas.append(None)
                print(f"  k={k} | Inercia: {km.inertia_:.2f} | Silueta: N/A")

        res = {"k": list(rango_k), "inercia": inercias, "silueta": siluetas}

        if plot:
            fig, ax1 = plt.subplots(figsize=(10, 5))

            color = "#3b7fc4"
            ax1.set_xlabel("Número de clústeres (k)")
            ax1.set_ylabel("Inercia (Suma de distancias al cuadrado)", color=color)
            line1 = ax1.plot(rango_k, inercias, marker="o", color=color, linewidth=2, label="Inercia (Codo)")
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.grid(True, linestyle="--", alpha=0.5)

            ax2 = ax1.twinx()
            color = "#2ecc71"
            ax2.set_ylabel("Coeficiente de Silueta", color=color)
            rango_k_sil = [k for k, s in zip(rango_k, siluetas) if s is not None]
            siluetas_plot = [s for s in siluetas if s is not None]
            line2 = ax2.plot(rango_k_sil, siluetas_plot, marker="s", color=color, linestyle="--", linewidth=2, label="Silueta")
            ax2.tick_params(axis="y", labelcolor=color)

            lines = line1 + line2
            labs = [l.get_label() for l in lines]
            ax1.legend(lines, labs, loc="upper right")

            plt.title("Evaluación del Número Óptimo de Clústeres (k)", fontsize=13, fontweight="bold")
            plt.tight_layout()

        return res

    def entrenar_kmeans(self, k: int) -> np.ndarray:
        """
        Entrena el modelo K-Means definitivo con el k indicado.
        Asigna las etiquetas al DataFrame municipal.
        """
        if self.X_scaled is None:
            raise ValueError("Primero debes ejecutar escalar_datos().")

        print(f"Entrenando K-Means con k={k} y random_state={self.random_state}")
        self.kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=15)
        labels = self.kmeans.fit_predict(self.X_scaled)
        self.df_mun["Cluster"] = labels
        return labels

    def obtener_resumen_clusters(self) -> pd.DataFrame:
        """
        Genera un perfil descriptivo de cada clúster usando las variables originales
        (sin transformación logarítmica) para facilitar la interpretación agronómica
        y geográfica de los grupos.
        """
        if self.kmeans is None:
            raise ValueError("El modelo K-Means no ha sido entrenado. Ejecuta entrenar_kmeans().")

        resumen = self.df_mun.groupby("Cluster").agg(
            Municipios=("Nommunicipio", "count"),
            Sembrada_media=("Sembrada_total", "mean"),
            Siniestrada_media=("Siniestrada_total", "mean"),
            Riesgo_siniestro_medio=("proporcion_siniestro_total", "mean"),
            Prop_temporal_media=("prop_temporal", "mean"),
            Rendimiento_promedio=("Rendimiento_medio", "mean"),
            Precio_promedio=("Precio_medio", "mean"),
            Diversidad_media=("Diversidad_cultivos", "mean")
        ).round(4)

        return resumen

    def graficar_clusters_pca(self, title: str = "Agrupamiento de Municipios por Vulnerabilidad Agrícola",
                          mostrar_centroides: bool = True) -> None:
        """
        Genera un scatter plot en 2D coloreado por clúster utilizando las dos primeras
        componentes principales de PCA. Si mostrar_centroides es True, se añaden los
        centroides de cada clúster proyectados en el espacio PCA.
        """
        if self.X_pca is None:
            self.aplicar_pca(n_components=2)

        if self.kmeans is None:
            raise ValueError("El modelo K-Means debe estar entrenado antes de graficar.")

        df_plot = pd.DataFrame(self.X_pca, columns=["PC1", "PC2"])
        df_plot["Cluster"] = self.df_mun["Cluster"].values
        df_plot["Municipio"] = self.df_mun["Nommunicipio"].values
        df_plot["Estado"] = self.df_mun["Nomestado"].values

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x="PC1", y="PC2",
            hue="Cluster",
            palette="viridis",
            data=df_plot,
            alpha=0.7,
            s=40,
            edgecolor="none"
        )

        if mostrar_centroides:
            centroides_escalados = self.kmeans.cluster_centers_
            centroides_pca = self.pca.transform(centroides_escalados)
            plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
                marker='o', s=200, c='red', edgecolor='white', linewidth=2,
                label='Centroides')
            for i, (x, y) in enumerate(centroides_pca):
                plt.text(x, y, f'  C{i}', fontsize=12, fontweight='bold', va='center')

        plt.title(title, fontsize=13, fontweight="bold")
        plt.xlabel(f"Componente Principal 1 ({self.pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
        plt.ylabel(f"Componente Principal 2 ({self.pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
        plt.legend(title="Clúster", loc="best")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

    def graficar_boxplot_siniestralidad(self) -> None:
        """
        Genera un boxplot de la proporción de siniestro por clúster para mostrar
        la dispersión real de cada perfil, no solo el promedio.
        Complementa la tabla de resumen con información distribucional.
        """
        if "Cluster" not in self.df_mun.columns:
            raise ValueError("Entrena el modelo primero con entrenar_kmeans().")

        df_plot = self.df_mun.copy()
        df_plot["Cluster_nombre"] = df_plot["Cluster"].map(NOMBRES_CLUSTERS)
        orden = list(NOMBRES_CLUSTERS.values())

        colores = {
            "Temporal\nGran Escala": "#2E7D32",
            "Agricultura\nCampesina": "#1976D2",
            "Riego\nComercial": "#F9A825",
            "Vulnerabilidad\nCrítica": "#C62828"
        }
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(
            data=df_plot,
            x="Cluster_nombre",
            y="proporcion_siniestro_total",
            order=orden,
            hue="Cluster_nombre",
            palette=colores,
            width=0.5,
            ax=ax,
            legend=False
        )
        ax.set_title("Distribución de Siniestralidad por Perfil de Clúster",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Perfil de Clúster")
        ax.set_ylabel("Proporción de Siniestro")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

    def analizar_distribucion_geografica(self, top_n: int = 5) -> pd.DataFrame:
        """
        Cruza los clústeres con la variable de estado para identificar
        qué entidades federativas concentran cada perfil de vulnerabilidad.
        Complementa los hallazgos geográficos del EDA.
        """
        if "Cluster" not in self.df_mun.columns:
            raise ValueError("Entrena el modelo primero con entrenar_kmeans().")

        dist = (
            self.df_mun.groupby(["Cluster", "Nomestado"])
            .size()
            .reset_index(name="Municipios")
            .sort_values(["Cluster", "Municipios"], ascending=[True, False])
        )

        top_estados = (
            dist.groupby("Cluster")
            .head(top_n)
            .reset_index(drop=True)
        )

        print("Distribución geográfica por clúster, top estados:")
        print(top_estados.to_string(index=False))
        return top_estados

    def graficar_distribucion_geografica(self, top_n: int = 5) -> None:
        """
        Genera un gráfico de barras horizontales con los estados más representados
        en cada clúster, organizado en una cuadrícula de 2x2.
        Permite validar la coherencia geográfica del agrupamiento.
        """
        if "Cluster" not in self.df_mun.columns:
            raise ValueError("Entrena el modelo primero con entrenar_kmeans().")

        dist_geo = self.analizar_distribucion_geografica(top_n=top_n)
        palette = sns.color_palette("viridis", 4)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, (cluster_id, grupo) in enumerate(dist_geo.groupby("Cluster")):
            grupo_sorted = grupo.sort_values("Municipios", ascending=True)
            axes[i].barh(
                grupo_sorted["Nomestado"],
                grupo_sorted["Municipios"],
                color=palette[i]
            )
            axes[i].set_title(
                f"Clúster {cluster_id}: {NOMBRES_CLUSTERS_LARGO[cluster_id]}",
                fontweight="bold", fontsize=11
            )
            axes[i].set_xlabel("Número de municipios")
            axes[i].grid(True, linestyle="--", alpha=0.4, axis="x")

        plt.suptitle("Top 5 Estados por Clúster de Vulnerabilidad Agrícola",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()

    def graficar_radar_perfiles(self) -> None:
        """
        Genera un gráfico de radar (spider chart) con el perfil normalizado de las
        seis variables de clustering para cada clúster.
        Permite comparar visualmente los perfiles de vulnerabilidad de forma simultánea.
        """
        if self.kmeans is None:
            raise ValueError("Entrena el modelo primero con entrenar_kmeans().")

        resumen = self.obtener_resumen_clusters()

        variables = ["Sembrada\nmedia", "Riesgo\nsiniestro", "Prop\ntemporal",
                     "Rendimiento\nmedio", "Precio\nmedio", "Diversidad\nmedia"]

        valores_raw = resumen[[
            "Sembrada_media", "Riesgo_siniestro_medio", "Prop_temporal_media",
            "Rendimiento_promedio", "Precio_promedio", "Diversidad_media"
        ]].values

        valores_norm = MinMaxScaler().fit_transform(valores_raw)

        N = len(variables)
        angulos = [n / float(N) * 2 * np.pi for n in range(N)]
        angulos += angulos[:1]

        colores = ["#2E7D32", "#1976D2", "#FF8F00", "#C62828"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw=dict(polar=True))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            vals = list(valores_norm[i]) + [valores_norm[i][0]]
            ax.plot(angulos, vals, color=colores[i], linewidth=2)
            ax.fill(angulos, vals, color=colores[i], alpha=0.25)
            ax.set_xticks(angulos[:-1])
            ax.set_xticklabels(variables, size=9)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7)
            ax.set_title(
                f"Clúster {i}\n{NOMBRES_CLUSTERS_LARGO[i]}",
                size=11, fontweight="bold", pad=15
            )

        plt.suptitle("Perfil Comparativo de Variables por Clúster",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()

    def guardar_pipeline(self, ruta: str | Path) -> None:
        """
        Serializa y guarda el pipeline (escalador, pca, kmeans y nombres de variables)
        en disco usando pickle para su posterior reutilización.
        """
        ruta_path = Path(ruta)
        ruta_path.parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            "scaler": self.scaler,
            "pca": self.pca,
            "kmeans": self.kmeans,
            "features_cols": self.features_cols,
            "random_state": self.random_state
        }

        with open(ruta_path, "wb") as f:
            pickle.dump(data_to_save, f)
        print(f"Pipeline serializado correctamente en: {ruta_path}")

    @classmethod
    def cargar_pipeline(cls, ruta: str | Path) -> "AgroClusteringPipeline":
        """
        Carga un pipeline serializado desde el disco y devuelve una instancia reconstruida.
        """
        ruta_path = Path(ruta)
        if not ruta_path.exists():
            raise FileNotFoundError(f"No existe el archivo en la ruta: {ruta_path}")

        with open(ruta_path, "rb") as f:
            saved_data = pickle.load(f)

        instance = cls(random_state=saved_data["random_state"])
        instance.scaler = saved_data["scaler"]
        instance.pca = saved_data["pca"]
        instance.kmeans = saved_data["kmeans"]
        instance.features_cols = saved_data["features_cols"]
        print(f"Pipeline cargado correctamente desde: {ruta_path}")
        return instance
    def graficar_voronoi_pca(self) -> None:
        """
        Visualiza las regiones de Voronoi aproximadas en el espacio PCA 2D.
        Muestra cómo K-Means divide el espacio de características en regiones
        donde cada punto pertenece al centroide más cercano.
        Solo es una aproximación visual: las fronteras reales son en 6D.
        """
        if self.X_pca is None or self.kmeans is None:
            raise ValueError("Ejecuta entrenar_kmeans() y aplicar_pca() primero.")

        from scipy.spatial import Voronoi, voronoi_plot_2d

        centroides_pca = self.pca.transform(self.kmeans.cluster_centers_)

        x_min, x_max = self.X_pca[:, 0].min() - 1, self.X_pca[:, 0].max() + 1
        y_min, y_max = self.X_pca[:, 1].min() - 1, self.X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

        from sklearn.metrics import pairwise_distances
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        dist_grid = pairwise_distances(grid_points, centroides_pca)
        Z = np.argmin(dist_grid, axis=1).reshape(xx.shape)

        colores_region = ["#d8cce8", "#c5d9e8", "#c8ead8", "#fdf8d0"]
        colores_punto  = ["#440154", "#31688e", "#35b779", "#fde725"]

        fig, ax = plt.subplots(figsize=(10, 7))

        for i in range(4):
            ax.contourf(xx, yy, Z == i, levels=[0.5, 1.5],
                        colors=[colores_region[i]], alpha=0.6)

        ax.contour(xx, yy, Z, levels=[0.5, 1.5, 2.5, 3.5], colors="white", linewidths=1.5, linestyles="--")

        for i in range(4):
            mask = self.df_mun["Cluster"].values == i
            ax.scatter(self.X_pca[mask, 0], self.X_pca[mask, 1], c=colores_punto[i], s=25, alpha=0.6, edgecolors="none", label=NOMBRES_CLUSTERS[i])

        ax.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
           c="red", s=200, marker="o", zorder=5,
           edgecolors="white", linewidths=1.5, label="Centroides")

        ax.set_title("Regiones de Voronoi Aproximadas en Espacio PCA\n"
                    "(Partición geométrica del espacio de K-Means)",
                    fontsize=13, fontweight="bold")
        ax.set_xlabel(f"PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
        ax.set_ylabel(f"PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
        ax.legend(title="Clúster", loc="best", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()