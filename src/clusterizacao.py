import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples


class FlightClusterer:
    """
    Clusterização de aeroportos por perfil operacional de atrasos.

    Pipeline:
        1. Carrega parquet processado
        2. Agrega features por aeroporto de origem (com mínimo de voos)
        3. Padroniza features com StandardScaler
        4. Determina k via Elbow + Silhouette (testando k=2..10)
        5. Treina KMeans final com o k escolhido
        6. Resume perfil de cada cluster (valores originais)
        7. Reduz para 2D com PCA e visualiza
        8. Plota silhouette por cluster
    """

    FEATURE_COLS = [
        "media_arrival_delay",
        "mediana_arrival_delay",
        "std_arrival_delay",
        "taxa_atraso",
        "taxa_atraso_grave",
        "media_departure_delay",
        "distancia_media",
        "tempo_voo_medio",
        "n_airlines",
        "n_destinos",
        "pct_manha",
        "pct_tarde",
        "pct_noite",
    ]

    SUMMARY_COLS = [
        "taxa_atraso",
        "taxa_atraso_grave",
        "media_arrival_delay",
        "media_departure_delay",
        "std_arrival_delay",
        "distancia_media",
        "n_destinos",
        "qtd_voos",
    ]

    def __init__(
        self,
        input_path=None,
        min_voos=500,
        k_range=range(2, 11),
        k_final=None,
        random_state=42,
    ):
        """
        Parameters:
        - input_path (str or Path): Path to the processed flights parquet file.
        - min_voos (int): Mínimo de voos para um aeroporto entrar na clusterização.
        - k_range (range): Range de k testados no Elbow/Silhouette.
        - k_final (int or None): k usado no KMeans final. Se None, usa o k que
          maximiza o Silhouette no k_range.
        - random_state (int): Semente para reprodutibilidade.
        """
        self.input_path = Path(input_path) if input_path else None
        self.min_voos = min_voos
        self.k_range = k_range
        self.k_final = k_final
        self.random_state = random_state

        # Dados
        self.df = None
        self.airport_features = None
        self.ap_pd = None
        self.X = None
        self.X_scaled = None
        self.X_scaled_df = None
        self.scaler = None

        # Resultados da seleção de k
        self.inertias = None
        self.silhouettes = None
        self.best_k = None

        # Modelo final
        self.km_final = None
        self.perfil = None

        # PCA
        self.pca_full = None
        self.pca_2d = None
        self.var_pc1 = None
        self.var_pc2 = None

    # ------------------------------------------------------------------
    # 1. Carregamento e agregação por aeroporto
    # ------------------------------------------------------------------
    def load_data(self):
        """Carrega o parquet processado."""
        print(f"Carregando dados de: {self.input_path} ...")
        print(f"Existe? {self.input_path.exists()}")

        self.df = pl.read_parquet(self.input_path)
        print(f"Shape: {self.df.shape}")
        return self.df

    def aggregate_by_airport(self):
        """Agrega features por aeroporto de origem (com mínimo de voos)."""
        if self.df is None:
            self.load_data()

        print(f"\n=== Agregando features por aeroporto (mín. {self.min_voos} voos) ===")
        self.airport_features = (
            self.df.filter(pl.col("ARRIVAL_DELAY").is_not_null())
            .group_by("ORIGIN_AIRPORT")
            .agg(
                [
                    pl.len().alias("qtd_voos"),
                    # Atraso na chegada
                    pl.col("ARRIVAL_DELAY").mean().alias("media_arrival_delay"),
                    pl.col("ARRIVAL_DELAY").median().alias("mediana_arrival_delay"),
                    pl.col("ARRIVAL_DELAY").std().alias("std_arrival_delay"),
                    (pl.col("ARRIVAL_DELAY") > 0).mean().alias("taxa_atraso"),
                    (pl.col("ARRIVAL_DELAY") > 15).mean().alias("taxa_atraso_grave"),
                    # Atraso na partida
                    pl.col("DEPARTURE_DELAY").mean().alias("media_departure_delay"),
                    # Características operacionais
                    pl.col("DISTANCE").mean().alias("distancia_media"),
                    pl.col("SCHEDULED_TIME").mean().alias("tempo_voo_medio"),
                    pl.col("AIRLINE").n_unique().alias("n_airlines"),
                    pl.col("DESTINATION_AIRPORT").n_unique().alias("n_destinos"),
                    # Mix de período do dia
                    (pl.col("periodo_dia") == "manha").mean().alias("pct_manha"),
                    (pl.col("periodo_dia") == "tarde").mean().alias("pct_tarde"),
                    (pl.col("periodo_dia") == "noite").mean().alias("pct_noite"),
                ]
            )
            .filter(pl.col("qtd_voos") >= self.min_voos)
            .sort("qtd_voos", descending=True)
        )

        print(f"Aeroportos com >= {self.min_voos} voos: {self.airport_features.height}")
        return self.airport_features

    # ------------------------------------------------------------------
    # 2. Pré-processamento (padronização)
    # ------------------------------------------------------------------
    def preprocess_features(self):
        """Converte para pandas, preenche nulos com mediana e padroniza com StandardScaler."""
        if self.airport_features is None:
            self.aggregate_by_airport()

        self.ap_pd = self.airport_features.to_pandas().set_index("ORIGIN_AIRPORT")
        self.X = self.ap_pd[self.FEATURE_COLS].copy()

        print("\n=== Pré-processamento ===")
        print("Nulos por feature:")
        print(self.X.isnull().sum())

        # Preencher nulos com mediana
        self.X = self.X.fillna(self.X.median())

        # Padronizar
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled_df = pd.DataFrame(
            self.X_scaled, index=self.X.index, columns=self.FEATURE_COLS
        )

        print(f"\nShape para clusterização: {self.X_scaled.shape}")
        return self.X_scaled

    # ------------------------------------------------------------------
    # 3. Determinação do número de clusters (Elbow + Silhouette)
    # ------------------------------------------------------------------
    def find_best_k(self):
        """Testa vários k e calcula inércia + silhouette score."""
        if self.X_scaled is None:
            self.preprocess_features()

        print("\n=== Buscando melhor k ===")
        self.inertias = []
        self.silhouettes = []

        for k in self.k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            labels = km.fit_predict(self.X_scaled)
            self.inertias.append(km.inertia_)
            self.silhouettes.append(silhouette_score(self.X_scaled, labels))
            print(
                f"k={k}  inertia={km.inertia_:.1f}  silhouette={self.silhouettes[-1]:.4f}"
            )

        # k com maior silhouette
        self.best_k = list(self.k_range)[
            self.silhouettes.index(max(self.silhouettes))
        ]
        print(f"\nMelhor k pelo Silhouette: {self.best_k}")
        return self.best_k

    def plot_k_selection(self):
        """Plota Elbow e Silhouette lado a lado."""
        if self.inertias is None:
            self.find_best_k()

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # Elbow
        axes[0].plot(list(self.k_range), self.inertias, marker="o", color="steelblue")
        axes[0].set_title("Método Elbow — Inércia por k")
        axes[0].set_xlabel("Número de clusters (k)")
        axes[0].set_ylabel("Inércia")
        axes[0].grid(alpha=0.3)

        # Silhouette
        axes[1].plot(
            list(self.k_range), self.silhouettes, marker="o", color="darkorange"
        )
        axes[1].set_title("Silhouette Score por k")
        axes[1].set_xlabel("Número de clusters (k)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 4. KMeans final
    # ------------------------------------------------------------------
    def fit_kmeans(self):
        """Treina o KMeans com k_final (ou best_k se k_final não foi setado)."""
        if self.best_k is None:
            self.find_best_k()

        # Se o usuário não passou k_final no construtor, usa o best_k
        if self.k_final is None:
            self.k_final = self.best_k

        print(f"\n=== Treinando KMeans final com k={self.k_final} ===")
        self.km_final = KMeans(
            n_clusters=self.k_final,
            random_state=self.random_state,
            n_init="auto",
        )
        self.ap_pd["cluster"] = self.km_final.fit_predict(self.X_scaled)

        print(f"Distribuição de aeroportos por cluster:")
        print(self.ap_pd["cluster"].value_counts().sort_index())
        return self.km_final

    # ------------------------------------------------------------------
    # 5. Resumo dos clusters
    # ------------------------------------------------------------------
    def summarize_clusters(self):
        """Calcula perfil médio de cada cluster (valores originais, não escalados)."""
        if self.km_final is None:
            self.fit_kmeans()

        self.perfil = (
            self.ap_pd[self.SUMMARY_COLS + ["cluster"]]
            .groupby("cluster")
            .mean()
            .round(2)
            .sort_index()
        )

        print("\n=== Perfil médio de cada cluster ===")
        print(self.perfil)
        return self.perfil

    def show_top_airports_per_cluster(self, top_n=8):
        """Lista os aeroportos com maior volume em cada cluster."""
        if self.km_final is None:
            self.fit_kmeans()

        print(f"\n=== Top {top_n} aeroportos por volume em cada cluster ===")
        for c in sorted(self.ap_pd["cluster"].unique()):
            subset = self.ap_pd[self.ap_pd["cluster"] == c].sort_values(
                "qtd_voos", ascending=False
            )
            top = subset.head(top_n).index.tolist()
            taxa = subset["taxa_atraso"].mean()
            print(
                f"Cluster {c} ({len(subset)} aeroportos | taxa_atraso média={taxa:.2%}): {top}"
            )

    def plot_centroids_heatmap(self):
        """Heatmap dos centróides (valores padronizados)."""
        if self.km_final is None:
            self.fit_kmeans()

        centroids_df = pd.DataFrame(
            self.km_final.cluster_centers_,
            columns=self.FEATURE_COLS,
            index=[f"Cluster {i}" for i in range(self.k_final)],
        )

        plt.figure(figsize=(14, max(3, self.k_final * 0.8)))
        sns.heatmap(
            centroids_df,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r",
            center=0,
            linewidths=0.5,
        )
        plt.title("Centróides dos clusters (valores padronizados)")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 6. PCA — variância explicada e projeção 2D
    # ------------------------------------------------------------------
    def fit_pca_full(self):
        """PCA completo para análise de variância explicada."""
        if self.X_scaled is None:
            self.preprocess_features()

        self.pca_full = PCA(random_state=self.random_state)
        self.pca_full.fit(self.X_scaled)
        return self.pca_full

    def plot_pca_variance(self):
        """Plota variância explicada acumulada."""
        if self.pca_full is None:
            self.fit_pca_full()

        variancia_acumulada = np.cumsum(self.pca_full.explained_variance_ratio_)

        plt.figure(figsize=(9, 4))
        plt.plot(
            range(1, len(variancia_acumulada) + 1),
            variancia_acumulada,
            marker="o",
            color="steelblue",
        )
        plt.axhline(0.90, color="red", linestyle="--", label="90% variância")
        plt.axhline(0.80, color="orange", linestyle="--", label="80% variância")
        plt.xlabel("Número de componentes")
        plt.ylabel("Variância explicada acumulada")
        plt.title("PCA — Variância explicada acumulada")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        n_90 = np.argmax(variancia_acumulada >= 0.90) + 1
        print(f"Componentes necessários para 90% da variância: {n_90}")

    def fit_pca_2d(self):
        """Projeta para 2D para visualização dos clusters."""
        if self.X_scaled is None:
            self.preprocess_features()
        if self.km_final is None:
            self.fit_kmeans()

        self.pca_2d = PCA(n_components=2, random_state=self.random_state)
        X_2d = self.pca_2d.fit_transform(self.X_scaled)

        self.var_pc1 = self.pca_2d.explained_variance_ratio_[0] * 100
        self.var_pc2 = self.pca_2d.explained_variance_ratio_[1] * 100

        self.ap_pd["PC1"] = X_2d[:, 0]
        self.ap_pd["PC2"] = X_2d[:, 1]

        print(
            f"PC1 explica {self.var_pc1:.1f}% | PC2 explica {self.var_pc2:.1f}% | "
            f"Total: {self.var_pc1 + self.var_pc2:.1f}%"
        )
        return X_2d

    def plot_pca_clusters(self, n_label=20):
        """Scatter 2D dos clusters via PCA, com rótulos nos maiores aeroportos."""
        if self.pca_2d is None:
            self.fit_pca_2d()

        colors = plt.cm.tab10(np.linspace(0, 1, self.k_final))

        fig, ax = plt.subplots(figsize=(11, 7))

        for c, color in zip(sorted(self.ap_pd["cluster"].unique()), colors):
            mask = self.ap_pd["cluster"] == c
            subset = self.ap_pd[mask]
            ax.scatter(
                subset["PC1"],
                subset["PC2"],
                label=f"Cluster {c} (n={mask.sum()})",
                color=color,
                s=60,
                alpha=0.75,
                edgecolors="white",
                linewidths=0.4,
            )

        # Anotar os maiores aeroportos
        top_airports = self.ap_pd.sort_values("qtd_voos", ascending=False).head(n_label).index
        for ap in top_airports:
            row = self.ap_pd.loc[ap]
            ax.annotate(
                ap,
                (row["PC1"], row["PC2"]),
                fontsize=7,
                alpha=0.85,
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax.set_xlabel(f"PC1 ({self.var_pc1:.1f}% variância)")
        ax.set_ylabel(f"PC2 ({self.var_pc2:.1f}% variância)")
        ax.set_title(f"Clusters de aeroportos — Projeção PCA 2D (k={self.k_final})")
        ax.legend(loc="upper right", framealpha=0.8)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    def plot_pca_loadings(self):
        """Mostra os loadings de PC1 e PC2 — o que cada componente captura."""
        if self.pca_2d is None:
            self.fit_pca_2d()

        loadings = pd.DataFrame(
            self.pca_2d.components_.T,
            index=self.FEATURE_COLS,
            columns=["PC1", "PC2"],
        ).sort_values("PC1", ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for i, pc in enumerate(["PC1", "PC2"]):
            data = loadings[pc].sort_values()
            colors_bar = ["#d62728" if v > 0 else "#1f77b4" for v in data]
            axes[i].barh(data.index, data.values, color=colors_bar)
            axes[i].axvline(0, color="black", linewidth=0.8)
            axes[i].set_title(
                f"Loadings — {pc} ({self.pca_2d.explained_variance_ratio_[i]*100:.1f}%)"
            )
            axes[i].set_xlabel("Loading")
            axes[i].grid(alpha=0.25, axis="x")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 7. Silhouette plot
    # ------------------------------------------------------------------
    def plot_silhouette(self):
        """Silhouette plot detalhado por cluster."""
        if self.km_final is None:
            self.fit_kmeans()

        sample_silhouette_values = silhouette_samples(
            self.X_scaled, self.ap_pd["cluster"]
        )
        avg_score = silhouette_score(self.X_scaled, self.ap_pd["cluster"])

        fig, ax = plt.subplots(figsize=(9, max(4, self.k_final * 1.2)))
        y_lower = 10

        for c in range(self.k_final):
            cluster_vals = np.sort(
                sample_silhouette_values[self.ap_pd["cluster"] == c]
            )
            size = cluster_vals.shape[0]
            y_upper = y_lower + size
            color = cm.nipy_spectral(float(c) / self.k_final)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_vals,
                alpha=0.7,
                color=color,
            )
            ax.text(-0.05, y_lower + 0.5 * size, str(c), fontsize=10)
            y_lower = y_upper + 10

        ax.axvline(
            x=avg_score,
            color="red",
            linestyle="--",
            label=f"Avg silhouette = {avg_score:.3f}",
        )
        ax.set_xlabel("Silhouette coefficient")
        ax.set_ylabel("Cluster")
        ax.set_title(f"Silhouette plot — k={self.k_final}")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------
    def run_all(self, show_plots=True):
        """Executa o pipeline completo de clusterização."""
        self.load_data()
        self.aggregate_by_airport()
        self.preprocess_features()
        self.find_best_k()
        if show_plots:
            self.plot_k_selection()
        self.fit_kmeans()
        self.summarize_clusters()
        self.show_top_airports_per_cluster()
        if show_plots:
            self.plot_centroids_heatmap()
            self.fit_pca_full()
            self.plot_pca_variance()
            self.fit_pca_2d()
            self.plot_pca_clusters()
            self.plot_pca_loadings()
            self.plot_silhouette()


if __name__ == "__main__":
    clusterer = FlightClusterer(
        input_path="../data/processed/flights_model.parquet",
    )
    clusterer.run_all(show_plots=True)
