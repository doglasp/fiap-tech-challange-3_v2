import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples


class FlightAnomalyDetector:
    """
    Detecção de aeroportos com perfil operacional atípico.

    Aborda o problema com três métodos independentes e combina via consenso:
        1. Isolation Forest — isola pontos atípicos no espaço de features
        2. Local Outlier Factor (LOF) — detecta anomalias por densidade local
        3. Silhouette individual — aeroportos mal alocados pelo KMeans
        4. Consenso — aeroportos sinalizados por pelo menos 2 dos 3 métodos

    Pipeline:
        1. Carrega parquet e agrega features por aeroporto (mesmas do notebook 04 + extras)
        2. Padroniza com StandardScaler
        3. Roda os três métodos de detecção
        4. Cruza resultados e identifica consenso
        5. Reduz para 2D com PCA para visualização
        6. Gera gráficos comparativos e perfil dos anômalos
    """

    FEATURE_COLS = [
        "media_arrival_delay",
        "mediana_arrival_delay",
        "std_arrival_delay",
        "taxa_atraso",
        "taxa_atraso_grave",
        "taxa_atraso_critico",
        "p95_arrival_delay",
        "media_departure_delay",
        "distancia_media",
        "tempo_voo_medio",
        "n_airlines",
        "n_destinos",
        "pct_manha",
        "pct_tarde",
        "pct_noite",
    ]

    COMPARE_COLS = [
        "taxa_atraso",
        "taxa_atraso_grave",
        "taxa_atraso_critico",
        "media_arrival_delay",
        "std_arrival_delay",
        "p95_arrival_delay",
        "n_destinos",
        "n_airlines",
    ]

    def __init__(
        self,
        input_path=None,
        min_voos=500,
        contamination=0.05,
        n_neighbors=20,
        k_final=4,
        silhouette_threshold=0.0,
        consensus_threshold=2,
        random_state=42,
    ):
        """
        Parameters:
        - input_path (str or Path): Path to the processed flights parquet file.
        - min_voos (int): Mínimo de voos para um aeroporto entrar na análise.
        - contamination (float): Proporção esperada de anomalias (0.05 = 5%).
          Usada pelo Isolation Forest e LOF.
        - n_neighbors (int): Número de vizinhos para o LOF.
        - k_final (int): k usado no KMeans para o silhouette individual.
          Mantenha igual ao notebook 04.
        - silhouette_threshold (float): Aeroportos com silhouette abaixo desse valor
          são considerados mal alocados (default 0.0 = troca de cluster recomendada).
        - consensus_threshold (int): Quantos métodos precisam sinalizar pra ser
          considerado consenso (default 2 de 3).
        - random_state (int): Semente para reprodutibilidade.
        """
        self.input_path = Path(input_path) if input_path else None
        self.min_voos = min_voos
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.k_final = k_final
        self.silhouette_threshold = silhouette_threshold
        self.consensus_threshold = consensus_threshold
        self.random_state = random_state

        # Dados
        self.df = None
        self.airport_features = None
        self.X = None
        self.X_scaled = None
        self.X_scaled_df = None
        self.scaler = None

        # Modelos
        self.iso = None
        self.lof = None
        self.km = None
        self.pca = None

        # Resultados
        self.var_pc1 = None
        self.var_pc2 = None
        self.anomalos_consenso = None
        self.perfil_comparativo = None

    # ------------------------------------------------------------------
    # 1. Agregação por aeroporto
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
                    pl.col("ARRIVAL_DELAY").mean().alias("media_arrival_delay"),
                    pl.col("ARRIVAL_DELAY").median().alias("mediana_arrival_delay"),
                    pl.col("ARRIVAL_DELAY").std().alias("std_arrival_delay"),
                    (pl.col("ARRIVAL_DELAY") > 0).mean().alias("taxa_atraso"),
                    (pl.col("ARRIVAL_DELAY") > 15).mean().alias("taxa_atraso_grave"),
                    (pl.col("ARRIVAL_DELAY") > 60).mean().alias("taxa_atraso_critico"),
                    pl.col("ARRIVAL_DELAY").quantile(0.95).alias("p95_arrival_delay"),
                    pl.col("DEPARTURE_DELAY").mean().alias("media_departure_delay"),
                    pl.col("DISTANCE").mean().alias("distancia_media"),
                    pl.col("SCHEDULED_TIME").mean().alias("tempo_voo_medio"),
                    pl.col("AIRLINE").n_unique().alias("n_airlines"),
                    pl.col("DESTINATION_AIRPORT").n_unique().alias("n_destinos"),
                    (pl.col("periodo_dia") == "manha").mean().alias("pct_manha"),
                    (pl.col("periodo_dia") == "tarde").mean().alias("pct_tarde"),
                    (pl.col("periodo_dia") == "noite").mean().alias("pct_noite"),
                ]
            )
            .filter(pl.col("qtd_voos") >= self.min_voos)
            .to_pandas()
            .set_index("ORIGIN_AIRPORT")
        )

        print(f"Aeroportos: {len(self.airport_features)}")
        return self.airport_features

    # ------------------------------------------------------------------
    # 2. Pré-processamento
    # ------------------------------------------------------------------
    def preprocess_features(self):
        """Preenche nulos com mediana e padroniza com StandardScaler."""
        if self.airport_features is None:
            self.aggregate_by_airport()

        self.X = self.airport_features[self.FEATURE_COLS].fillna(
            self.airport_features[self.FEATURE_COLS].median()
        )

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled_df = pd.DataFrame(
            self.X_scaled, index=self.X.index, columns=self.FEATURE_COLS
        )

        print(f"Shape: {self.X_scaled.shape}")
        return self.X_scaled

    # ------------------------------------------------------------------
    # 3. Métodos de detecção
    # ------------------------------------------------------------------
    def run_isolation_forest(self):
        """Detecção via Isolation Forest."""
        if self.X_scaled is None:
            self.preprocess_features()

        print("\n=== Isolation Forest ===")
        self.iso = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.iso.fit(self.X_scaled)

        self.airport_features["iso_pred"] = self.iso.predict(self.X_scaled)
        self.airport_features["iso_score"] = self.iso.score_samples(self.X_scaled)
        self.airport_features["iso_anomalia"] = self.airport_features["iso_pred"] == -1

        n = self.airport_features["iso_anomalia"].sum()
        print(f"Anomalias detectadas: {n} de {len(self.airport_features)}")
        print("\nTop 10 mais anômalos:")
        print(
            self.airport_features.sort_values("iso_score")
            .head(10)[
                [
                    "taxa_atraso",
                    "media_arrival_delay",
                    "std_arrival_delay",
                    "n_destinos",
                    "qtd_voos",
                    "iso_score",
                ]
            ]
        )
        return self.iso

    def run_lof(self):
        """Detecção via Local Outlier Factor."""
        if self.X_scaled is None:
            self.preprocess_features()

        print("\n=== Local Outlier Factor ===")
        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            n_jobs=-1,
        )

        lof_pred = self.lof.fit_predict(self.X_scaled)
        lof_scores = -self.lof.negative_outlier_factor_  # positivo = mais anômalo

        self.airport_features["lof_pred"] = lof_pred
        self.airport_features["lof_score"] = lof_scores
        self.airport_features["lof_anomalia"] = lof_pred == -1

        n = self.airport_features["lof_anomalia"].sum()
        print(f"Anomalias detectadas: {n} de {len(self.airport_features)}")
        print("\nTop 10 mais anômalos:")
        print(
            self.airport_features.sort_values("lof_score", ascending=False)
            .head(10)[
                [
                    "taxa_atraso",
                    "media_arrival_delay",
                    "std_arrival_delay",
                    "n_destinos",
                    "qtd_voos",
                    "lof_score",
                ]
            ]
        )
        return self.lof

    def run_silhouette_anomaly(self):
        """Detecção via silhouette individual do KMeans."""
        if self.X_scaled is None:
            self.preprocess_features()

        print(f"\n=== KMeans Silhouette (k={self.k_final}) ===")
        self.km = KMeans(
            n_clusters=self.k_final,
            random_state=self.random_state,
            n_init="auto",
        )
        self.airport_features["cluster"] = self.km.fit_predict(self.X_scaled)

        sil_values = silhouette_samples(self.X_scaled, self.airport_features["cluster"])
        self.airport_features["silhouette"] = sil_values
        self.airport_features["sil_anomalia"] = (
            self.airport_features["silhouette"] < self.silhouette_threshold
        )

        n = self.airport_features["sil_anomalia"].sum()
        print(
            f"Aeroportos mal alocados (silhouette < {self.silhouette_threshold}): {n}"
        )
        print("\nPiores silhouettes:")
        print(
            self.airport_features.sort_values("silhouette")
            .head(10)[
                [
                    "cluster",
                    "silhouette",
                    "taxa_atraso",
                    "media_arrival_delay",
                    "n_destinos",
                ]
            ]
        )
        return self.km

    # ------------------------------------------------------------------
    # 4. Consenso
    # ------------------------------------------------------------------
    def build_consensus(self):
        """Cruza os três métodos e identifica aeroportos de consenso."""
        # Garante que todos os três métodos rodaram
        if "iso_anomalia" not in self.airport_features.columns:
            self.run_isolation_forest()
        if "lof_anomalia" not in self.airport_features.columns:
            self.run_lof()
        if "sil_anomalia" not in self.airport_features.columns:
            self.run_silhouette_anomaly()

        self.airport_features["n_metodos_anomalia"] = (
            self.airport_features["iso_anomalia"].astype(int)
            + self.airport_features["lof_anomalia"].astype(int)
            + self.airport_features["sil_anomalia"].astype(int)
        )

        print("\n=== Consenso ===")
        print("Distribuição por número de métodos que sinalizaram anomalia:")
        print(
            self.airport_features["n_metodos_anomalia"].value_counts().sort_index()
        )

        self.anomalos_consenso = self.airport_features[
            self.airport_features["n_metodos_anomalia"] >= self.consensus_threshold
        ].copy()

        print(
            f"\nAnomalias por consenso (≥{self.consensus_threshold} métodos): "
            f"{len(self.anomalos_consenso)}"
        )
        print(
            self.anomalos_consenso.sort_values(
                "n_metodos_anomalia", ascending=False
            )[
                [
                    "n_metodos_anomalia",
                    "cluster",
                    "taxa_atraso",
                    "taxa_atraso_grave",
                    "media_arrival_delay",
                    "std_arrival_delay",
                    "n_destinos",
                    "qtd_voos",
                ]
            ].round(3)
        )

        return self.anomalos_consenso

    # ------------------------------------------------------------------
    # 5. PCA para visualização
    # ------------------------------------------------------------------
    def fit_pca_2d(self):
        """Projeta para 2D para visualização."""
        if self.X_scaled is None:
            self.preprocess_features()

        self.pca = PCA(n_components=2, random_state=self.random_state)
        X_2d = self.pca.fit_transform(self.X_scaled)

        self.var_pc1, self.var_pc2 = self.pca.explained_variance_ratio_ * 100

        self.airport_features["PC1"] = X_2d[:, 0]
        self.airport_features["PC2"] = X_2d[:, 1]

        print(
            f"PC1={self.var_pc1:.1f}%  PC2={self.var_pc2:.1f}%  "
            f"Total={self.var_pc1 + self.var_pc2:.1f}%"
        )
        return X_2d

    # ------------------------------------------------------------------
    # 6. Visualizações
    # ------------------------------------------------------------------
    def plot_methods_comparison(self):
        """Três scatters PCA lado a lado — um por método."""
        if self.pca is None:
            self.fit_pca_2d()
        if "n_metodos_anomalia" not in self.airport_features.columns:
            self.build_consensus()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        method_cols = [
            ("iso_anomalia", "iso_score", "Isolation Forest", "Score IF (↓ = mais anômalo)"),
            ("lof_anomalia", "lof_score", "Local Outlier Factor", "LOF Score (↑ = mais anômalo)"),
            ("sil_anomalia", "silhouette", "KMeans Silhouette", "Silhouette (↓ = mal alocado)"),
        ]

        for ax, (flag_col, score_col, title, cb_label) in zip(axes, method_cols):
            normal = self.airport_features[~self.airport_features[flag_col]]
            anomalo = self.airport_features[self.airport_features[flag_col]]

            sc = ax.scatter(
                normal["PC1"],
                normal["PC2"],
                c=normal[score_col],
                cmap="RdYlGn",
                s=40,
                alpha=0.7,
                edgecolors="none",
                label="Normal",
            )
            ax.scatter(
                anomalo["PC1"],
                anomalo["PC2"],
                c="red",
                marker="X",
                s=120,
                zorder=5,
                edgecolors="black",
                linewidths=0.5,
                label="Anomalia",
            )

            # Anotar top 5 mais anômalos
            ascending = score_col in ["iso_score", "silhouette"]
            top5 = self.airport_features.sort_values(
                score_col, ascending=ascending
            ).head(5)
            for ap, row in top5.iterrows():
                ax.annotate(
                    ap,
                    (row["PC1"], row["PC2"]),
                    fontsize=7,
                    xytext=(4, 4),
                    textcoords="offset points",
                )

            plt.colorbar(sc, ax=ax, label=cb_label, shrink=0.8)
            ax.set_title(title)
            ax.set_xlabel(f"PC1 ({self.var_pc1:.1f}%)")
            ax.set_ylabel(f"PC2 ({self.var_pc2:.1f}%)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)

        plt.suptitle("Detecção de Anomalias — Projeção PCA 2D", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()

    def plot_consensus_map(self):
        """Mapa PCA único colorindo por número de métodos que sinalizaram."""
        if self.anomalos_consenso is None:
            self.build_consensus()
        if "PC1" not in self.airport_features.columns:
            self.fit_pca_2d()

        # Garantir que anomalos_consenso tenha as colunas PCA
        anomalos_consenso = self.airport_features.loc[self.anomalos_consenso.index]

        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(
            self.airport_features["PC1"],
            self.airport_features["PC2"],
            c=self.airport_features["n_metodos_anomalia"],
            cmap="RdYlGn_r",
            vmin=0,
            vmax=3,
            s=60,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.3,
        )

        # Anotar aeroportos de consenso
        for ap, row in anomalos_consenso.iterrows():
            ax.annotate(
                ap,
                (row["PC1"], row["PC2"]),
                fontsize=8,
                fontweight="bold",
                xytext=(5, 5),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.7),
            )

        cbar = plt.colorbar(sc, ax=ax, ticks=[0, 1, 2, 3])
        cbar.set_label("Nº de métodos que sinalizaram anomalia")
        cbar.ax.set_yticklabels(
            ["0 — Normal", "1 método", "2 métodos", "3 métodos"]
        )

        ax.set_title("Consenso de Anomalias — Projeção PCA 2D", fontsize=12)
        ax.set_xlabel(f"PC1 ({self.var_pc1:.1f}%)")
        ax.set_ylabel(f"PC2 ({self.var_pc2:.1f}%)")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

    def plot_iso_vs_lof(self):
        """Scatter dos scores IF vs LOF, destacando consenso."""
        if self.anomalos_consenso is None:
            self.build_consensus()

        consenso_mask = (
            self.airport_features["n_metodos_anomalia"] >= self.consensus_threshold
        )
        colors_dot = np.where(consenso_mask, "red", "steelblue")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            self.airport_features["iso_score"],
            self.airport_features["lof_score"],
            c=colors_dot,
            s=50,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.3,
        )

        # Linhas de corte
        iso_threshold = self.airport_features.loc[
            self.airport_features["iso_anomalia"], "iso_score"
        ].max()
        lof_threshold = self.airport_features.loc[
            self.airport_features["lof_anomalia"], "lof_score"
        ].min()
        ax.axvline(
            iso_threshold,
            color="orange",
            linestyle="--",
            lw=1,
            label=f"Corte IF ({iso_threshold:.3f})",
        )
        ax.axhline(
            lof_threshold,
            color="purple",
            linestyle="--",
            lw=1,
            label=f"Corte LOF ({lof_threshold:.2f})",
        )

        # Anotar consenso
        for ap, row in self.anomalos_consenso.iterrows():
            ax.annotate(
                ap,
                (row["iso_score"], row["lof_score"]),
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points",
            )

        legend_patches = [
            mpatches.Patch(color="red", label=f"Anomalia consenso (≥{self.consensus_threshold} métodos)"),
            mpatches.Patch(color="steelblue", label="Normal"),
        ]
        existing_handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=legend_patches + existing_handles)
        ax.set_xlabel("Isolation Forest score (↓ = mais anômalo)")
        ax.set_ylabel("LOF score (↑ = mais anômalo)")
        ax.set_title("Isolation Forest vs LOF — scores por aeroporto")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    def plot_profile_comparison(self):
        """Heatmap comparativo do perfil médio: Anômalos vs Normais."""
        if self.anomalos_consenso is None:
            self.build_consensus()

        self.airport_features["grupo"] = np.where(
            self.airport_features["n_metodos_anomalia"] >= self.consensus_threshold,
            "Anômalo",
            "Normal",
        )

        perfil = (
            self.airport_features.groupby("grupo")[self.COMPARE_COLS]
            .mean()
            .round(3)
            .T
        )
        self.perfil_comparativo = perfil

        # Normalizar para o colormap funcionar na escala certa
        perfil_norm = (perfil - perfil.min(axis=1).values.reshape(-1, 1)) / (
            perfil.max(axis=1) - perfil.min(axis=1) + 1e-9
        ).values.reshape(-1, 1)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            perfil_norm,
            annot=perfil,
            fmt=".3f",
            cmap="RdYlGn_r",
            ax=ax,
            linewidths=0.5,
            cbar=False,
        )
        ax.set_title("Perfil médio: Anômalos vs Normais")
        ax.set_xlabel("Grupo")
        plt.tight_layout()
        plt.show()

        print(perfil)
        return perfil

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------
    def run_all(self, show_plots=True):
        """Executa o pipeline completo de detecção de anomalias."""
        self.load_data()
        self.aggregate_by_airport()
        self.preprocess_features()
        self.run_isolation_forest()
        self.run_lof()
        self.run_silhouette_anomaly()
        self.build_consensus()
        self.fit_pca_2d()
        if show_plots:
            self.plot_methods_comparison()
            self.plot_consensus_map()
            self.plot_iso_vs_lof()
            self.plot_profile_comparison()


if __name__ == "__main__":
    detector = FlightAnomalyDetector(
        input_path="../data/processed/flights_model.parquet",
    )
    detector.run_all(show_plots=True)
