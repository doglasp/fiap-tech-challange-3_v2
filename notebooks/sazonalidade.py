import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path


MONTH_NAMES = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr",
    5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago",
    9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
}

DOW_NAMES = {
    1: "Segunda", 2: "Terça", 3: "Quarta",
    4: "Quinta", 5: "Sexta", 6: "Sábado", 7: "Domingo",
}

PERIODO_ORDER = ["madrugada", "manha", "tarde", "noite"]


class FlightSeasonality:
    """
    Análise de padrões sazonais de atrasos.

    Investiga **quando** os atrasos são mais frequentes e intensos:
        1. Sazonalidade mensal
        2. Padrão semanal (dias da semana)
        3. Padrão por período do dia
        4. Heatmaps cruzados (mês × dia, mês × período, dia × período)
        5. Sazonalidade por companhia aérea
        6. Sazonalidade dos aeroportos anômalos (cruzamento com notebook 06)
        7. Combinações críticas (mês + dia + período com piores taxas)
    """

    def __init__(
        self,
        input_path=None,
        top_n_airlines=8,
        anomalous_airports=None,
        min_voos_critical=100,
    ):
        """
        Parameters:
        - input_path (str or Path): Path to the processed flights parquet file.
        - top_n_airlines (int): Número de companhias com maior volume a analisar.
        - anomalous_airports (list[str] or None): Lista de aeroportos anômalos
          (geralmente vinda do notebook 06). Se None, a análise da seção 6 é pulada.
        - min_voos_critical (int): Mínimo de voos por combinação (mês/dia/período)
          para entrar no ranking de cenários críticos.
        """
        self.input_path = Path(input_path) if input_path else None
        self.top_n_airlines = top_n_airlines
        self.anomalous_airports = anomalous_airports or []
        self.min_voos_critical = min_voos_critical

        # Dados base
        self.df = None

        # Agregações
        self.monthly = None
        self.weekly = None
        self.period = None
        self.cross_md = None       # mês × dia da semana
        self.cross_mp = None       # mês × período do dia
        self.cross_dp = None       # dia da semana × período do dia
        self.top_airlines = None
        self.airline_monthly = None
        self.hm_airline = None
        self.anom_monthly = None
        self.critical = None

    # ------------------------------------------------------------------
    # 1. Carregamento e preparação
    # ------------------------------------------------------------------
    def load_data(self):
        """Carrega o parquet, filtra ARRIVAL_DELAY nulo e adiciona flags de atraso."""
        print(f"Carregando dados de: {self.input_path} ...")
        print(f"Existe? {self.input_path.exists()}")

        self.df = pl.read_parquet(self.input_path)

        # Manter apenas voos com delay conhecido (não cancelados/desviados)
        self.df = self.df.filter(pl.col("ARRIVAL_DELAY").is_not_null())

        # Adicionar flags binárias de atraso
        self.df = self.df.with_columns(
            [
                (pl.col("ARRIVAL_DELAY") > 0).cast(pl.Int8).alias("is_delayed"),
                (pl.col("ARRIVAL_DELAY") > 15).cast(pl.Int8).alias("is_delayed_grave"),
            ]
        )

        print(f"Voos: {self.df.height:,}")
        print(f"Meses presentes: {sorted(self.df['MONTH'].unique().to_list())}")
        return self.df

    # ------------------------------------------------------------------
    # 2. Sazonalidade mensal
    # ------------------------------------------------------------------
    def compute_monthly(self):
        """Agrega métricas por mês."""
        if self.df is None:
            self.load_data()

        self.monthly = (
            self.df.group_by("MONTH")
            .agg(
                [
                    pl.len().alias("qtd_voos"),
                    pl.col("is_delayed").mean().alias("taxa_atraso"),
                    pl.col("is_delayed_grave").mean().alias("taxa_atraso_grave"),
                    pl.col("ARRIVAL_DELAY").mean().alias("media_delay"),
                    pl.col("ARRIVAL_DELAY").median().alias("mediana_delay"),
                    pl.col("ARRIVAL_DELAY").quantile(0.90).alias("p90_delay"),
                ]
            )
            .sort("MONTH")
            .to_pandas()
        )
        self.monthly["mes_nome"] = self.monthly["MONTH"].map(MONTH_NAMES)
        return self.monthly

    def plot_monthly(self):
        """Painel 2x2: taxa de atraso, delay médio/mediana, p90 e volume mensal."""
        if self.monthly is None:
            self.compute_monthly()

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # Taxa de atraso
        ax = axes[0, 0]
        ax.bar(
            self.monthly["mes_nome"],
            self.monthly["taxa_atraso"],
            color=plt.cm.RdYlGn_r(
                self.monthly["taxa_atraso"] / self.monthly["taxa_atraso"].max()
            ),
        )
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title("Taxa de atraso por mês")
        ax.set_xlabel("Mês")
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(
            self.monthly["taxa_atraso"].mean(),
            color="gray", linestyle="--", lw=1, label="Média",
        )
        ax.legend()

        # Delay médio e mediana
        ax = axes[0, 1]
        ax.plot(self.monthly["mes_nome"], self.monthly["media_delay"],
                marker="o", color="steelblue", lw=2, label="Média")
        ax.plot(self.monthly["mes_nome"], self.monthly["mediana_delay"],
                marker="s", color="darkorange", lw=2, linestyle="--", label="Mediana")
        ax.fill_between(
            self.monthly["mes_nome"], self.monthly["mediana_delay"],
            self.monthly["media_delay"], alpha=0.15, color="steelblue",
        )
        ax.set_title("Delay médio e mediana por mês (minutos)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        ax.axhline(0, color="black", lw=0.7)

        # P90
        ax = axes[1, 0]
        ax.bar(self.monthly["mes_nome"], self.monthly["p90_delay"],
               color="salmon", alpha=0.8)
        ax.set_title("Percentil 90 do delay por mês (minutos)")
        ax.set_xlabel("Mês")
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(self.monthly["p90_delay"].mean(),
                   color="gray", linestyle="--", lw=1, label="Média")
        ax.legend()

        # Volume
        ax = axes[1, 1]
        ax.bar(self.monthly["mes_nome"], self.monthly["qtd_voos"] / 1000,
               color="lightsteelblue", alpha=0.9)
        ax.set_title("Volume de voos por mês (mil)")
        ax.set_xlabel("Mês")
        ax.tick_params(axis="x", rotation=45)

        plt.suptitle("Sazonalidade Mensal dos Atrasos", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 3. Padrão semanal
    # ------------------------------------------------------------------
    def compute_weekly(self):
        """Agrega métricas por dia da semana."""
        if self.df is None:
            self.load_data()

        self.weekly = (
            self.df.group_by("DAY_OF_WEEK")
            .agg(
                [
                    pl.len().alias("qtd_voos"),
                    pl.col("is_delayed").mean().alias("taxa_atraso"),
                    pl.col("is_delayed_grave").mean().alias("taxa_atraso_grave"),
                    pl.col("ARRIVAL_DELAY").mean().alias("media_delay"),
                    pl.col("ARRIVAL_DELAY").quantile(0.90).alias("p90_delay"),
                ]
            )
            .sort("DAY_OF_WEEK")
            .to_pandas()
        )
        self.weekly["dia_nome"] = self.weekly["DAY_OF_WEEK"].map(DOW_NAMES)
        return self.weekly

    def plot_weekly(self):
        """Três gráficos: taxa, delay médio e volume por dia da semana."""
        if self.weekly is None:
            self.compute_weekly()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        color_dow = plt.cm.RdYlGn_r(
            self.weekly["taxa_atraso"] / self.weekly["taxa_atraso"].max()
        )

        axes[0].bar(self.weekly["dia_nome"], self.weekly["taxa_atraso"], color=color_dow)
        axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        axes[0].set_title("Taxa de atraso por dia da semana")
        axes[0].tick_params(axis="x", rotation=40)
        axes[0].axhline(self.weekly["taxa_atraso"].mean(),
                        color="gray", linestyle="--", lw=1)

        axes[1].bar(self.weekly["dia_nome"], self.weekly["media_delay"],
                    color="steelblue", alpha=0.8)
        axes[1].set_title("Delay médio por dia da semana (min)")
        axes[1].tick_params(axis="x", rotation=40)
        axes[1].axhline(0, color="black", lw=0.7)

        axes[2].bar(self.weekly["dia_nome"], self.weekly["qtd_voos"] / 1000,
                    color="lightsteelblue")
        axes[2].set_title("Volume de voos por dia (mil)")
        axes[2].tick_params(axis="x", rotation=40)

        plt.suptitle("Padrão Semanal dos Atrasos", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 4. Padrão por período do dia
    # ------------------------------------------------------------------
    def compute_period(self):
        """Agrega métricas por período do dia (madrugada/manhã/tarde/noite)."""
        if self.df is None:
            self.load_data()

        self.period = (
            self.df.group_by("periodo_dia")
            .agg(
                [
                    pl.len().alias("qtd_voos"),
                    pl.col("is_delayed").mean().alias("taxa_atraso"),
                    pl.col("is_delayed_grave").mean().alias("taxa_atraso_grave"),
                    pl.col("ARRIVAL_DELAY").mean().alias("media_delay"),
                    pl.col("ARRIVAL_DELAY").quantile(0.90).alias("p90_delay"),
                ]
            )
            .to_pandas()
        )
        self.period["periodo_dia"] = pd.Categorical(
            self.period["periodo_dia"], categories=PERIODO_ORDER, ordered=True
        )
        self.period = self.period.sort_values("periodo_dia")
        return self.period

    def plot_period(self):
        """Três gráficos: taxa, delay médio e volume por período do dia."""
        if self.period is None:
            self.compute_period()

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        color_p = plt.cm.RdYlGn_r(
            self.period["taxa_atraso"].values / self.period["taxa_atraso"].max()
        )

        axes[0].bar(self.period["periodo_dia"], self.period["taxa_atraso"], color=color_p)
        axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        axes[0].set_title("Taxa de atraso por período do dia")

        axes[1].bar(self.period["periodo_dia"], self.period["media_delay"],
                    color="steelblue", alpha=0.8)
        axes[1].set_title("Delay médio por período (min)")
        axes[1].axhline(0, color="black", lw=0.7)

        axes[2].bar(self.period["periodo_dia"], self.period["qtd_voos"] / 1000,
                    color="lightsteelblue")
        axes[2].set_title("Volume de voos por período (mil)")

        plt.suptitle("Padrão por Período do Dia", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 5. Heatmaps cruzados
    # ------------------------------------------------------------------
    def compute_cross_month_dow(self):
        """Pivot: dia da semana × mês com taxa de atraso."""
        if self.df is None:
            self.load_data()

        self.cross_md = (
            self.df.group_by(["MONTH", "DAY_OF_WEEK"])
            .agg(pl.col("is_delayed").mean().alias("taxa_atraso"))
            .to_pandas()
            .pivot(index="DAY_OF_WEEK", columns="MONTH", values="taxa_atraso")
        )
        self.cross_md.index = [DOW_NAMES[i] for i in self.cross_md.index]
        self.cross_md.columns = [MONTH_NAMES[c] for c in self.cross_md.columns]
        return self.cross_md

    def plot_cross_month_dow(self):
        """Heatmap: dia da semana × mês."""
        if self.cross_md is None:
            self.compute_cross_month_dow()

        fig, ax = plt.subplots(figsize=(13, 5))
        sns.heatmap(
            self.cross_md, annot=True, fmt=".1%", cmap="RdYlGn_r",
            linewidths=0.4, ax=ax,
            cbar_kws={"format": mticker.PercentFormatter(xmax=1, decimals=0)},
        )
        ax.set_title("Taxa de atraso: Dia da Semana × Mês", fontsize=12)
        ax.set_xlabel("Mês")
        ax.set_ylabel("Dia da Semana")
        plt.tight_layout()
        plt.show()

    def compute_cross_month_period(self):
        """Pivot: período do dia × mês com taxa de atraso."""
        if self.df is None:
            self.load_data()

        self.cross_mp = (
            self.df.group_by(["MONTH", "periodo_dia"])
            .agg(pl.col("is_delayed").mean().alias("taxa_atraso"))
            .to_pandas()
            .pivot(index="periodo_dia", columns="MONTH", values="taxa_atraso")
        )
        self.cross_mp = self.cross_mp.reindex(PERIODO_ORDER)
        self.cross_mp.columns = [MONTH_NAMES[c] for c in self.cross_mp.columns]
        return self.cross_mp

    def plot_cross_month_period(self):
        """Heatmap: período do dia × mês."""
        if self.cross_mp is None:
            self.compute_cross_month_period()

        fig, ax = plt.subplots(figsize=(13, 4))
        sns.heatmap(
            self.cross_mp, annot=True, fmt=".1%", cmap="RdYlGn_r",
            linewidths=0.4, ax=ax,
            cbar_kws={"format": mticker.PercentFormatter(xmax=1, decimals=0)},
        )
        ax.set_title("Taxa de atraso: Período do Dia × Mês", fontsize=12)
        ax.set_xlabel("Mês")
        ax.set_ylabel("Período do dia")
        plt.tight_layout()
        plt.show()

    def compute_cross_dow_period(self):
        """Pivot: período do dia × dia da semana com taxa de atraso."""
        if self.df is None:
            self.load_data()

        self.cross_dp = (
            self.df.group_by(["DAY_OF_WEEK", "periodo_dia"])
            .agg(pl.col("is_delayed").mean().alias("taxa_atraso"))
            .to_pandas()
            .pivot(index="periodo_dia", columns="DAY_OF_WEEK", values="taxa_atraso")
        )
        self.cross_dp = self.cross_dp.reindex(PERIODO_ORDER)
        self.cross_dp.columns = [DOW_NAMES[c] for c in self.cross_dp.columns]
        return self.cross_dp

    def plot_cross_dow_period(self):
        """Heatmap: período do dia × dia da semana."""
        if self.cross_dp is None:
            self.compute_cross_dow_period()

        fig, ax = plt.subplots(figsize=(11, 4))
        sns.heatmap(
            self.cross_dp, annot=True, fmt=".1%", cmap="RdYlGn_r",
            linewidths=0.4, ax=ax,
            cbar_kws={"format": mticker.PercentFormatter(xmax=1, decimals=0)},
        )
        ax.set_title("Taxa de atraso: Período do Dia × Dia da Semana", fontsize=12)
        ax.set_xlabel("Dia da Semana")
        ax.set_ylabel("Período do dia")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 6. Sazonalidade por companhia
    # ------------------------------------------------------------------
    def compute_airline_monthly(self):
        """Agrega taxa de atraso mensal para as top N companhias."""
        if self.df is None:
            self.load_data()

        self.top_airlines = (
            self.df.group_by("AIRLINE")
            .agg(pl.len().alias("qtd_voos"))
            .sort("qtd_voos", descending=True)
            .head(self.top_n_airlines)["AIRLINE"]
            .to_list()
        )

        self.airline_monthly = (
            self.df.filter(pl.col("AIRLINE").is_in(self.top_airlines))
            .group_by(["AIRLINE", "MONTH"])
            .agg(pl.col("is_delayed").mean().alias("taxa_atraso"))
            .sort(["AIRLINE", "MONTH"])
            .to_pandas()
        )
        self.airline_monthly["mes_nome"] = self.airline_monthly["MONTH"].map(MONTH_NAMES)

        # Pivot para heatmap
        self.hm_airline = self.airline_monthly.pivot(
            index="AIRLINE", columns="MONTH", values="taxa_atraso"
        )
        self.hm_airline.columns = [MONTH_NAMES[c] for c in self.hm_airline.columns]
        return self.airline_monthly

    def plot_airline_heatmap(self):
        """Heatmap: companhia × mês."""
        if self.hm_airline is None:
            self.compute_airline_monthly()

        fig, ax = plt.subplots(figsize=(13, 5))
        sns.heatmap(
            self.hm_airline, annot=True, fmt=".1%", cmap="RdYlGn_r",
            linewidths=0.4, ax=ax,
            cbar_kws={"format": mticker.PercentFormatter(xmax=1, decimals=0)},
        )
        ax.set_title("Taxa de atraso por Companhia × Mês", fontsize=12)
        ax.set_xlabel("Mês")
        ax.set_ylabel("Companhia")
        plt.tight_layout()
        plt.show()

    def plot_airline_lines(self):
        """Evolução mensal da taxa de atraso por companhia (linhas)."""
        if self.airline_monthly is None:
            self.compute_airline_monthly()

        fig, ax = plt.subplots(figsize=(13, 5))

        for airline in self.top_airlines:
            subset = self.airline_monthly[
                self.airline_monthly["AIRLINE"] == airline
            ].sort_values("MONTH")
            ax.plot(subset["mes_nome"], subset["taxa_atraso"],
                    marker="o", lw=2, label=airline, alpha=0.85)

        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title("Evolução mensal da taxa de atraso por companhia", fontsize=12)
        ax.set_xlabel("Mês")
        ax.set_ylabel("Taxa de atraso")
        ax.legend(title="Companhia", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 7. Sazonalidade dos aeroportos anômalos
    # ------------------------------------------------------------------
    def set_anomalous_airports(self, airports):
        """Define a lista de aeroportos anômalos (após rodar o notebook 06)."""
        self.anomalous_airports = list(airports)
        print(f"Aeroportos anômalos definidos: {self.anomalous_airports}")

    def plot_anomalous_seasonality(self):
        """Linhas: sazonalidade mensal dos aeroportos anômalos vs média geral."""
        if not self.anomalous_airports:
            print(
                "⚠ Lista anomalous_airports vazia. "
                "Defina com set_anomalous_airports([...]) usando os anômalos do notebook 06."
            )
            return

        if self.df is None:
            self.load_data()
        if self.monthly is None:
            self.compute_monthly()

        self.anom_monthly = (
            self.df.filter(pl.col("ORIGIN_AIRPORT").is_in(self.anomalous_airports))
            .group_by(["ORIGIN_AIRPORT", "MONTH"])
            .agg(
                [
                    pl.col("is_delayed").mean().alias("taxa_atraso"),
                    pl.col("ARRIVAL_DELAY").mean().alias("media_delay"),
                ]
            )
            .sort(["ORIGIN_AIRPORT", "MONTH"])
            .to_pandas()
        )
        self.anom_monthly["mes_nome"] = self.anom_monthly["MONTH"].map(MONTH_NAMES)

        media_geral = self.monthly.set_index("MONTH")["taxa_atraso"]

        fig, ax = plt.subplots(figsize=(13, 5))

        for ap in self.anomalous_airports:
            subset = self.anom_monthly[
                self.anom_monthly["ORIGIN_AIRPORT"] == ap
            ].sort_values("MONTH")
            if subset.empty:
                continue
            ax.plot(subset["mes_nome"], subset["taxa_atraso"],
                    marker="o", lw=2, label=ap)

        ax.plot(
            [MONTH_NAMES[m] for m in sorted(media_geral.index)],
            media_geral.sort_index().values,
            color="black", lw=2, linestyle="--", label="Média geral",
        )

        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title("Sazonalidade dos aeroportos anômalos vs média geral", fontsize=12)
        ax.set_xlabel("Mês")
        ax.set_ylabel("Taxa de atraso")
        ax.legend(title="Aeroporto", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 8. Combinações críticas
    # ------------------------------------------------------------------
    def compute_critical_combinations(self, top_n=15):
        """Top N combinações (mês, dia da semana, período) com maior taxa de atraso."""
        if self.df is None:
            self.load_data()

        self.critical = (
            self.df.group_by(["MONTH", "DAY_OF_WEEK", "periodo_dia"])
            .agg(
                [
                    pl.len().alias("qtd_voos"),
                    pl.col("is_delayed").mean().alias("taxa_atraso"),
                    pl.col("ARRIVAL_DELAY").mean().alias("media_delay"),
                ]
            )
            .filter(pl.col("qtd_voos") >= self.min_voos_critical)
            .sort("taxa_atraso", descending=True)
            .head(top_n)
            .to_pandas()
        )

        self.critical["mes_nome"] = self.critical["MONTH"].map(MONTH_NAMES)
        self.critical["dia_nome"] = self.critical["DAY_OF_WEEK"].map(DOW_NAMES)
        self.critical["combinacao"] = (
            self.critical["mes_nome"] + " / "
            + self.critical["dia_nome"] + " / "
            + self.critical["periodo_dia"]
        )
        return self.critical

    def plot_critical_combinations(self, top_n=15):
        """Barra horizontal: combinações críticas com maior taxa de atraso."""
        if self.critical is None or len(self.critical) != top_n:
            self.compute_critical_combinations(top_n=top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(self.critical))[::-1])
        ax.barh(
            self.critical["combinacao"][::-1],
            self.critical["taxa_atraso"][::-1],
            color=colors,
        )
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(
            f"Top {top_n} combinações com maior taxa de atraso\n"
            "(Mês / Dia da semana / Período)",
            fontsize=11,
        )
        ax.set_xlabel("Taxa de atraso")
        ax.grid(alpha=0.2, axis="x")
        plt.tight_layout()
        plt.show()

        print("\nTop 5 cenários críticos:")
        print(
            self.critical[["combinacao", "taxa_atraso", "media_delay", "qtd_voos"]]
            .head(5)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------
    def run_all(self, show_plots=True):
        """Executa o pipeline completo de análise sazonal."""
        self.load_data()
        self.compute_monthly()
        self.compute_weekly()
        self.compute_period()
        self.compute_cross_month_dow()
        self.compute_cross_month_period()
        self.compute_cross_dow_period()
        self.compute_airline_monthly()
        self.compute_critical_combinations()

        if show_plots:
            self.plot_monthly()
            self.plot_weekly()
            self.plot_period()
            self.plot_cross_month_dow()
            self.plot_cross_month_period()
            self.plot_cross_dow_period()
            self.plot_airline_heatmap()
            self.plot_airline_lines()
            if self.anomalous_airports:
                self.plot_anomalous_seasonality()
            self.plot_critical_combinations()


if __name__ == "__main__":
    seasonality = FlightSeasonality(
        input_path="../data/processed/flights_model.parquet",
    )
    seasonality.run_all(show_plots=True)
