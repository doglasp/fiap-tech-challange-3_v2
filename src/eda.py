import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

class FlightEDA:
    def __init__(self, input_path=None, output_path=None):
        """
        Initializes the FlightEDA class.
        
        Parameters:
        - input_path (str or Path): Path to the raw flights CSV file.
        - output_path (str or Path): Path to save the processed parquet file.
        """

        self.input_path = Path(input_path) if input_path else None      # ← converter
        self.output_path = Path(output_path) if output_path else None   # ← converter


        self.df = None
        self.df_model = None
        self.delay_by_airline = None
        self.delay_by_origin = None
        self.delay_by_period = None
        self.delay_by_day = None

    def load_data(self):
        """Loads raw CSV data using Polars."""
        print(f"Carregando dados de: {self.input_path} ...")
        self.df = pl.read_csv(self.input_path)
        return self.df

    def print_basic_info(self):
        """Prints basic schema, shape, and null count details."""
        if self.df is None:
            self.load_data()
            
        print("\n=== Informações Básicas ===")
        print(f"Shape: {self.df.shape}")
        print("\nHead:")
        print(self.df.head())
        print(f"\nColunas: {self.df.columns}")
        print(f"\nSchema: {self.df.schema}")
        print("\nDescrição estatística:")
        print(self.df.describe())
        
        # Null count percentage
        n_rows = self.df.height
        null_pct = (
            self.df.null_count()
            .transpose(include_header=True)
            .rename({"column": "coluna", "column_0": "qtd_nulos"})
            .with_columns(
                (pl.col("qtd_nulos") / n_rows * 100).alias("pct_nulos")
            )
            .sort("pct_nulos", descending=True)
        )
        print("\nPercentual de Valores Ausentes por Coluna:")
        print(null_pct)

    def preprocess_and_filter(self):
        """Preprocesses the target column and filters out cancelled/diverted flights."""
        if self.df is None:
            self.load_data()
            
        print("\n=== Pré-processamento e Filtragem ===")
        
        # Criar target para classificação
        self.df = self.df.with_columns(
            (pl.col("ARRIVAL_DELAY") > 0).cast(pl.Int8).alias("is_delayed")
        )
        
        print("Amostra do Target (ARRIVAL_DELAY vs is_delayed):")
        print(self.df.select("ARRIVAL_DELAY", "is_delayed").head())
        
        print("\nDistribuição do target (is_delayed):")
        target_dist = (
            self.df.group_by("is_delayed")
            .len()
            .with_columns(
                (pl.col("len") / pl.col("len").sum() * 100).alias("pct")
            )
        )
        print(target_dist)
        
        # Filtrar voos cancelados/desviados
        self.df_model = self.df.filter(
            (pl.col("CANCELLED") == 0) &
            (pl.col("DIVERTED") == 0)
        )
        print(f"Shape após filtrar cancelados e desviados: {self.df_model.shape}")
        
        # Estatísticas básicas de atraso
        delay_stats = self.df_model.select(
            [
                pl.col("ARRIVAL_DELAY").mean().alias("media_atraso_chegada"),
                pl.col("ARRIVAL_DELAY").median().alias("mediana_atraso_chegada"),
                pl.col("DEPARTURE_DELAY").mean().alias("media_atraso_partida"),
                pl.col("DEPARTURE_DELAY").median().alias("mediana_atraso_partida"),
            ]
        )
        print("\nEstatísticas médias de atraso:")
        print(delay_stats)
        
        return self.df_model

    def run_aggregations(self):
        """Computes aggregations by airline, origin airport, period of the day, and day of week."""
        if self.df_model is None:
            self.preprocess_and_filter()
            
        print("\n=== Executando Agregações ===")
        
        # 1. Por Airline
        self.delay_by_airline = (
            self.df_model.group_by("AIRLINE")
            .agg(
                [
                    pl.len().alias("qtd_voos"),
                    pl.col("ARRIVAL_DELAY").mean().alias("media_arrival_delay"),
                    pl.col("DEPARTURE_DELAY").mean().alias("media_departure_delay"),
                    pl.col("is_delayed").mean().alias("taxa_atraso"),
                ]
            )
            .sort("taxa_atraso", descending=True)
        )
        
        # 2. Por Aeroporto de Origem (mínimo 1000 voos)
        self.delay_by_origin = (
            self.df_model.group_by("ORIGIN_AIRPORT")
            .agg(
                [
                    pl.len().alias("qtd_voos"),
                    pl.col("ARRIVAL_DELAY").mean().alias("media_arrival_delay"),
                    pl.col("is_delayed").mean().alias("taxa_atraso"),
                ]
            )
            .filter(pl.col("qtd_voos") >= 1000)
            .sort("taxa_atraso", descending=True)
        )
        
        # Criar variável periodo_dia baseada em SCHEDULED_DEPARTURE
        self.df_model = self.df_model.with_columns(
            (pl.col("SCHEDULED_DEPARTURE") // 100).alias("scheduled_departure_hour")
        )
        self.df_model = self.df_model.with_columns(
            pl.when(pl.col("scheduled_departure_hour") < 6).then(pl.lit("madrugada"))
            .when(pl.col("scheduled_departure_hour") < 12).then(pl.lit("manha"))
            .when(pl.col("scheduled_departure_hour") < 18).then(pl.lit("tarde"))
            .otherwise(pl.lit("noite"))
            .alias("periodo_dia")
        )
        
        # 3. Por Período do Dia
        self.delay_by_period = (
            self.df_model.group_by("periodo_dia")
            .agg(
                [
                    pl.len().alias("qtd_voos"),
                    pl.col("is_delayed").mean().alias("taxa_atraso"),
                    pl.col("ARRIVAL_DELAY").mean().alias("media_arrival_delay"),
                ]
            )
            .sort("taxa_atraso", descending=True)
        )
        
        # 4. Por Dia da Semana
        self.delay_by_day = (
            self.df_model.group_by("DAY_OF_WEEK")
            .agg(
                [
                    pl.len().alias("qtd_voos"),
                    pl.col("is_delayed").mean().alias("taxa_atraso"),
                    pl.col("ARRIVAL_DELAY").mean().alias("media_arrival_delay"),
                ]
            )
            .sort("DAY_OF_WEEK")
        )
        
        print("\nTop 10 Companhias com maior taxa de atraso:")
        print(self.delay_by_airline.head(10))
        print("\nTop 10 Aeroportos com maior taxa de atraso (mínimo 1000 voos):")
        print(self.delay_by_origin.head(10))
        print("\nAtraso por Período do Dia:")
        print(self.delay_by_period)
        print("\nAtraso por Dia da Semana:")
        print(self.delay_by_day)

    def plot_charts(self):
        """Generates plots for the exploratory analysis."""
        if self.delay_by_airline is None or self.delay_by_period is None:
            self.run_aggregations()
            
        print("\nGerando Gráficos...")
        
        # Plot 1: Top 10 Airlines
        plot_df_airline = self.delay_by_airline.sort("taxa_atraso", descending=True).head(10).to_pandas()
        plt.figure(figsize=(10, 5))
        plt.bar(plot_df_airline["AIRLINE"], plot_df_airline["taxa_atraso"])
        plt.title("Top 10 companhias com maior taxa de atraso")
        plt.xlabel("Companhia")
        plt.ylabel("Taxa de atraso")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Plot 2: Delay by period of the day
        plot_df_period = self.delay_by_period.to_pandas()
        plt.figure(figsize=(8, 4))
        plt.bar(plot_df_period["periodo_dia"], plot_df_period["taxa_atraso"])
        plt.title("Taxa de atraso por período do dia")
        plt.xlabel("Período")
        plt.ylabel("Taxa de atraso")
        plt.tight_layout()
        plt.show()
        
        print("Voos no período da noite apresentam maior taxa de atraso, "
              "possivelmente devido ao acúmulo de atrasos ao longo do dia.")

    def save_processed_data(self):
        """Saves the preprocessed data to parquet file."""
        if self.df_model is None:
            self.preprocess_and_filter()
            
        print(f"\nSalvando dados processados em parquet...")
        # Garantir que a pasta de destino exista
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df_model.write_parquet(self.output_path)
        print(f"Arquivo salvo com sucesso em: {self.output_path}")

    def show_nulls_discussion(self):
        """Displays explanation of missing values handling."""
        print("\n=== Análise de Valores Ausentes ===")
        print("1. ARRIVAL_DELAY:")
        print("   - Valores nulos associados a voos cancelados ou desviados")
        print("   - Estratégia: remoção desses registros para viabilizar a modelagem")
        print("\n2. Variáveis de delay (AIRLINE_DELAY, WEATHER_DELAY, etc.):")
        print("   - Alto volume de valores nulos (~80% dos dados)")
        print("   - Interpretação: ausência de atraso daquela causa")
        print("   - Estratégia:")
        print("       - Para EDA: preenchimento com 0")
        print("       - Para modelagem: remoção para evitar data leakage")
        print("\n3. Variáveis categóricas:")
        print("   - Baixa incidência de valores ausentes")
        print("   - Estratégia: imputação pela moda (pipeline de modelagem)")
        print("\n4. Variáveis numéricas:")
        print("   - Baixa incidência de valores ausentes")
        print("   - Estratégia: imputação pela mediana (pipeline de modelagem)")
        
        # Preenchimento de nulos para demonstração na EDA
        df_fill_demo = self.df.with_columns([
            pl.col("AIRLINE_DELAY").fill_null(0),
            pl.col("WEATHER_DELAY").fill_null(0),
            pl.col("LATE_AIRCRAFT_DELAY").fill_null(0),
            pl.col("SECURITY_DELAY").fill_null(0),
        ])
        print("\nNull counts após demonstração de preenchimento:")
        print(df_fill_demo.null_count().transpose(include_header=True))

    def run_all(self, show_plots=True):
        """Runs the entire EDA pipeline."""
        self.load_data()
        self.print_basic_info()
        self.preprocess_and_filter()
        self.run_aggregations()
        if show_plots:
            self.plot_charts()
        self.save_processed_data()
        self.show_nulls_discussion()

if __name__ == "__main__":
    eda = FlightEDA()
    eda.run_all(show_plots=True)