import time
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from lightgbm import LGBMRegressor


class FlightRegressor:
    """
    Regressão do atraso de chegada (ARRIVAL_DELAY em minutos).

    Pipeline:
        1. Carrega parquet processado e filtra nulos da target
        2. Seleciona features e target
        3. Amostra (sample_frac), converte para pandas e separa X / y
        4. Split treino/teste
        5. Pré-processa (imputação + scaler + OHE) via ColumnTransformer com fit em cache
        6. Treina Linear Regression, Decision Tree e LightGBM
           (três famílias: linear → árvore simples → boosting de árvores)
        7. Avalia e compara (MAE, RMSE, R²)
        8. Plota comparação e dispersão predito vs real
    """

    FEATURE_COLS = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE",
        "SCHEDULED_TIME",
        "DISTANCE",
        "periodo_dia",
    ]

    CATEGORICAL_FEATURES = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "periodo_dia",
    ]

    NUMERIC_FEATURES = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "SCHEDULED_DEPARTURE",
        "SCHEDULED_TIME",
        "DISTANCE",
    ]

    COLS_DROP = [
        "AIRLINE_DELAY",
        "WEATHER_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "AIR_SYSTEM_DELAY",
        "SECURITY_DELAY",
        "DEPARTURE_DELAY",
    ]

    TARGET_COL = "ARRIVAL_DELAY"

    def __init__(
        self,
        input_path=None,
        test_size=0.2,
        random_state=42,
        sample_frac=0.3,
    ):
        """
        Parameters:
        - input_path (str or Path): Path to the processed flights parquet file.
        - test_size (float): Proporção do conjunto de teste.
        - random_state (int): Semente para reprodutibilidade.
        - sample_frac (float): Fração do dataset usada no treino (default 0.3 = 30%).
          Use valor menor (ex: 0.1) para iterar mais rápido, ou 1.0 para o dataset completo.
        """
        self.input_path = Path(input_path) if input_path else None
        self.test_size = test_size
        self.random_state = random_state
        self.sample_frac = sample_frac

        # Dados
        self.df = None
        self.df_model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Pipelines
        self.preprocessor = None
        self.lr_pipeline = None
        self.dt_pipeline = None
        self.gb_pipeline = None

        # Tempos de treino
        self.lr_time = None
        self.dt_time = None
        self.gb_time = None

        # Predições
        self.y_pred_lr = None
        self.y_pred_dt = None
        self.y_pred_gb = None

        # Métricas
        self.metrics_lr = None
        self.metrics_dt = None
        self.metrics_gb = None
        self.comparison_df = None

    # ------------------------------------------------------------------
    # 1. Carregamento, filtro de nulos e seleção de features
    # ------------------------------------------------------------------
    def load_data(self):
        """Carrega o parquet, remove nulos da target e seleciona features + target."""
        print(f"Carregando dados de: {self.input_path} ...")
        print(f"Existe? {self.input_path.exists()}")

        self.df = pl.read_parquet(self.input_path)
        print(f"Shape original: {self.df.shape}")

        # Remover nulos da target
        self.df_model = self.df.filter(pl.col(self.TARGET_COL).is_not_null())

        # Dropar colunas que causariam data leakage (já se sabe ao chegar)
        self.df_model = self.df_model.drop(self.COLS_DROP)

        # Selecionar apenas features + target
        self.df_model = self.df_model.select(self.FEATURE_COLS + [self.TARGET_COL])

        print(f"Shape após filtro e seleção: {self.df_model.shape}")
        return self.df_model

    # ------------------------------------------------------------------
    # 2. Amostragem e separação X / y
    # ------------------------------------------------------------------
    def prepare_features(self):
        """Aplica amostragem, converte pra pandas e separa X / y."""
        if self.df_model is None:
            self.load_data()

        if self.sample_frac is not None and self.sample_frac < 1.0:
            df_sampled = self.df_model.sample(
                fraction=self.sample_frac, seed=self.random_state
            )
            print(f"Amostragem aplicada ({self.sample_frac:.0%}): {df_sampled.shape}")
        else:
            df_sampled = self.df_model
            print(f"Sem amostragem — usando dataset completo: {df_sampled.shape}")

        df_pd = df_sampled.to_pandas()
        self.X = df_pd[self.FEATURE_COLS]
        self.y = df_pd[self.TARGET_COL]

        print(f"X: {self.X.shape} | y: {self.y.shape}")
        print("Categóricas:", self.CATEGORICAL_FEATURES)
        print("Numéricas:", self.NUMERIC_FEATURES)
        return self.X, self.y

    def split_data(self):
        """Train/test split (sem stratify — target é contínuo)."""
        if self.X is None:
            self.prepare_features()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        print(f"X_train: {self.X_train.shape}")
        print(f"X_test:  {self.X_test.shape}")
        print(f"y_train: {self.y_train.shape}")
        print(f"y_test:  {self.y_test.shape}")

    # ------------------------------------------------------------------
    # 3. Pré-processamento (fit uma única vez, reutiliza em todos os modelos)
    # ------------------------------------------------------------------
    def build_preprocessor(self):
        """Monta o ColumnTransformer e faz fit no X_train pra ficar em cache."""
        if self.X_train is None:
            self.split_data()

        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            [
                ("num", numeric_transformer, self.NUMERIC_FEATURES),
                ("cat", categorical_transformer, self.CATEGORICAL_FEATURES),
            ]
        )

        # Fit uma única vez — todos os modelos reutilizam
        self.preprocessor.fit(self.X_train)
        print("✓ Preprocessor cached — fit uma única vez")
        return self.preprocessor

    # ------------------------------------------------------------------
    # 4. Treinamento dos três modelos
    # ------------------------------------------------------------------
    def train_linear_regression(self):
        """Treina a Regressão Linear."""
        if self.preprocessor is None:
            self.build_preprocessor()

        print("\n=== Treinando Linear Regression ===")
        start = time.time()
        self.lr_pipeline = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                ("model", LinearRegression()),
            ]
        )
        self.lr_pipeline.fit(self.X_train, self.y_train)
        self.lr_time = time.time() - start
        print(f"⏱ Linear Regression treinou em {self.lr_time:.2f}s")

        self.y_pred_lr = self.lr_pipeline.predict(self.X_test)
        return self.lr_pipeline

    def train_decision_tree(self):
        """Treina o Decision Tree Regressor (árvore única — base do Random Forest)."""
        if self.preprocessor is None:
            self.build_preprocessor()

        print("\n=== Treinando Decision Tree ===")
        start = time.time()
        self.dt_pipeline = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=8,
                        min_samples_split=20,
                        min_samples_leaf=10,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.dt_pipeline.fit(self.X_train, self.y_train)
        self.dt_time = time.time() - start
        print(f"⏱ Decision Tree treinou em {self.dt_time:.2f}s")

        self.y_pred_dt = self.dt_pipeline.predict(self.X_test)
        return self.dt_pipeline

    def train_lightgbm(self):
        """Treina o LightGBM."""
        if self.preprocessor is None:
            self.build_preprocessor()

        print("\n=== Treinando LightGBM ===")
        start = time.time()
        self.gb_pipeline = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.gb_pipeline.fit(self.X_train, self.y_train)
        self.gb_time = time.time() - start
        print(f"⏱ LightGBM treinou em {self.gb_time:.2f}s")

        self.y_pred_gb = self.gb_pipeline.predict(self.X_test)
        return self.gb_pipeline

    # ------------------------------------------------------------------
    # 5. Avaliação
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_metrics(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
        }

    def evaluate(self):
        """Calcula métricas dos três modelos e monta DataFrame comparativo."""
        if self.y_pred_lr is None:
            self.train_linear_regression()
        if self.y_pred_dt is None:
            self.train_decision_tree()
        if self.y_pred_gb is None:
            self.train_lightgbm()

        self.metrics_lr = self._compute_metrics(self.y_test, self.y_pred_lr)
        self.metrics_dt = self._compute_metrics(self.y_test, self.y_pred_dt)
        self.metrics_gb = self._compute_metrics(self.y_test, self.y_pred_gb)

        self.comparison_df = pd.DataFrame(
            [
                {"modelo": "Linear Regression", **self.metrics_lr},
                {"modelo": "Decision Tree", **self.metrics_dt},
                {"modelo": "LightGBM", **self.metrics_gb},
            ]
        ).set_index("modelo")

        print("\n=== Comparação de modelos ===")
        print(self.comparison_df.to_string())
        return self.comparison_df

    # ------------------------------------------------------------------
    # 6. Gráficos
    # ------------------------------------------------------------------
    def plot_comparison(self):
        """Gráfico de barras comparando MAE, RMSE e R² dos modelos."""
        if self.comparison_df is None:
            self.evaluate()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, metric, color in zip(
            axes, ["MAE", "RMSE", "R2"], ["steelblue", "darkorange", "seagreen"]
        ):
            self.comparison_df[metric].plot(kind="bar", ax=ax, color=color)
            ax.set_title(metric)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(alpha=0.3, axis="y")

        plt.suptitle("Comparação de modelos — Regressão")
        plt.tight_layout()
        plt.show()

    def plot_predicted_vs_real(self, sample_size=5000):
        """Dispersão predito vs real para os três modelos."""
        if self.y_pred_lr is None or self.y_pred_dt is None or self.y_pred_gb is None:
            self.evaluate()

        # Amostra pra não poluir o scatter com milhões de pontos
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(self.y_test), size=min(sample_size, len(self.y_test)), replace=False)
        y_true_sample = self.y_test.iloc[idx].values

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        models = [
            ("Linear Regression", self.y_pred_lr, "steelblue"),
            ("Decision Tree", self.y_pred_dt, "darkorange"),
            ("LightGBM", self.y_pred_gb, "seagreen"),
        ]

        lim_min = y_true_sample.min()
        lim_max = y_true_sample.max()

        for ax, (name, y_pred, color) in zip(axes, models):
            y_pred_sample = y_pred[idx]
            ax.scatter(y_true_sample, y_pred_sample, alpha=0.25, s=8, color=color)
            ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=0.8)
            ax.set_xlabel("Real (min)")
            ax.set_ylabel("Predito (min)")
            ax.set_title(f"{name}\nR²={self.comparison_df.loc[name, 'R2']:.3f}")
            ax.grid(alpha=0.3)

        plt.suptitle(f"Predito vs Real (amostra de {len(idx)} pontos)")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------
    def run_all(self, show_plots=True):
        """Executa o pipeline completo de regressão."""
        self.load_data()
        self.prepare_features()
        self.split_data()
        self.build_preprocessor()
        self.train_linear_regression()
        self.train_decision_tree()
        self.train_lightgbm()
        self.evaluate()
        if show_plots:
            self.plot_comparison()
            self.plot_predicted_vs_real()


if __name__ == "__main__":
    reg = FlightRegressor(
        input_path="../data/processed/flights_model.parquet",
        sample_frac=0.1,
    )
    reg.run_all(show_plots=True)
