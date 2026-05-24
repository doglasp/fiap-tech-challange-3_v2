import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    confusion_matrix,
)


class FlightClassifier:
    """
    Classificação binária de atrasos de voos (is_delayed).

    Pipeline:
        1. Carrega parquet processado
        2. Seleciona features e target
        3. Amostra (sample_frac), converte para pandas e separa X / y
        4. Split treino/teste estratificado
        5. Pré-processa (imputação + scaler + OHE) via ColumnTransformer
        6. Treina Logistic Regression e Random Forest
        7. Avalia e compara (accuracy, precision, recall, f1, ROC-AUC)
        8. Plota curva ROC, matrizes de confusão e feature importance
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

    TARGET_COL = "is_delayed"

    def __init__(
        self,
        input_path=None,
        test_size=0.2,
        random_state=42,
        sample_frac=None,
    ):
        """
        Parameters:
        - input_path (str or Path): Path to the processed flights parquet file.
        - test_size (float): Proporção do conjunto de teste.
        - random_state (int): Semente para reprodutibilidade.
        - sample_frac (float or None): Fração do dataset usada no treino.
          None ou 1.0 = dataset completo (~5.7M linhas — pode travar PCs com pouca RAM).
          Use 0.1 para iterar rápido durante desenvolvimento.
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
        self.log_reg_pipeline = None
        self.rf_pipeline = None

        # Resultados
        self.y_pred_lr = None
        self.y_prob_lr = None
        self.y_pred_rf = None
        self.y_prob_rf = None
        self.metrics_lr = None
        self.metrics_rf = None
        self.comparison_df = None

    # ------------------------------------------------------------------
    # 1. Carregamento e seleção de features
    # ------------------------------------------------------------------
    def load_data(self):
        """Carrega o parquet processado e seleciona apenas features + target."""
        print(f"Carregando dados de: {self.input_path} ...")
        print(f"Existe? {self.input_path.exists()}")

        self.df = pl.read_parquet(self.input_path)
        print(f"Shape original: {self.df.shape}")

        self.df_model = self.df.select(self.FEATURE_COLS + [self.TARGET_COL])
        print(f"Shape após seleção de features: {self.df_model.shape}")
        return self.df_model

    # ------------------------------------------------------------------
    # 2. Preparação de X / y e split
    # ------------------------------------------------------------------
    def prepare_features(self):
        """Aplica amostragem, converte para pandas e separa X / y."""
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
        """Train/test split estratificado."""
        if self.X is None:
            self.prepare_features()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )

        print(f"X_train: {self.X_train.shape}")
        print(f"X_test:  {self.X_test.shape}")
        print(f"y_train: {self.y_train.shape}")
        print(f"y_test:  {self.y_test.shape}")

    # ------------------------------------------------------------------
    # 3. Pré-processamento
    # ------------------------------------------------------------------
    def build_preprocessor(self):
        """Monta o ColumnTransformer com imputação + scaler + OHE."""
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.NUMERIC_FEATURES),
                ("cat", categorical_transformer, self.CATEGORICAL_FEATURES),
            ]
        )
        return self.preprocessor

    # ------------------------------------------------------------------
    # 4. Treinamento
    # ------------------------------------------------------------------
    def train_logistic_regression(self):
        """Treina a Regressão Logística."""
        if self.preprocessor is None:
            self.build_preprocessor()
        if self.X_train is None:
            self.split_data()

        print("\n=== Treinando Logistic Regression ===")
        self.log_reg_pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=self.random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self.log_reg_pipeline.fit(self.X_train, self.y_train)
        self.y_pred_lr = self.log_reg_pipeline.predict(self.X_test)
        self.y_prob_lr = self.log_reg_pipeline.predict_proba(self.X_test)[:, 1]
        return self.log_reg_pipeline

    def train_random_forest(self):
        """Treina o Random Forest."""
        if self.preprocessor is None:
            self.build_preprocessor()
        if self.X_train is None:
            self.split_data()

        print("\n=== Treinando Random Forest ===")
        self.rf_pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=12,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        random_state=self.random_state,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self.rf_pipeline.fit(self.X_train, self.y_train)
        self.y_pred_rf = self.rf_pipeline.predict(self.X_test)
        self.y_prob_rf = self.rf_pipeline.predict_proba(self.X_test)[:, 1]
        return self.rf_pipeline

    # ------------------------------------------------------------------
    # 5. Avaliação
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_metrics(y_true, y_pred, y_prob):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob),
        }

    def evaluate(self):
        """Calcula métricas dos dois modelos e monta DataFrame comparativo."""
        if self.y_pred_lr is None:
            self.train_logistic_regression()
        if self.y_pred_rf is None:
            self.train_random_forest()

        self.metrics_lr = self._compute_metrics(
            self.y_test, self.y_pred_lr, self.y_prob_lr
        )
        self.metrics_rf = self._compute_metrics(
            self.y_test, self.y_pred_rf, self.y_prob_rf
        )

        self.comparison_df = pd.DataFrame(
            [
                {"modelo": "Logistic Regression", **self.metrics_lr},
                {"modelo": "Random Forest", **self.metrics_rf},
            ]
        )

        print("\n=== Comparação de modelos ===")
        print(self.comparison_df.to_string(index=False))
        return self.comparison_df

    # ------------------------------------------------------------------
    # 6. Gráficos
    # ------------------------------------------------------------------
    def plot_comparison(self):
        """Gráfico de barras comparando métricas dos modelos."""
        if self.comparison_df is None:
            self.evaluate()

        plot_df = self.comparison_df.set_index("modelo")
        plot_df.plot(kind="bar", figsize=(10, 5))
        plt.title("Comparação de modelos")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    def plot_roc_and_confusion(self):
        """Curva ROC e matrizes de confusão lado a lado."""
        if self.metrics_lr is None or self.metrics_rf is None:
            self.evaluate()

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # Curva ROC
        ax = axes[0]
        fpr_lr, tpr_lr, _ = roc_curve(self.y_test, self.y_prob_lr)
        fpr_rf, tpr_rf, _ = roc_curve(self.y_test, self.y_prob_rf)
        ax.plot(fpr_lr, tpr_lr, label=f"LR (AUC={self.metrics_lr['roc_auc']:.3f})")
        ax.plot(fpr_rf, tpr_rf, label=f"RF (AUC={self.metrics_rf['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Curva ROC")
        ax.legend()
        ax.grid(alpha=0.3)

        # Confusion Matrix — LR
        ConfusionMatrixDisplay(
            confusion_matrix(self.y_test, self.y_pred_lr),
            display_labels=["Não atrasado", "Atrasado"],
        ).plot(ax=axes[1], colorbar=False, cmap="Blues")
        axes[1].set_title("Confusion Matrix — Logistic Regression")

        # Confusion Matrix — RF
        ConfusionMatrixDisplay(
            confusion_matrix(self.y_test, self.y_pred_rf),
            display_labels=["Não atrasado", "Atrasado"],
        ).plot(ax=axes[2], colorbar=False, cmap="Oranges")
        axes[2].set_title("Confusion Matrix — Random Forest")

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        """Feature importance do Random Forest, agregada por feature original."""
        if self.rf_pipeline is None:
            self.train_random_forest()

        ohe = (
            self.rf_pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )
        cat_feature_names = ohe.get_feature_names_out(
            self.CATEGORICAL_FEATURES
        ).tolist()
        all_feature_names = self.NUMERIC_FEATURES + cat_feature_names

        importances = self.rf_pipeline.named_steps["model"].feature_importances_

        fi_df = pd.DataFrame(
            {"feature": all_feature_names, "importance": importances}
        )

        def base_feature(name):
            for f in self.CATEGORICAL_FEATURES:
                if name.startswith(f):
                    return f
            return name

        fi_df["feature_base"] = fi_df["feature"].apply(base_feature)
        fi_agg = (
            fi_df.groupby("feature_base")["importance"]
            .sum()
            .sort_values(ascending=True)
        )

        fi_agg.plot(kind="barh", figsize=(8, 5), color="steelblue")
        plt.title("Feature Importance — Random Forest")
        plt.xlabel("Importância total")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------
    def run_all(self, show_plots=True):
        """Executa o pipeline completo de classificação."""
        self.load_data()
        self.prepare_features()
        self.split_data()
        self.build_preprocessor()
        self.train_logistic_regression()
        self.train_random_forest()
        self.evaluate()
        if show_plots:
            self.plot_comparison()
            self.plot_roc_and_confusion()
            self.plot_feature_importance()


if __name__ == "__main__":
    clf = FlightClassifier(
        input_path="../data/processed/flights_model.parquet",
        sample_frac=0.1,
    )
    clf.run_all(show_plots=True)