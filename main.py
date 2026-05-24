import sys
import pickle
from pathlib import Path

# Adiciona o diretório raiz do projeto ao PYTHONPATH para garantir que o python encontre os módulos
sys.path.append(str(Path(__file__).resolve().parent))

from src.eda import FlightEDA
from src.classificacao import FlightClassifier
from src.regressao import FlightRegressor
from src.clusterizacao import FlightClusterer
from src.mapa_geografico import FlightGeoMapper
from src.anomalias import FlightAnomalyDetector
from src.sazonalidade import FlightSeasonality

from reports import generate_report


# Caminhos centralizados
RAW_DATA = "data/raw/flights.csv"
PROCESSED_DATA = "data/processed/flights_model.parquet"
REPORT_PATH = "outputs/relatorio.html"
PICKLES_DIR = Path("outputs/pickles")


def save_pickles(eda, clf, reg, clusterer, mapper, detector, season):
    """
    Salva todas as instâncias treinadas em pickle pra serem reutilizadas pelo Streamlit.

    IMPORTANTE: Antes de salvar, esvaziamos os atributos com DataFrames grandes
    que NAO sao usados pelo dashboard. Isso evita estourar memoria durante o
    pickle.dump (DataFrames precisam ser materializados em RAM).
    """
    import gc
    PICKLES_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("  Salvando objetos treinados em pickle...")
    print("  (esvaziando DataFrames grandes pra evitar estouro de memória)")
    print("=" * 80)

    # ---------------- EDA ----------------
    # Guarda metadados pequenos que o dashboard usa antes de descartar o DF
    eda._meta_n_voos = eda.df_model.height
    eda._meta_n_companhias = eda.df_model["AIRLINE"].n_unique()
    eda._meta_n_aeroportos = eda.df_model["ORIGIN_AIRPORT"].n_unique()
    eda._meta_taxa_atraso = eda.df_model.select(
        (eda.df_model["ARRIVAL_DELAY"] > 0).mean()
    ).item()

    # Esvazia DataFrames grandes
    df_backup_eda = (eda.df, eda.df_model)
    eda.df = None
    eda.df_model = None
    gc.collect()

    with open(PICKLES_DIR / "eda.pkl", "wb") as f:
        pickle.dump(eda, f)
    print(f"  ✓ eda.pkl ({(PICKLES_DIR / 'eda.pkl').stat().st_size / 1024:.1f} KB)")
    eda.df, eda.df_model = df_backup_eda  # restaura

    # ---------------- Classifier ----------------
    # Salva listas únicas pra dropdowns antes de descartar X
    if clf.X is not None:
        clf._airlines = sorted(clf.X["AIRLINE"].unique().tolist())
        clf._origins = sorted(clf.X["ORIGIN_AIRPORT"].unique().tolist())
        clf._destinations = sorted(clf.X["DESTINATION_AIRPORT"].unique().tolist())

    clf_backup = (clf.df, clf.df_model, clf.X, clf.y, clf.X_train, clf.X_test, clf.y_train, clf.y_test)
    clf.df = None
    clf.df_model = None
    clf.X = None
    clf.y = None
    clf.X_train = None
    clf.X_test = None
    clf.y_train = None
    clf.y_test = None
    gc.collect()

    with open(PICKLES_DIR / "clf.pkl", "wb") as f:
        pickle.dump(clf, f)
    print(f"  ✓ clf.pkl ({(PICKLES_DIR / 'clf.pkl').stat().st_size / 1024 / 1024:.2f} MB)")
    (clf.df, clf.df_model, clf.X, clf.y, clf.X_train, clf.X_test, clf.y_train, clf.y_test) = clf_backup

    # ---------------- Regressor ----------------
    reg_backup = (reg.df, reg.df_model, reg.X, reg.y, reg.X_train, reg.X_test, reg.y_train, reg.y_test)
    reg.df = None
    reg.df_model = None
    reg.X = None
    reg.y = None
    reg.X_train = None
    reg.X_test = None
    reg.y_train = None
    reg.y_test = None
    gc.collect()

    with open(PICKLES_DIR / "reg.pkl", "wb") as f:
        pickle.dump(reg, f)
    print(f"  ✓ reg.pkl ({(PICKLES_DIR / 'reg.pkl').stat().st_size / 1024 / 1024:.2f} MB)")
    (reg.df, reg.df_model, reg.X, reg.y, reg.X_train, reg.X_test, reg.y_train, reg.y_test) = reg_backup

    # ---------------- Clusterer ----------------
    # CORREÇÃO: Garante o backup e a limpeza de df_model e df para não vazar os 5.7M de linhas
    clusterer_df_backup = getattr(clusterer, "df", None)
    clusterer_model_backup = getattr(clusterer, "df_model", None)
    
    clusterer.df = None
    if hasattr(clusterer, "df_model"):
        clusterer.df_model = None
    gc.collect()

    with open(PICKLES_DIR / "clusterer.pkl", "wb") as f:
        pickle.dump(clusterer, f)
    print(f"  ✓ clusterer.pkl ({(PICKLES_DIR / 'clusterer.pkl').stat().st_size / 1024:.1f} KB)")
    
    # Restaura os dados do clusterer
    if clusterer_df_backup is not None:
        clusterer.df = clusterer_df_backup
    if clusterer_model_backup is not None:
        clusterer.df_model = clusterer_model_backup

    # ---------------- Detector ----------------
    # AJUSTE: Garante que limpa o df_model se ele tiver sido criado nessa classe
    det_df_backup = detector.df
    det_model_backup = getattr(detector, "df_model", None)
    
    detector.df = None
    if hasattr(detector, "df_model"):
        detector.df_model = None
    gc.collect()

    with open(PICKLES_DIR / "detector.pkl", "wb") as f:
        pickle.dump(detector, f)
    print(f"  ✓ detector.pkl ({(PICKLES_DIR / 'detector.pkl').stat().st_size / 1024:.1f} KB)")
    
    detector.df = det_df_backup
    if det_model_backup is not None:
        detector.df_model = det_model_backup

    # ---------------- Season ----------------
    # AJUSTE: Garante que limpa o df_model e restaura tudo ao final de forma limpa
    season_df_backup = season.df
    season_model_backup = getattr(season, "df_model", None)
    
    season.df = None
    if hasattr(season, "df_model"):
        season.df_model = None
    gc.collect()

    with open(PICKLES_DIR / "season.pkl", "wb") as f:
        pickle.dump(season, f)
    print(f"  ✓ season.pkl ({(PICKLES_DIR / 'season.pkl').stat().st_size / 1024:.1f} KB)")
    
    season.df = season_df_backup
    if season_model_backup is not None:
        season.df_model = season_model_backup

    # ---------------- Mapper ----------------
    # AJUSTE: Garante a limpeza e a correta restauração do df_model para evitar lixo em memória
    folium_map = mapper.folium_map
    mapper_df_backup = mapper.df
    mapper_model_backup = getattr(mapper, "df_model", None)
    mapper_routes_backup = mapper.routes
    
    mapper.folium_map = None
    mapper.df = None
    mapper.routes = None  
    if hasattr(mapper, "df_model"):
        mapper.df_model = None
    gc.collect()

    with open(PICKLES_DIR / "mapper.pkl", "wb") as f:
        pickle.dump(mapper, f)
    print(f"  ✓ mapper.pkl ({(PICKLES_DIR / 'mapper.pkl').stat().st_size / 1024 / 1024:.2f} MB)")

    mapper.folium_map = folium_map
    mapper.df = mapper_df_backup
    mapper.routes = mapper_routes_backup
    if mapper_model_backup is not None:
        mapper.df_model = mapper_model_backup

    # Salva o HTML do folium separadamente
    if folium_map is not None:
        folium_html_path = PICKLES_DIR / "folium_map.html"
        folium_map.save(str(folium_html_path))
        print(f"  ✓ folium_map.html ({folium_html_path.stat().st_size / 1024:.1f} KB)")

    print(f"\n  Pickles salvos em: {PICKLES_DIR.resolve()}")

def run_eda(show_plots=True):
    """1. Análise Exploratória — gera o parquet processado consumido pelas demais etapas."""
    print("\n" + "=" * 80)
    print("ETAPA 1 — Análise Exploratória de Dados (EDA)")
    print("=" * 80)

    eda = FlightEDA(input_path=RAW_DATA, output_path=PROCESSED_DATA)

    # Se o parquet já existe, roda EDA em memória sem salvar de novo
    if Path(PROCESSED_DATA).exists():
        print(f"✓ Parquet processado já existe em {PROCESSED_DATA}")
        print("  Rodando EDA em memória sem salvar (já temos o arquivo).")
        eda.load_data()
        eda.print_basic_info()
        eda.preprocess_and_filter()
        eda.run_aggregations()
        if show_plots:
            eda.plot_charts()
        return eda

    eda.run_all(show_plots=show_plots)
    return eda


def run_classification(show_plots=True):
    """2. Modelagem supervisionada — classificação (atrasa ou não)."""
    print("\n" + "=" * 80)
    print("ETAPA 2 — Modelagem Supervisionada: Classificação")
    print("=" * 80)

    clf = FlightClassifier(
        input_path=PROCESSED_DATA,
        sample_frac=0.1,
    )
    clf.run_all(show_plots=show_plots)
    return clf


def run_regression(show_plots=True):
    """3. Modelagem supervisionada — regressão (quantos minutos de atraso)."""
    print("\n" + "=" * 80)
    print("ETAPA 3 — Modelagem Supervisionada: Regressão")
    print("=" * 80)

    reg = FlightRegressor(
        input_path=PROCESSED_DATA,
        sample_frac=0.3,
    )
    reg.run_all(show_plots=show_plots)
    return reg


def run_clustering(show_plots=True):
    """4. Modelagem não supervisionada — clusterização de aeroportos."""
    print("\n" + "=" * 80)
    print("ETAPA 4 — Modelagem Não Supervisionada: Clusterização")
    print("=" * 80)

    clusterer = FlightClusterer(input_path=PROCESSED_DATA)
    clusterer.run_all(show_plots=show_plots)
    return clusterer


def run_geomap(show_plots=True):
    """5. Visualização geográfica — mapas de aeroportos, rotas e atrasos."""
    print("\n" + "=" * 80)
    print("ETAPA 5 — Visualização Geográfica")
    print("=" * 80)

    mapper = FlightGeoMapper(input_path=PROCESSED_DATA)
    mapper.run_all(show_plots=show_plots)
    return mapper


def run_anomalies(show_plots=True):
    """6. Detecção de anomalias — aeroportos com perfil atípico."""
    print("\n" + "=" * 80)
    print("ETAPA 6 — Detecção de Anomalias")
    print("=" * 80)

    detector = FlightAnomalyDetector(input_path=PROCESSED_DATA)
    detector.run_all(show_plots=show_plots)
    return detector


def run_seasonality(anomalous_airports=None, show_plots=True):
    """7. Análise de sazonalidade — padrões mensais, semanais e horários."""
    print("\n" + "=" * 80)
    print("ETAPA 7 — Análise de Sazonalidade")
    print("=" * 80)

    season = FlightSeasonality(
        input_path=PROCESSED_DATA,
        anomalous_airports=anomalous_airports,
    )
    season.run_all(show_plots=show_plots)
    return season


def main(show_plots=False, generate_html_report=True, save_artifacts=True):
    """
    Executa o pipeline completo.

    Parameters
    ----------
    show_plots : bool
        Se True, exibe gráficos via plt.show() (interrompe execução em terminal).
        Se False (default), pula a exibição interativa e só gera o HTML no final.
    generate_html_report : bool
        Se True (default), gera o relatório HTML consolidado ao final.
    save_artifacts : bool
        Se True (default), salva os objetos treinados em pickle para uso pelo dashboard.
    """
    print("=" * 80)
    print("  PIPELINE COMPLETO — Tech Challenge: Atrasos de Voos nos EUA (2015)")
    print("=" * 80)

    eda = run_eda(show_plots=show_plots)
    clf = run_classification(show_plots=show_plots)
    reg = run_regression(show_plots=show_plots)
    clusterer = run_clustering(show_plots=show_plots)
    mapper = run_geomap(show_plots=show_plots)
    detector = run_anomalies(show_plots=show_plots)
    anomalous_airports = detector.anomalos_consenso.index.tolist()
    print(f"\nAeroportos anômalos detectados: {anomalous_airports}")
    season = run_seasonality(
        anomalous_airports=anomalous_airports, show_plots=show_plots
    )

    if generate_html_report:
        generate_report(
            eda=eda,
            clf=clf,
            reg=reg,
            clusterer=clusterer,
            mapper=mapper,
            detector=detector,
            season=season,
            output_path=REPORT_PATH,
        )

    if save_artifacts:
        save_pickles(eda, clf, reg, clusterer, mapper, detector, season)

    print("\n" + "=" * 80)
    print("  ✓ Pipeline finalizado com sucesso")
    print("=" * 80)


if __name__ == "__main__":
    # Por padrão: roda sem gráficos interativos e gera o HTML no final
    main(show_plots=False, generate_html_report=True)
