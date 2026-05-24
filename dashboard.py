"""
Dashboard interativo do Tech Challenge — Atrasos de Voos.

Execute com:
    streamlit run dashboard.py

O dashboard carrega os objetos treinados de outputs/pickles/ se existirem.
Se não existirem, executa o pipeline completo (main.py) na primeira execução.
"""

import sys
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Adiciona raiz do projeto ao PYTHONPATH
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

PICKLES_DIR = ROOT / "outputs" / "pickles"
FOLIUM_HTML_PATH = PICKLES_DIR / "folium_map.html"


# ============================================================================
# Configuração da página
# ============================================================================
st.set_page_config(
    page_title="Tech Challenge — Atrasos de Voos",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Loaders (com cache do Streamlit)
# ============================================================================
@st.cache_resource(show_spinner="Carregando modelos treinados...")
def load_artifacts():
    """
    Carrega objetos treinados de pickle. Se não existir, executa o pipeline.

    Returns
    -------
    dict com chaves: eda, clf, reg, clusterer, mapper, detector, season
    """
    expected_files = ["eda", "clf", "reg", "clusterer", "mapper", "detector", "season"]
    missing = [f for f in expected_files if not (PICKLES_DIR / f"{f}.pkl").exists()]

    if missing:
        st.warning(
            f"⚠ Pickles não encontrados: {missing}. "
            "Executando o pipeline completo... isso pode demorar alguns minutos."
        )
        from main import main as run_pipeline
        run_pipeline(
            show_plots=False,
            generate_html_report=False,
            save_artifacts=True,
        )

    artifacts = {}
    for name in expected_files:
        with open(PICKLES_DIR / f"{name}.pkl", "rb") as f:
            artifacts[name] = pickle.load(f)

    return artifacts


# ============================================================================
# Sidebar de navegação
# ============================================================================
st.sidebar.title("✈️ Atrasos de Voos")
st.sidebar.markdown("**Tech Challenge — FIAP**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegação",
    [
        "🏠 Visão Geral",
        "📊 Análise Exploratória",
        "🤖 Modelos de ML",
        "🎯 Predição Interativa",
        "🗺️ Mapas Geográficos",
        "🔍 Anomalias",
        "📅 Sazonalidade",
        "ℹ️ Sobre o Projeto",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: U.S. DOT — 2015")
st.sidebar.caption("5.7M voos · 14 companhias · 628 aeroportos")


# ============================================================================
# Carrega artefatos
# ============================================================================
artifacts = load_artifacts()
eda = artifacts["eda"]
clf = artifacts["clf"]
reg = artifacts["reg"]
clusterer = artifacts["clusterer"]
mapper = artifacts["mapper"]
detector = artifacts["detector"]
season = artifacts["season"]


# ============================================================================
# PÁGINA: Visão Geral
# ============================================================================
def page_overview():
    st.title("✈️ Análise de Atrasos em Voos Domésticos")
    st.markdown("### Dashboard interativo — Tech Challenge FIAP")
    st.markdown("---")

    # KPIs em colunas
    col1, col2, col3, col4 = st.columns(4)

    # Lê dos atributos _meta_ salvos pelo save_pickles (DataFrame original foi descartado)
    n_voos = getattr(eda, "_meta_n_voos", 0)
    n_companhias = getattr(eda, "_meta_n_companhias", 0)
    n_aeroportos = getattr(eda, "_meta_n_aeroportos", 0)
    taxa_atraso_geral = getattr(eda, "_meta_taxa_atraso", 0)

    col1.metric("Total de voos", f"{n_voos:,}".replace(",", "."))
    col2.metric("Companhias", n_companhias)
    col3.metric("Aeroportos", n_aeroportos)
    col4.metric("Taxa de atraso", f"{taxa_atraso_geral:.1%}")

    st.markdown("---")

    # 4 perguntas centrais
    st.subheader("📋 Perguntas centrais respondidas pela análise")

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**1. É possível prever se um voo vai atrasar?**\n\n"
            f"Sim, com ROC-AUC ~{clf.comparison_df['roc_auc'].max():.3f}. "
            "Útil como sinal probabilístico."
        )
        st.info(
            "**3. Quais aeroportos têm perfis semelhantes?**\n\n"
            f"Identificamos **{clusterer.k_final} perfis** distintos: "
            "hubs grandes, regionais tranquilos e intermediários."
        )

    with col2:
        st.warning(
            "**2. Quanto tempo de atraso podemos esperar?**\n\n"
            f"Não com os dados atuais. R² máximo = {reg.comparison_df['R2'].max():.3f} — "
            "faltam variáveis externas (clima, congestionamento)."
        )
        st.info(
            "**4. Quando e onde os atrasos acontecem?**\n\n"
            "**Junho e dezembro**, em **noites de quinta e sexta**. "
            "Corredor nordeste dos EUA é o pior geograficamente."
        )

    st.markdown("---")

    # Gráfico mensal
    st.subheader("📈 Padrão mensal de atrasos")
    if season.monthly is not None:
        chart_df = season.monthly.set_index("mes_nome")[
            ["taxa_atraso", "media_delay"]
        ]
        chart_df.columns = ["Taxa de atraso", "Delay médio (min)"]
        st.line_chart(chart_df)

    # Distribuição por período
    st.subheader("⏰ Distribuição por período do dia")
    if eda.delay_by_period is not None:
        period_df = eda.delay_by_period.to_pandas()
        st.bar_chart(period_df.set_index("periodo_dia")["taxa_atraso"])


# ============================================================================
# PÁGINA: EDA
# ============================================================================
def page_eda():
    st.title("📊 Análise Exploratória")
    st.markdown(
        "Estatísticas descritivas, distribuições e tratamento de valores ausentes."
    )

    tab1, tab2, tab3 = st.tabs(["Por Companhia", "Por Período", "Por Aeroporto"])

    with tab1:
        st.subheader("Taxa de atraso por companhia aérea")
        if eda.delay_by_airline is not None:
            df = eda.delay_by_airline.to_pandas()

            # Slider pra filtrar quantas mostrar
            n_show = st.slider("Quantas companhias mostrar", 5, len(df), 10)

            df_show = df.head(n_show).copy()
            df_show["taxa_atraso"] = (df_show["taxa_atraso"] * 100).round(2)
            df_show.columns = [
                "Companhia",
                "Voos",
                "Delay médio (min)",
                "Delay partida (min)",
                "Taxa atraso (%)",
            ]
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            st.bar_chart(df_show.set_index("Companhia")["Taxa atraso (%)"])

    with tab2:
        st.subheader("Distribuição por período do dia")
        if eda.delay_by_period is not None:
            df = eda.delay_by_period.to_pandas()
            df.columns = ["Período", "Voos", "Taxa atraso", "Delay médio (min)"]
            df["Taxa atraso"] = (df["Taxa atraso"] * 100).round(2)
            df["Delay médio (min)"] = df["Delay médio (min)"].round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(df.set_index("Período")["Taxa atraso"])
            with col2:
                st.bar_chart(df.set_index("Período")["Voos"])

    with tab3:
        st.subheader("Top aeroportos com maior atraso")
        if eda.delay_by_origin is not None:
            df = eda.delay_by_origin.to_pandas()
            n_show = st.slider("Top N aeroportos", 5, 30, 15, key="airport_slider")
            df_show = df.head(n_show).copy()
            df_show["taxa_atraso"] = (df_show["taxa_atraso"] * 100).round(2)
            df_show.columns = ["Aeroporto", "Voos", "Delay médio (min)", "Taxa atraso (%)"]
            st.dataframe(df_show, use_container_width=True, hide_index=True)


# ============================================================================
# PÁGINA: Modelos
# ============================================================================
def page_models():
    st.title("🤖 Modelos de Machine Learning")

    tab1, tab2 = st.tabs(["Classificação", "Regressão"])

    with tab1:
        st.subheader("Classificação: o voo vai atrasar?")
        st.markdown(
            "Comparação entre **Logistic Regression** (linear) e **Random Forest** "
            "(ensemble não-linear)."
        )

        df_clf = clf.comparison_df.copy()
        for col in df_clf.columns:
            if col != "modelo":
                df_clf[col] = df_clf[col].round(4)
        st.dataframe(df_clf, use_container_width=True, hide_index=True)

        # Gráfico de barras das métricas
        chart_df = df_clf.set_index("modelo")
        st.bar_chart(chart_df)

        best_idx = clf.comparison_df["roc_auc"].idxmax()
        best = clf.comparison_df.loc[best_idx, "modelo"]
        st.success(
            f"🏆 **{best}** apresentou o melhor desempenho "
            f"(ROC-AUC = {clf.comparison_df.loc[best_idx, 'roc_auc']:.4f})"
        )

    with tab2:
        st.subheader("Regressão: quantos minutos de atraso?")
        st.markdown(
            "Comparação entre **Linear Regression**, **Decision Tree** e **LightGBM** "
            "(complexidade crescente)."
        )

        df_reg = reg.comparison_df.copy().round(4)
        st.dataframe(df_reg, use_container_width=True)

        # Tempos de treino
        col1, col2, col3 = st.columns(3)
        col1.metric("Linear Regression", f"{reg.lr_time:.2f}s", f"R²={reg.metrics_lr['R2']:.4f}")
        col2.metric("Decision Tree", f"{reg.dt_time:.2f}s", f"R²={reg.metrics_dt['R2']:.4f}")
        col3.metric("LightGBM 🏆", f"{reg.gb_time:.2f}s", f"R²={reg.metrics_gb['R2']:.4f}")

        st.warning(
            f"⚠ **R² máximo = {reg.comparison_df['R2'].max():.4f}** — todos os modelos "
            "têm desempenho baixo. Isso indica que o tempo exato de atraso depende "
            "fortemente de variáveis externas (clima, congestionamento, efeito cascata) "
            "que não estão presentes no dataset."
        )


# ============================================================================
# PÁGINA: Predição Interativa
# ============================================================================
def page_predict():
    st.title("🎯 Predição Interativa")
    st.markdown(
        "Preencha as informações de um voo hipotético e veja a **probabilidade de atraso** "
        "calculada pelo modelo treinado."
    )

    # Pega listas de opções dos atributos pré-salvos (clf.X foi descartado pra economizar memória)
    if not hasattr(clf, "_airlines"):
        st.error("Modelo não está pronto. Tente recarregar o dashboard.")
        return

    airlines = clf._airlines
    origins = clf._origins
    destinations = clf._destinations

    st.markdown("### Dados do voo")

    col1, col2 = st.columns(2)

    with col1:
        month = st.selectbox(
            "Mês",
            options=list(range(1, 13)),
            format_func=lambda m: [
                "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                "Jul", "Ago", "Set", "Out", "Nov", "Dez"
            ][m - 1],
            index=5,  # Junho
        )
        day = st.number_input("Dia do mês", min_value=1, max_value=31, value=15)
        day_of_week = st.selectbox(
            "Dia da semana",
            options=list(range(1, 8)),
            format_func=lambda d: ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"][d - 1],
            index=3,  # Quinta
        )
        airline = st.selectbox("Companhia aérea", options=airlines, index=0)

    with col2:
        origin = st.selectbox(
            "Aeroporto de origem",
            options=origins,
            index=origins.index("ATL") if "ATL" in origins else 0,
        )
        destination = st.selectbox(
            "Aeroporto de destino",
            options=destinations,
            index=destinations.index("LAX") if "LAX" in destinations else 0,
        )

        hora_partida = st.slider(
            "Horário programado de partida",
            min_value=0, max_value=23, value=20,
            help="Hora cheia (formato 24h)",
        )
        scheduled_departure = hora_partida * 100  # HHMM format

        scheduled_time = st.number_input(
            "Duração planejada (min)", min_value=30, max_value=600, value=240, step=10
        )
        distance = st.number_input(
            "Distância (milhas)", min_value=50, max_value=5000, value=1500, step=50
        )

    # Calcula período do dia
    if hora_partida < 6:
        periodo_dia = "madrugada"
    elif hora_partida < 12:
        periodo_dia = "manha"
    elif hora_partida < 18:
        periodo_dia = "tarde"
    else:
        periodo_dia = "noite"

    st.markdown(f"**Período do dia computado:** `{periodo_dia}`")

    st.markdown("---")

    if st.button("🔮 Calcular probabilidade de atraso", type="primary", use_container_width=True):
        # Monta o DataFrame na ordem esperada pelo modelo
        X_pred = pd.DataFrame([{
            "MONTH": month,
            "DAY": day,
            "DAY_OF_WEEK": day_of_week,
            "AIRLINE": airline,
            "ORIGIN_AIRPORT": origin,
            "DESTINATION_AIRPORT": destination,
            "SCHEDULED_DEPARTURE": scheduled_departure,
            "SCHEDULED_TIME": scheduled_time,
            "DISTANCE": distance,
            "periodo_dia": periodo_dia,
        }])

        try:
            # Logistic Regression
            prob_lr = clf.log_reg_pipeline.predict_proba(X_pred)[0, 1]
            # Random Forest
            prob_rf = clf.rf_pipeline.predict_proba(X_pred)[0, 1]
            # Média dos modelos
            prob_ensemble = (prob_lr + prob_rf) / 2

            # Predição de minutos
            minutos_pred = reg.gb_pipeline.predict(X_pred)[0]

            st.markdown("### 📊 Resultados")

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Logistic Regression",
                f"{prob_lr:.1%}",
                help="Probabilidade do voo atrasar",
            )
            col2.metric(
                "Random Forest",
                f"{prob_rf:.1%}",
            )
            col3.metric(
                "Média (ensemble)",
                f"{prob_ensemble:.1%}",
            )

            st.markdown("---")

            # Visualização gráfica
            if prob_ensemble < 0.30:
                st.success(f"✅ **Baixo risco de atraso** ({prob_ensemble:.1%})")
            elif prob_ensemble < 0.45:
                st.info(f"ℹ️ **Risco moderado** ({prob_ensemble:.1%})")
            elif prob_ensemble < 0.60:
                st.warning(f"⚠️ **Alto risco de atraso** ({prob_ensemble:.1%})")
            else:
                st.error(f"🚨 **Risco muito alto de atraso** ({prob_ensemble:.1%})")

            st.markdown(
                f"**Tempo de atraso esperado (LightGBM):** "
                f"`{minutos_pred:+.1f}` minutos"
            )
            st.caption(
                "⚠ Lembrete: o modelo de regressão tem R² baixo (~0.06). "
                "Use essa estimativa apenas como referência grosseira — a margem de erro é alta."
            )

        except Exception as e:
            st.error(f"Erro na predição: {e}")
            st.exception(e)


# ============================================================================
# PÁGINA: Mapas
# ============================================================================
def page_maps():
    st.title("🗺️ Visualização Geográfica")

    map_choice = st.selectbox(
        "Selecione o mapa",
        [
            "Aeroportos por taxa de atraso",
            "Aeroportos por cluster",
            "Top rotas com atraso médio",
            "Choropleth por estado",
            "Heatmap interativo (Folium)",
        ],
    )

    if map_choice == "Aeroportos por taxa de atraso":
        if mapper.fig_delay_rate is not None:
            st.plotly_chart(mapper.fig_delay_rate, use_container_width=True)
        else:
            st.warning("Mapa não disponível.")

    elif map_choice == "Aeroportos por cluster":
        if mapper.fig_clusters is not None:
            st.plotly_chart(mapper.fig_clusters, use_container_width=True)

    elif map_choice == "Top rotas com atraso médio":
        if mapper.fig_routes is not None:
            st.plotly_chart(mapper.fig_routes, use_container_width=True)

    elif map_choice == "Choropleth por estado":
        if mapper.fig_state_choropleth is not None:
            st.plotly_chart(mapper.fig_state_choropleth, use_container_width=True)

        if mapper.state_stats is not None:
            st.subheader("Top 10 estados por taxa de atraso")
            df = mapper.state_stats.sort_values("taxa_atraso", ascending=False).head(10).copy()
            df["taxa_atraso"] = (df["taxa_atraso"] * 100).round(2)
            df["media_delay"] = df["media_delay"].round(2)
            df.columns = ["Estado", "Taxa atraso (%)", "Delay médio (min)", "Aeroportos"]
            st.dataframe(df, use_container_width=True, hide_index=True)

    elif map_choice == "Heatmap interativo (Folium)":
        if FOLIUM_HTML_PATH.exists():
            with open(FOLIUM_HTML_PATH, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600)
        else:
            st.warning(
                "Heatmap Folium não encontrado. "
                f"Esperado em: {FOLIUM_HTML_PATH}"
            )


# ============================================================================
# PÁGINA: Anomalias
# ============================================================================
def page_anomalies():
    st.title("🔍 Detecção de Anomalias")
    st.markdown(
        "Aeroportos com **perfil operacional atípico** identificados por consenso "
        "entre três métodos: Isolation Forest, LOF e Silhouette individual."
    )

    if detector.anomalos_consenso is None or detector.anomalos_consenso.empty:
        st.warning("Nenhuma anomalia detectada.")
        return

    n_anomalos = len(detector.anomalos_consenso)
    st.metric("Aeroportos anômalos detectados", n_anomalos)

    # Filtro de número de métodos
    min_methods = st.radio(
        "Mínimo de métodos que sinalizaram",
        [1, 2, 3],
        index=1,
        horizontal=True,
        help="2 é o padrão (consenso). 3 = só os mais óbvios.",
    )

    df_filtered = detector.airport_features[
        detector.airport_features["n_metodos_anomalia"] >= min_methods
    ].copy()

    st.markdown(f"**{len(df_filtered)} aeroportos** com pelo menos {min_methods} método(s) sinalizando.")

    cols_show = [
        "n_metodos_anomalia",
        "cluster",
        "taxa_atraso",
        "taxa_atraso_grave",
        "media_arrival_delay",
        "std_arrival_delay",
        "n_destinos",
        "qtd_voos",
    ]
    df_display = df_filtered[cols_show].round(3).reset_index().copy()
    df_display.columns = [
        "Aeroporto", "Métodos", "Cluster", "Taxa atraso", "Taxa grave",
        "Delay médio (min)", "Desvio", "Destinos", "Voos",
    ]
    st.dataframe(df_display, use_container_width=True, hide_index=True)


# ============================================================================
# PÁGINA: Sazonalidade
# ============================================================================
def page_seasonality():
    st.title("📅 Análise de Sazonalidade")
    st.markdown("Quando os atrasos acontecem? Padrões mensais, semanais e horários.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Mensal", "Semanal", "Por período", "Cenários críticos"]
    )

    with tab1:
        st.subheader("Padrão mensal")
        if season.monthly is not None:
            df = season.monthly.copy()
            df["taxa_atraso_pct"] = (df["taxa_atraso"] * 100).round(2)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Taxa de atraso**")
                st.bar_chart(df.set_index("mes_nome")["taxa_atraso_pct"])
            with col2:
                st.markdown("**Delay médio (min)**")
                st.bar_chart(df.set_index("mes_nome")["media_delay"])

            st.dataframe(
                df[["mes_nome", "qtd_voos", "taxa_atraso_pct", "media_delay", "p90_delay"]].round(2),
                use_container_width=True, hide_index=True,
            )

    with tab2:
        st.subheader("Padrão por dia da semana")
        if season.weekly is not None:
            df = season.weekly.copy()
            df["taxa_atraso_pct"] = (df["taxa_atraso"] * 100).round(2)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Taxa de atraso (%)**")
                st.bar_chart(df.set_index("dia_nome")["taxa_atraso_pct"])
            with col2:
                st.markdown("**Volume de voos**")
                st.bar_chart(df.set_index("dia_nome")["qtd_voos"])

    with tab3:
        st.subheader("Padrão por período do dia")
        if season.period is not None:
            df = season.period.copy()
            df["taxa_atraso_pct"] = (df["taxa_atraso"] * 100).round(2)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Taxa de atraso (%)**")
                st.bar_chart(df.set_index("periodo_dia")["taxa_atraso_pct"])
            with col2:
                st.markdown("**Delay médio (min)**")
                st.bar_chart(df.set_index("periodo_dia")["media_delay"])

    with tab4:
        st.subheader("Top cenários críticos")
        st.caption("Combinações de Mês + Dia da semana + Período com piores taxas de atraso.")

        top_n = st.slider("Top N", 5, 25, 10)
        if season.critical is not None:
            df = season.critical.head(top_n)[
                ["combinacao", "taxa_atraso", "media_delay", "qtd_voos"]
            ].copy()
            df["taxa_atraso"] = (df["taxa_atraso"] * 100).round(2)
            df["media_delay"] = df["media_delay"].round(2)
            df.columns = ["Cenário", "Taxa atraso (%)", "Delay médio (min)", "Voos"]
            st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================================
# PÁGINA: Sobre
# ============================================================================
def page_about():
    st.title("ℹ️ Sobre o Projeto")

    st.markdown("""
    ### Tech Challenge — FIAP Postech Engenharia de Machine Learning

    **Objetivo:** análise completa de atrasos em voos domésticos nos Estados Unidos
    no ano de 2015, abrangendo todas as fases de um projeto de ciência de dados.

    ### Dataset
    - **Origem:** U.S. Department of Transportation — Bureau of Transportation Statistics
    - **Ano:** 2015
    - **Volume:** 5.714.008 voos
    - **Companhias:** 14 distintas
    - **Aeroportos:** 628

    ### Stack tecnológico
    - **Processamento:** Polars (DataFrame de alto desempenho)
    - **Modelagem supervisionada:** scikit-learn, LightGBM
    - **Modelagem não supervisionada:** scikit-learn (KMeans, IsolationForest, LOF, PCA)
    - **Visualização:** matplotlib, seaborn, plotly, folium
    - **Dashboard:** Streamlit
    - **Relatório:** Python + HTML/CSS

    ### Arquitetura
    O projeto é organizado em **7 classes Python independentes**, cada uma responsável
    por uma análise específica:

    ```
    notebooks/
    ├── eda.py              ← Análise exploratória
    ├── classificacao.py    ← Logistic Regression + Random Forest
    ├── regressao.py        ← Linear + Decision Tree + LightGBM
    ├── clusterizacao.py    ← KMeans + PCA
    ├── mapa_geografico.py  ← 5 mapas (Plotly + Folium)
    ├── anomalias.py        ← Isolation Forest + LOF + Silhouette
    └── sazonalidade.py     ← Padrões temporais multi-dimensão

    main.py                 ← Orquestrador
    dashboard.py            ← Este dashboard
    reports/                ← Gerador de relatório HTML estático
    ```

    ### Principais achados

    1. **Classificação:** ROC-AUC ~0.66 — previsibilidade moderada
    2. **Regressão:** R² baixo (≤ 0.06) — faltam features causais
    3. **Clusterização:** 3 perfis distintos identificados
    4. **Sazonalidade:** junho/dezembro à noite são os piores cenários
    5. **Anomalias:** aeroportos remotos (Alasca, Havaí) emergem como atípicos

    ### Limitações
    - Sem dados de clima (provavelmente o maior fator não modelado)
    - Sem efeito cascata (atraso anterior da mesma aeronave)
    - Apenas 1 ano de dados (2015)
    """)


# ============================================================================
# Roteamento
# ============================================================================
PAGES = {
    "🏠 Visão Geral": page_overview,
    "📊 Análise Exploratória": page_eda,
    "🤖 Modelos de ML": page_models,
    "🎯 Predição Interativa": page_predict,
    "🗺️ Mapas Geográficos": page_maps,
    "🔍 Anomalias": page_anomalies,
    "📅 Sazonalidade": page_seasonality,
    "ℹ️ Sobre o Projeto": page_about,
}

PAGES[page]()
