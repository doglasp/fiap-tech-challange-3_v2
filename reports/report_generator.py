"""
Gerador do relatório HTML consolidado.
Recebe as instâncias das 7 classes do projeto e produz um único arquivo HTML
auto-contido (gráficos embutidos como base64 ou HTML inline).
"""

import io
import base64
from datetime import datetime
from pathlib import Path

# IMPORTANTE: força backend não-interativo ANTES de importar pyplot.
# Sem isso, plt.show() dentro dos métodos plot_* pode fechar a figura
# antes de conseguirmos capturá-la, resultando em imagens em branco.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reports import templates as T


# ----------------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------------
def _capture_plot(plot_func, *args, **kwargs):
    """
    Executa um método de plotagem e captura a figura ANTES que ela seja fechada.

    Estratégia:
    1. Fecha qualquer figura aberta antes (limpa estado)
    2. Executa o método (que cria figura + faz plt.show())
    3. Procura QUALQUER figura que ainda esteja na memória
    4. Se não houver, captura a última figura conhecida pelo matplotlib

    Retorna o HTML <img> com a figura em base64, ou mensagem de erro.
    """
    try:
        # Limpa estado anterior
        plt.close("all")

        # Captura figuras antes (deveria ser zero, mas garantia)
        figs_before = set(plt.get_fignums())

        # Executa o método de plotagem
        plot_func(*args, **kwargs)

        # Captura figuras que apareceram durante a execução
        figs_after = set(plt.get_fignums())
        new_figs = figs_after - figs_before

        if not new_figs:
            # plt.show() pode ter fechado tudo — tenta a última figura
            # mesmo assim (pode estar zumbi mas com dados)
            if figs_after:
                fig = plt.figure(max(figs_after))
            else:
                return "<p><em>Figura não capturada (foi fechada antes do save).</em></p>"
        else:
            # Pega a figura mais recente criada pelo método
            fig = plt.figure(max(new_figs))

        # Salva em buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        plt.close("all")

        return f'<img src="data:image/png;base64,{img_b64}" alt="gráfico" />'

    except Exception as e:
        plt.close("all")
        return f'<p><em>Erro ao gerar gráfico: {type(e).__name__}: {e}</em></p>'


def _fig_to_base64(fig=None, dpi=110):
    """Versão antiga (mantida pra compatibilidade). Use _capture_plot quando possível."""
    if fig is None:
        fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{img_b64}" alt="gráfico" />'


def _plotly_to_html(fig, height=520):
    """Converte uma figura Plotly em HTML embutível (mantém interatividade)."""
    if fig is None:
        return "<p><em>Gráfico Plotly indisponível.</em></p>"
    return fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        default_height=f"{height}px",
        config={"displayModeBar": False},
    )


def _folium_to_html(m, height=520):
    """Converte um mapa Folium em iframe HTML embutível."""
    if m is None:
        return "<p><em>Mapa Folium indisponível.</em></p>"
    html_str = m.get_root().render()
    # Embutir via srcdoc pra evitar arquivo externo
    html_escaped = html_str.replace('"', "&quot;")
    return (
        f'<iframe srcdoc="{html_escaped}" '
        f'style="width:100%; height:{height}px; border:none; border-radius:8px;"></iframe>'
    )


def _df_to_html(df, index=True, formatters=None):
    """Converte DataFrame em HTML com classes CSS, envolvido em wrapper rolável."""
    table_html = df.to_html(
        index=index,
        classes="data-table",
        border=0,
        formatters=formatters,
        escape=False,
    )
    return f'<div class="table-wrapper">{table_html}</div>'


def _section(title, content, subtitle=None, level=1):
    """Bloco padrão de seção com header e conteúdo."""
    subtitle_html = f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""
    return f"""
    <section class="report-section section-level-{level}">
        <h{level + 1} class="section-title">{title}</h{level + 1}>
        {subtitle_html}
        <div class="section-content">{content}</div>
    </section>
    """


def _figure_block(html_content, caption=""):
    """Wrapper de figura com legenda."""
    caption_html = f'<figcaption>{caption}</figcaption>' if caption else ""
    return f"""
    <figure class="report-figure">
        {html_content}
        {caption_html}
    </figure>
    """


# ----------------------------------------------------------------------------
# CSS (embutido)
# ----------------------------------------------------------------------------
CSS = """
:root {
    --primary: #1e3a5f;
    --primary-light: #3a5a85;
    --accent: #d97757;
    --bg: #fafafa;
    --bg-card: #ffffff;
    --text: #1a1a1a;
    --text-muted: #6b7280;
    --border: #e5e7eb;
    --success: #059669;
    --warning: #d97706;
    --danger: #dc2626;
}

* { box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.7;
    color: var(--text);
    background: var(--bg);
    margin: 0;
    padding: 0;
}

.container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem;
}

/* Capa */
.cover {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
    color: white;
    padding: 4rem 2rem;
    text-align: center;
    border-radius: 12px;
    margin-bottom: 3rem;
}

.cover h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0 0 0.5rem;
    letter-spacing: -0.02em;
}

.cover .subtitle {
    font-size: 1.15rem;
    opacity: 0.85;
    margin-bottom: 2rem;
}

.cover .meta {
    font-size: 0.9rem;
    opacity: 0.7;
    margin-top: 1.5rem;
}

/* Sumário executivo */
.executive-summary {
    background: var(--bg-card);
    border-left: 4px solid var(--accent);
    padding: 2rem;
    margin-bottom: 3rem;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.executive-summary h2 {
    margin-top: 0;
    color: var(--primary);
}

.findings-list {
    list-style: none;
    padding: 0;
    margin: 1.5rem 0 0;
}

.findings-list li {
    padding: 0.75rem 0 0.75rem 2rem;
    position: relative;
    border-bottom: 1px solid var(--border);
}

.findings-list li:last-child { border-bottom: none; }

.findings-list li::before {
    content: "→";
    position: absolute;
    left: 0;
    color: var(--accent);
    font-weight: bold;
    font-size: 1.2rem;
}

/* Seções */
.report-section {
    background: var(--bg-card);
    padding: 2.5rem;
    margin-bottom: 2.5rem;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.section-title {
    color: var(--primary);
    margin-top: 0;
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.75rem;
    margin-bottom: 1.5rem;
}

.section-subtitle {
    color: var(--text-muted);
    margin-top: -0.75rem;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
}

.section-content { color: var(--text); }

.section-content p { margin: 1rem 0; }

.section-content ul, .section-content ol {
    margin: 1rem 0;
    padding-left: 1.5rem;
}

.section-content code {
    background: #f3f4f6;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-family: "SF Mono", Monaco, Consolas, monospace;
    font-size: 0.9em;
    color: var(--danger);
}

/* Figuras */
.report-figure {
    margin: 1.5rem 0;
    text-align: center;
}

.report-figure img {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.report-figure figcaption {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
    font-style: italic;
}

/* Tabelas */
.table-wrapper {
    overflow-x: auto;
    margin: 1rem 0;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    border-radius: 8px;
    overflow: hidden;
}

.data-table thead {
    background: var(--primary);
    color: white;
}

.data-table th, .data-table td {
    padding: 0.6rem 0.85rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}

.data-table tbody tr:nth-child(even) { background: #f9fafb; }

.data-table tbody tr:hover { background: #f3f4f6; }

@media print {
    .data-table { font-size: 0.78rem; }
    .data-table th, .data-table td { padding: 0.4rem 0.55rem; }
    .table-wrapper { overflow: visible; }
}

/* Highlight boxes */
.highlight-box {
    background: #fff7ed;
    border-left: 4px solid var(--accent);
    padding: 1rem 1.5rem;
    margin: 1.5rem 0;
    border-radius: 4px;
}

.highlight-box.success {
    background: #ecfdf5;
    border-left-color: var(--success);
}

.highlight-box.warning {
    background: #fffbeb;
    border-left-color: var(--warning);
}

/* Layout grid pra modelos */
.models-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin: 1.5rem 0;
}

.model-card {
    background: #f9fafb;
    padding: 1.5rem;
    border-radius: 8px;
    border: 1px solid var(--border);
}

.model-card h4 {
    margin-top: 0;
    color: var(--primary);
}

/* Footer */
.report-footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.85rem;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border);
}

/* Responsivo */
@media (max-width: 768px) {
    .container { padding: 1rem; }
    .cover { padding: 2rem 1rem; }
    .cover h1 { font-size: 1.75rem; }
    .report-section { padding: 1.5rem; }
    .models-grid { grid-template-columns: 1fr; }
}

/* Print */
@media print {
    body { background: white; }
    .report-section { page-break-inside: avoid; box-shadow: none; }
    .cover { background: var(--primary) !important; -webkit-print-color-adjust: exact; }
}
"""


# ----------------------------------------------------------------------------
# Construtores de cada seção
# ----------------------------------------------------------------------------
def build_cover():
    """Capa do relatório."""
    today = datetime.now().strftime("%d/%m/%Y")
    return f"""
    <div class="cover">
        <h1>{T.CAPA['titulo']}</h1>
        <p class="subtitle">{T.CAPA['subtitulo']}</p>
        <div class="meta">
            <p>Dataset: {T.CAPA['dataset']}</p>
            <p>Gerado em: {today}</p>
        </div>
    </div>
    """


def build_executive_summary():
    """Sumário executivo com principais achados."""
    findings_html = "\n".join(
        f"<li>{finding}</li>" for finding in T.PRINCIPAIS_ACHADOS
    )
    return f"""
    <div class="executive-summary">
        <h2>Sumário Executivo</h2>
        <div>{T.SUMARIO_EXECUTIVO}</div>
        <h3>Principais achados</h3>
        <ul class="findings-list">{findings_html}</ul>
    </div>
    """


def build_eda_section(eda):
    """Seção 1 — Análise Exploratória."""
    n_voos = eda.df_model.height if eda.df_model is not None else 0
    n_companhias = eda.df_model["AIRLINE"].n_unique() if eda.df_model is not None else 0
    n_aeroportos = (
        eda.df_model["ORIGIN_AIRPORT"].n_unique() if eda.df_model is not None else 0
    )

    intro = T.EDA_INTRO.format(
        n_voos=n_voos, n_companhias=n_companhias, n_aeroportos=n_aeroportos
    )

    # Tabela: top 10 companhias por taxa de atraso
    if eda.delay_by_airline is not None:
        top_airlines_df = (
            eda.delay_by_airline.head(10)
            .to_pandas()[["AIRLINE", "qtd_voos", "media_arrival_delay", "taxa_atraso"]]
            .rename(
                columns={
                    "AIRLINE": "Companhia",
                    "qtd_voos": "Voos",
                    "media_arrival_delay": "Delay médio (min)",
                    "taxa_atraso": "Taxa de atraso",
                }
            )
        )
        top_airlines_df["Taxa de atraso"] = top_airlines_df["Taxa de atraso"].apply(
            lambda x: f"{x:.1%}"
        )
        top_airlines_df["Delay médio (min)"] = top_airlines_df["Delay médio (min)"].round(2)
        top_airlines_df["Voos"] = top_airlines_df["Voos"].apply(lambda x: f"{x:,}")
        airlines_table = _df_to_html(top_airlines_df, index=False)
    else:
        airlines_table = "<p><em>Tabela indisponível.</em></p>"

    # Tabela: atraso por período
    if eda.delay_by_period is not None:
        period_df = eda.delay_by_period.to_pandas().rename(
            columns={
                "periodo_dia": "Período",
                "qtd_voos": "Voos",
                "taxa_atraso": "Taxa de atraso",
                "media_arrival_delay": "Delay médio (min)",
            }
        )
        period_df["Taxa de atraso"] = period_df["Taxa de atraso"].apply(lambda x: f"{x:.1%}")
        period_df["Delay médio (min)"] = period_df["Delay médio (min)"].round(2)
        period_df["Voos"] = period_df["Voos"].apply(lambda x: f"{x:,}")
        period_table = _df_to_html(period_df, index=False)
    else:
        period_table = "<p><em>Tabela indisponível.</em></p>"

    content = f"""
    {intro}

    <h3>Top 10 companhias com maior taxa de atraso</h3>
    {airlines_table}

    <h3>Distribuição de atrasos por período do dia</h3>
    {period_table}

    <div class="highlight-box">
        <strong>Insight:</strong> {T.EDA_ANALISE}
    </div>
    """
    return _section("1. Análise Exploratória dos Dados", content)


def build_classification_section(clf):
    """Seção 2 — Classificação."""
    if clf is None or clf.comparison_df is None:
        return _section(
            "2a. Modelagem Supervisionada — Classificação",
            "<p><em>Resultados de classificação não disponíveis.</em></p>",
        )

    df = clf.comparison_df.copy()
    df_fmt = df.copy()
    for col in df_fmt.columns:
        if col != "modelo":
            df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.4f}")
    comparison_table = _df_to_html(df_fmt, index=False)

    best_idx = clf.comparison_df["roc_auc"].idxmax()
    best_clf = clf.comparison_df.loc[best_idx, "modelo"]
    best_auc = clf.comparison_df.loc[best_idx, "roc_auc"]
    analise = T.CLASSIFICACAO_ANALISE.format(best_clf=best_clf, best_auc=best_auc)

    # Gerar gráficos
    comparison_fig = _capture_plot(clf.plot_comparison)
    roc_fig = _capture_plot(clf.plot_roc_and_confusion)
    fi_fig = _capture_plot(clf.plot_feature_importance)

    content = f"""
    {T.CLASSIFICACAO_INTRO}

    <h3>Métricas comparativas</h3>
    {comparison_table}

    {_figure_block(comparison_fig, "Comparação de modelos por métrica")}
    {_figure_block(roc_fig, "Curva ROC e matrizes de confusão")}
    {_figure_block(fi_fig, "Feature importance do Random Forest (agregada por feature original)")}

    <div class="highlight-box">
        <strong>Análise:</strong> {analise}
    </div>
    """
    return _section("2a. Modelagem Supervisionada — Classificação", content)


def build_regression_section(reg):
    """Seção 2b — Regressão."""
    if reg is None or reg.comparison_df is None:
        return _section(
            "2b. Modelagem Supervisionada — Regressão",
            "<p><em>Resultados de regressão não disponíveis.</em></p>",
        )

    df_fmt = reg.comparison_df.copy().reset_index()
    df_fmt = df_fmt.rename(columns={"modelo": "Modelo"})
    df_fmt["MAE"] = df_fmt["MAE"].apply(lambda x: f"{x:.3f}")
    df_fmt["RMSE"] = df_fmt["RMSE"].apply(lambda x: f"{x:.3f}")
    df_fmt["R2"] = df_fmt["R2"].apply(lambda x: f"{x:.4f}")
    comparison_table = _df_to_html(df_fmt, index=False)

    lgbm_mae = reg.metrics_gb["MAE"]
    lgbm_r2 = reg.metrics_gb["R2"]
    analise = T.REGRESSAO_ANALISE.format(lgbm_mae=lgbm_mae, lgbm_r2=lgbm_r2)

    comparison_fig = _capture_plot(reg.plot_comparison)
    scatter_fig = _capture_plot(reg.plot_predicted_vs_real)

    # Tempos de treino
    times_html = f"""
    <div class="models-grid">
        <div class="model-card">
            <h4>Linear Regression</h4>
            <p>Tempo de treino: <strong>{reg.lr_time:.2f}s</strong></p>
            <p>R²: <strong>{reg.metrics_lr['R2']:.4f}</strong></p>
        </div>
        <div class="model-card">
            <h4>Decision Tree</h4>
            <p>Tempo de treino: <strong>{reg.dt_time:.2f}s</strong></p>
            <p>R²: <strong>{reg.metrics_dt['R2']:.4f}</strong></p>
        </div>
        <div class="model-card">
            <h4>LightGBM</h4>
            <p>Tempo de treino: <strong>{reg.gb_time:.2f}s</strong></p>
            <p>R²: <strong>{reg.metrics_gb['R2']:.4f}</strong></p>
        </div>
        <div class="model-card" style="background: #ecfdf5; border-color: #059669;">
            <h4>🏆 Vencedor</h4>
            <p><strong>LightGBM</strong></p>
            <p>Melhor em todas as métricas</p>
        </div>
    </div>
    """

    content = f"""
    {T.REGRESSAO_INTRO}

    {times_html}

    <h3>Métricas comparativas</h3>
    {comparison_table}

    {_figure_block(comparison_fig, "Comparação de modelos (MAE, RMSE, R²)")}
    {_figure_block(scatter_fig, "Predito vs Real — quanto mais próximo da diagonal, melhor")}

    <div class="highlight-box warning">
        <strong>Análise crítica:</strong> {analise}
    </div>
    """
    return _section("2b. Modelagem Supervisionada — Regressão", content)


def build_clustering_section(clusterer):
    """Seção 3 — Clusterização."""
    if clusterer is None or clusterer.perfil is None:
        return _section(
            "3. Modelagem Não Supervisionada — Clusterização",
            "<p><em>Resultados de clusterização não disponíveis.</em></p>",
        )

    best_k = clusterer.k_final
    silhouettes_list = clusterer.silhouettes or []
    silhouette = max(silhouettes_list) if silhouettes_list else 0.0
    var_total = (
        (clusterer.var_pc1 or 0) + (clusterer.var_pc2 or 0)
        if clusterer.var_pc1 is not None
        else 0
    )

    analise = T.CLUSTERIZACAO_ANALISE.format(
        best_k=best_k, silhouette=silhouette, var_total=var_total
    )

    # Reseta o índice e seleciona apenas as colunas mais informativas
    # (a tabela completa tem 8 colunas e estoura a largura da página)
    perfil_full = clusterer.perfil.round(3).reset_index()
    cols_pt = {
        "cluster": "Cluster",
        "taxa_atraso": "Taxa atraso",
        "taxa_atraso_grave": "Taxa grave",
        "media_arrival_delay": "Delay médio",
        "std_arrival_delay": "Desvio",
        "n_destinos": "Destinos",
        "qtd_voos": "Voos médios",
    }
    perfil_df = perfil_full[list(cols_pt.keys())].rename(columns=cols_pt)
    # Formata "Voos médios" como inteiro com separador
    perfil_df["Voos médios"] = perfil_df["Voos médios"].apply(lambda x: f"{int(x):,}")
    perfil_table = _df_to_html(perfil_df, index=False)

    k_fig = _capture_plot(clusterer.plot_k_selection)
    centroids_fig = _capture_plot(clusterer.plot_centroids_heatmap)
    pca_fig = _capture_plot(clusterer.plot_pca_clusters)

    content = f"""
    {T.CLUSTERIZACAO_INTRO}

    {_figure_block(k_fig, "Elbow e Silhouette Score por k — escolha do número ideal de clusters")}

    <h3>Perfil médio de cada cluster</h3>
    {perfil_table}

    {_figure_block(centroids_fig, "Heatmap dos centróides (valores padronizados)")}
    {_figure_block(pca_fig, "Projeção PCA 2D — separação visual dos clusters")}

    <div class="highlight-box success">
        <strong>Insight:</strong> {analise}
    </div>
    """
    return _section("3. Modelagem Não Supervisionada — Clusterização", content)


def build_geomap_section(mapper):
    """Seção 4 — Mapas geográficos."""
    if mapper is None:
        return _section(
            "4. Visualização Geográfica",
            "<p><em>Mapas não disponíveis.</em></p>",
        )

    delay_map = _plotly_to_html(mapper.fig_delay_rate)
    cluster_map = _plotly_to_html(mapper.fig_clusters)
    routes_map = _plotly_to_html(mapper.fig_routes)
    state_map = _plotly_to_html(mapper.fig_state_choropleth)

    content = f"""
    {T.MAPAS_INTRO}

    <h3>Mapa 1 — Aeroportos por taxa de atraso</h3>
    {_figure_block(delay_map, "Tamanho do círculo = volume de voos. Cor = taxa de atraso.")}

    <h3>Mapa 2 — Aeroportos por cluster</h3>
    {_figure_block(cluster_map, "Cada cor representa um perfil operacional distinto.")}

    <h3>Mapa 3 — Top rotas com atraso médio</h3>
    {_figure_block(routes_map, "Espessura da linha = volume da rota. Cor = atraso médio.")}

    <h3>Mapa 4 — Choropleth por estado</h3>
    {_figure_block(state_map, "Taxa de atraso média agregada por unidade federativa.")}

    <div class="highlight-box">
        <strong>Insight:</strong> {T.MAPAS_ANALISE}
    </div>
    """
    return _section("4. Visualização Geográfica", content)


def build_anomalies_section(detector):
    """Seção 5 — Anomalias."""
    if detector is None or detector.anomalos_consenso is None:
        return _section(
            "5. Detecção de Anomalias",
            "<p><em>Resultados de anomalias não disponíveis.</em></p>",
        )

    n_anomalos = len(detector.anomalos_consenso)
    analise = T.ANOMALIAS_ANALISE.format(n_anomalos=n_anomalos)

    # Tabela dos anômalos — reseta índice pra evitar faixa azul gigante,
    # mantém o código IATA como coluna nomeada
    anomalos_df = detector.anomalos_consenso[
        [
            "n_metodos_anomalia",
            "cluster",
            "taxa_atraso",
            "taxa_atraso_grave",
            "media_arrival_delay",
            "n_destinos",
            "qtd_voos",
        ]
    ].round(3).copy().reset_index()
    anomalos_df.columns = [
        "Aeroporto",
        "Métodos sinalizando",
        "Cluster",
        "Taxa atraso",
        "Taxa grave",
        "Delay médio (min)",
        "Destinos",
        "Voos",
    ]
    anomalos_table = _df_to_html(anomalos_df, index=False)

    methods_fig = _capture_plot(detector.plot_methods_comparison)
    consensus_fig = _capture_plot(detector.plot_consensus_map)
    profile_fig = _capture_plot(detector.plot_profile_comparison)

    content = f"""
    {T.ANOMALIAS_INTRO}

    {_figure_block(methods_fig, "Comparação dos três métodos de detecção lado a lado")}
    {_figure_block(consensus_fig, "Aeroportos coloridos pelo número de métodos que sinalizaram")}

    <h3>Aeroportos identificados como anômalos no consenso</h3>
    {anomalos_table}

    {_figure_block(profile_fig, "Perfil médio: Anômalos vs Normais")}

    <div class="highlight-box warning">
        <strong>Análise:</strong> {analise}
    </div>
    """
    return _section("5. Detecção de Anomalias", content)


def build_seasonality_section(season):
    """Seção 6 — Sazonalidade."""
    if season is None or season.monthly is None:
        return _section(
            "6. Análise de Sazonalidade",
            "<p><em>Resultados de sazonalidade não disponíveis.</em></p>",
        )

    monthly_fig = _capture_plot(season.plot_monthly)
    weekly_fig = _capture_plot(season.plot_weekly)
    heatmap_fig = _capture_plot(season.plot_cross_month_dow)
    critical_fig = _capture_plot(season.plot_critical_combinations, top_n=10)

    # Top 5 cenários críticos
    if season.critical is not None:
        critical_df = season.critical.head(5)[
            ["combinacao", "taxa_atraso", "media_delay", "qtd_voos"]
        ].copy()
        critical_df.columns = ["Cenário", "Taxa de atraso", "Delay médio (min)", "Voos"]
        critical_df["Taxa de atraso"] = critical_df["Taxa de atraso"].apply(
            lambda x: f"{x:.1%}"
        )
        critical_df["Delay médio (min)"] = critical_df["Delay médio (min)"].round(2)
        critical_df["Voos"] = critical_df["Voos"].apply(lambda x: f"{x:,}")
        critical_table = _df_to_html(critical_df, index=False)
    else:
        critical_table = "<p><em>Tabela indisponível.</em></p>"

    content = f"""
    {T.SAZONALIDADE_INTRO}

    {_figure_block(monthly_fig, "Padrão mensal: taxa, delay médio/mediana, p90 e volume")}
    {_figure_block(weekly_fig, "Padrão semanal por dia da semana")}
    {_figure_block(heatmap_fig, "Heatmap cruzado: dia da semana × mês")}

    <h3>Top 5 cenários críticos (Mês / Dia / Período)</h3>
    {critical_table}

    {_figure_block(critical_fig, "Top combinações com maior taxa de atraso")}

    <div class="highlight-box">
        <strong>Insight operacional:</strong> {T.SAZONALIDADE_ANALISE}
    </div>
    """
    return _section("6. Análise de Sazonalidade", content)


def build_conclusions_section():
    """Seção 7 — Conclusões e próximos passos."""
    content = f"""
    <h3>Síntese das descobertas</h3>
    {T.CONCLUSOES}

    <h3>Limitações</h3>
    {T.LIMITACOES}

    <h3>Próximos passos</h3>
    {T.PROXIMOS_PASSOS}
    """
    return _section("7. Conclusões e Próximos Passos", content)


def build_appendix_section():
    """Anexo técnico."""
    content = f"""
    <h3>Stack tecnológico</h3>
    {T.ANEXO_STACK}

    <p style="margin-top: 2rem; color: var(--text-muted); font-size: 0.9rem;">
        Código-fonte completo organizado em 7 classes Python independentes
        (<code>eda.py</code>, <code>classificacao.py</code>, <code>regressao.py</code>,
        <code>clusterizacao.py</code>, <code>mapa_geografico.py</code>, <code>anomalias.py</code>,
        <code>sazonalidade.py</code>) e orquestradas via <code>main.py</code>.
    </p>
    """
    return _section("Anexo Técnico", content)


# ----------------------------------------------------------------------------
# Função principal
# ----------------------------------------------------------------------------
def generate_report(
    eda=None,
    clf=None,
    reg=None,
    clusterer=None,
    mapper=None,
    detector=None,
    season=None,
    output_path="outputs/relatorio.html",
):
    """
    Gera o relatório HTML consolidado a partir das instâncias das 7 classes.

    Parameters
    ----------
    eda, clf, reg, clusterer, mapper, detector, season : instâncias das classes do projeto
        Podem ser None — nesse caso a seção correspondente vai exibir uma mensagem de indisponível.
    output_path : str or Path
        Caminho do HTML de saída. Cria diretórios pais se não existirem.

    Returns
    -------
    Path : caminho absoluto do arquivo gerado.
    """
    print("\n" + "=" * 80)
    print("  Gerando relatório HTML consolidado...")
    print("=" * 80)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections = []

    print("  → Capa e sumário executivo...")
    sections.append(build_cover())
    sections.append(build_executive_summary())

    print("  → Seção 1: EDA...")
    sections.append(build_eda_section(eda))

    print("  → Seção 2a: Classificação...")
    sections.append(build_classification_section(clf))

    print("  → Seção 2b: Regressão...")
    sections.append(build_regression_section(reg))

    print("  → Seção 3: Clusterização...")
    sections.append(build_clustering_section(clusterer))

    print("  → Seção 4: Mapas geográficos...")
    sections.append(build_geomap_section(mapper))

    print("  → Seção 5: Anomalias...")
    sections.append(build_anomalies_section(detector))

    print("  → Seção 6: Sazonalidade...")
    sections.append(build_seasonality_section(season))

    print("  → Seção 7: Conclusões...")
    sections.append(build_conclusions_section())
    sections.append(build_appendix_section())

    body_html = "\n".join(sections)

    full_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{T.CAPA['titulo']}</title>
    <style>{CSS}</style>
</head>
<body>
    <div class="container">
        {body_html}
        <div class="report-footer">
            <p>Relatório gerado automaticamente pelo pipeline de análise.</p>
            <p>Tech Challenge — FIAP Postech Engenharia de Machine Learning</p>
        </div>
    </div>
</body>
</html>
"""

    output_path.write_text(full_html, encoding="utf-8")
    abs_path = output_path.resolve()

    size_mb = abs_path.stat().st_size / (1024 * 1024)
    print(f"\n  ✓ Relatório gerado: {abs_path}")
    print(f"  ✓ Tamanho: {size_mb:.2f} MB")
    print("  ✓ Abra o arquivo no navegador para visualizar")

    return abs_path
