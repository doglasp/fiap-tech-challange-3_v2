"""
Textos narrativos do relatório HTML.
Centralizando aqui pra facilitar edição sem mexer no gerador.
Use f-strings com .format() para placeholders dinâmicos.
"""

CAPA = {
    "titulo": "Análise de Atrasos em Voos Domésticos",
    "subtitulo": "Tech Challenge — FIAP Postech Engenharia de Machine Learning",
    "autor": "Equipe de Engenharia de ML",
    "dataset": "U.S. Department of Transportation — Bureau of Transportation Statistics (2015)",
}

SUMARIO_EXECUTIVO = """
Este relatório apresenta uma análise completa dos atrasos em voos domésticos nos Estados Unidos
no ano de 2015, com base em 5,7 milhões de registros. O estudo combina técnicas de
<strong>análise exploratória</strong>, <strong>modelagem supervisionada</strong> (classificação e regressão),
<strong>modelagem não supervisionada</strong> (clusterização) e <strong>detecção de anomalias</strong>
para responder a quatro perguntas centrais:
<ol>
    <li>É possível prever se um voo vai atrasar?</li>
    <li>Quanto tempo de atraso podemos esperar?</li>
    <li>Quais aeroportos têm perfis operacionais semelhantes?</li>
    <li>Quando e onde os atrasos são mais frequentes?</li>
</ol>
"""

PRINCIPAIS_ACHADOS = [
    "A <strong>taxa de atraso média</strong> ficou em torno de 36% dos voos, com forte variação por período do dia (noite ~43%, madrugada ~22%).",
    "Modelos de classificação atingiram <strong>ROC-AUC entre 0.65 e 0.70</strong>, indicando previsibilidade moderada com as features disponíveis.",
    "Modelos de regressão tiveram <strong>R² baixo (≤ 0.06)</strong> — confirmando que o tempo exato de atraso depende fortemente de variáveis externas (clima, congestionamento, efeito cascata) não presentes no dataset.",
    "A clusterização identificou <strong>3 a 4 perfis distintos de aeroportos</strong> — hubs grandes, regionais tranquilos e operações mistas.",
    "A análise de sazonalidade revelou que <strong>junho e dezembro</strong> são os meses mais críticos, e que <strong>noites de quinta e sexta</strong> concentram os piores cenários.",
]

# ----------------------------------------------------------------------------
# Seção 1 — EDA
# ----------------------------------------------------------------------------
EDA_INTRO = """
A base de dados contém <strong>{n_voos:,}</strong> registros de voos domésticos nos EUA em 2015,
distribuídos entre <strong>{n_companhias}</strong> companhias aéreas e <strong>{n_aeroportos}</strong> aeroportos.
Antes da modelagem, foi feito um pré-processamento que envolveu:
<ul>
    <li>Filtragem de voos cancelados e desviados (esses não têm tempo de atraso medido)</li>
    <li>Criação da target binária <code>is_delayed</code> (atraso na chegada &gt; 0 minutos)</li>
    <li>Engenharia da feature <code>periodo_dia</code> (madrugada / manhã / tarde / noite) a partir do horário programado</li>
    <li>Tratamento de valores ausentes (mediana para numéricas, moda para categóricas, no pipeline de modelagem)</li>
</ul>
""".strip()

EDA_ANALISE = """
A distribuição do atraso médio mostra uma <strong>cauda longa à direita</strong>: a maioria dos voos
chega no horário ou com pequeno atraso, mas eventos extremos (atrasos &gt; 60 minutos) puxam a média.
Isso é importante porque sugere que tratar o problema como <strong>classificação</strong> (atrasa ou não)
deve funcionar melhor do que regressão (quantos minutos), já que os valores extremos são difíceis
de prever sem variáveis adicionais como clima.
"""

# ----------------------------------------------------------------------------
# Seção 2 — Modelagem Supervisionada
# ----------------------------------------------------------------------------
CLASSIFICACAO_INTRO = """
Para a classificação (prever se o voo vai atrasar), comparamos dois algoritmos de famílias diferentes:
<ul>
    <li><strong>Logistic Regression</strong> — modelo linear, leve, serve como baseline interpretável</li>
    <li><strong>Random Forest</strong> — ensemble de árvores, captura interações não-lineares</li>
</ul>
Ambos foram treinados com <code>class_weight="balanced"</code> para compensar o leve desbalanceamento
entre as classes (~36% positivos vs ~64% negativos).
"""

CLASSIFICACAO_ANALISE = """
O <strong>{best_clf}</strong> apresentou o melhor desempenho geral, com ROC-AUC de {best_auc:.3f}.
A diferença entre os dois modelos foi modesta, indicando que o ganho de complexidade do Random Forest
não traz retorno proporcional. Isso reforça a hipótese de que <strong>o gargalo não está no algoritmo</strong>,
e sim na ausência de features causais (clima, status do aeroporto em tempo real, atraso do voo anterior
da mesma aeronave).
"""

REGRESSAO_INTRO = """
Para a regressão (prever quantos minutos de atraso), comparamos três famílias distintas:
<ul>
    <li><strong>Linear Regression</strong> — baseline linear sem regularização</li>
    <li><strong>Decision Tree</strong> — árvore única (base intuitiva do Random Forest)</li>
    <li><strong>LightGBM</strong> — gradient boosting de árvores, estado da arte para tabular</li>
</ul>
A progressão de complexidade (linear → árvore → boosting) permite avaliar quanto cada degrau
adiciona em capacidade preditiva.
"""

REGRESSAO_ANALISE = """
Apesar do LightGBM ter ganhado em todas as métricas (MAE={lgbm_mae:.2f}, R²={lgbm_r2:.3f}),
o R² geral ficou abaixo de 0.10 para todos os modelos. Esse teto baixo é o resultado mais importante
da análise de regressão: <strong>os features disponíveis no dataset não capturam a maior parte da variância
do tempo de atraso</strong>. Variáveis ausentes que provavelmente explicariam a diferença incluem:
condições meteorológicas, congestionamento do espaço aéreo, estado da aeronave (atraso anterior),
e eventos pontuais (greves, problemas técnicos).
"""

# ----------------------------------------------------------------------------
# Seção 3 — Clusterização
# ----------------------------------------------------------------------------
CLUSTERIZACAO_INTRO = """
Para identificar perfis operacionais de aeroportos, aplicamos <strong>KMeans</strong> sobre 13 features
agregadas por aeroporto (taxa de atraso, volume, distância média, mix de horários, etc.).
O número ideal de clusters foi escolhido via <strong>Elbow + Silhouette Score</strong>, testando k de 2 a 10.
"""

CLUSTERIZACAO_ANALISE = """
O melhor k encontrado foi <strong>{best_k}</strong>, com silhouette de {silhouette:.3f}.
A interpretação dos clusters mostra perfis bem distintos:
<ul>
    <li><strong>Cluster de hubs grandes</strong>: alto volume, alta diversidade de destinos, taxa de atraso média-alta</li>
    <li><strong>Cluster de aeroportos regionais</strong>: baixo volume, poucas companhias, taxa de atraso baixa</li>
    <li><strong>Cluster intermediário</strong>: aeroportos médios com perfil misto</li>
</ul>
A redução para 2D via PCA preserva ~{var_total:.0f}% da variância e permite visualizar a separação
geográfica e operacional dos grupos.
"""

# ----------------------------------------------------------------------------
# Seção 4 — Mapas
# ----------------------------------------------------------------------------
MAPAS_INTRO = """
A visualização geográfica conecta os achados estatísticos ao território. Foram gerados cinco mapas
complementares:
<ol>
    <li>Scatter geo dos aeroportos coloridos por taxa de atraso</li>
    <li>Scatter geo dos aeroportos coloridos por cluster</li>
    <li>Linhas das top rotas, com cor proporcional ao atraso médio</li>
    <li>Heatmap interativo com camada de calor por atraso grave</li>
    <li>Choropleth por estado, agregando todos os aeroportos da UF</li>
</ol>
"""

MAPAS_ANALISE = """
A leitura espacial revela que os <strong>aeroportos com pior desempenho concentram-se na região nordeste</strong>
(corredor Boston–Washington), provavelmente devido ao tráfego intenso e congestionamento do espaço aéreo
nessa faixa. Estados do interior e do sul apresentam taxas de atraso significativamente menores,
o que está alinhado com o menor volume operacional.
"""

# ----------------------------------------------------------------------------
# Seção 5 — Anomalias
# ----------------------------------------------------------------------------
ANOMALIAS_INTRO = """
A detecção de anomalias buscou identificar aeroportos com <strong>perfil operacional atípico</strong>,
isto é, fora do padrão da maioria. Combinamos três métodos independentes:
<ul>
    <li><strong>Isolation Forest</strong> — isola pontos atípicos no espaço de features</li>
    <li><strong>Local Outlier Factor (LOF)</strong> — detecta anomalias por densidade local</li>
    <li><strong>Silhouette individual</strong> — aeroportos mal alocados pelo KMeans</li>
</ul>
Um aeroporto é considerado anômalo se foi sinalizado por <strong>pelo menos 2 dos 3 métodos</strong>,
um critério de consenso que reduz falsos positivos.
"""

ANOMALIAS_ANALISE = """
Foram identificados <strong>{n_anomalos} aeroportos</strong> no consenso de anomalias. A análise comparativa
de perfis mostra que os aeroportos anômalos têm, em média, taxa de atraso e variabilidade
significativamente maiores que o grupo normal. Muitos deles são <strong>aeroportos de regiões
remotas (Alasca, Havaí, áreas de montanha)</strong>, onde condições meteorológicas adversas e baixa
infraestrutura justificam o comportamento atípico.
"""

# ----------------------------------------------------------------------------
# Seção 6 — Sazonalidade
# ----------------------------------------------------------------------------
SAZONALIDADE_INTRO = """
A análise sazonal investiga <strong>quando</strong> os atrasos são mais frequentes. Cinco dimensões
foram exploradas:
<ul>
    <li>Padrão mensal — qual mês tem mais atrasos?</li>
    <li>Padrão semanal — qual dia da semana é o pior?</li>
    <li>Padrão horário — qual período do dia concentra os atrasos?</li>
    <li>Cruzamentos (heatmaps) — combinações críticas de mês × dia × período</li>
    <li>Sazonalidade por companhia aérea</li>
</ul>
"""

SAZONALIDADE_ANALISE = """
Os achados sazonais são claros e operacionalmente acionáveis:
<ul>
    <li><strong>Meses críticos</strong>: junho (chuvas de verão) e dezembro (festas de fim de ano)</li>
    <li><strong>Dia da semana mais problemático</strong>: quinta e sexta-feira</li>
    <li><strong>Período do dia</strong>: noite (acúmulo de atrasos ao longo do dia)</li>
    <li><strong>Pior combinação</strong>: noites de junho e dezembro em dias úteis no fim da semana</li>
</ul>
Essa informação tem valor operacional direto: companhias e aeroportos podem alocar buffer de tempo,
reservar mais staff e comunicar passageiros proativamente nesses cenários.
"""

# ----------------------------------------------------------------------------
# Seção 7 — Conclusões e Limitações
# ----------------------------------------------------------------------------
CONCLUSOES = """
A análise consolidada permite responder às quatro perguntas iniciais:

<ol>
    <li><strong>É possível prever se um voo vai atrasar?</strong> Sim, com desempenho moderado
        (ROC-AUC ~0.66). Útil como sinal probabilístico, mas não como decisão definitiva.</li>
    <li><strong>Quanto tempo de atraso podemos esperar?</strong> Não com os dados atuais.
        O R² baixo (≤ 0.06) indica que o tempo exato depende de variáveis externas ausentes.</li>
    <li><strong>Quais aeroportos têm perfis semelhantes?</strong> Foram identificados 3-4 perfis
        operacionais claros, com interpretação direta (hubs, regionais, intermediários).</li>
    <li><strong>Quando e onde os atrasos são mais frequentes?</strong> Junho e dezembro, noites,
        quintas e sextas. Geograficamente, corredor nordeste dos EUA é o mais crítico.</li>
</ol>
"""

LIMITACOES = """
<ul>
    <li><strong>Ausência de dados meteorológicos</strong> — clima é provavelmente a maior causa de
        atraso não capturada</li>
    <li><strong>Ausência de efeito cascata</strong> — atraso do voo anterior da mesma aeronave não está modelado</li>
    <li><strong>Janela temporal limitada</strong> (apenas 2015) — não captura tendências de longo prazo
        nem mudanças pós-pandemia</li>
    <li><strong>Granularidade horária restrita</strong> — apenas horário programado, sem hora real de
        decolagem/pouso para análise mais fina</li>
    <li><strong>Foco no mercado americano</strong> — resultados não generalizam direto para o Brasil
        ou outros países sem revalidação</li>
</ul>
"""

PROXIMOS_PASSOS = """
<ol>
    <li><strong>Enriquecer o dataset</strong> com dados meteorológicos (METAR/NOAA) e congestionamento
        do espaço aéreo (FAA)</li>
    <li><strong>Modelar o efeito cascata</strong> criando feature de "atraso do voo anterior da mesma
        aeronave"</li>
    <li><strong>Testar modelos sequenciais</strong> (séries temporais por aeroporto) para capturar
        dependências temporais</li>
    <li><strong>Validar em mais anos</strong> (2016-2019) para verificar estabilidade dos clusters
        e dos padrões sazonais</li>
    <li><strong>Deploy como serviço</strong> — API REST que recebe dados do voo e retorna probabilidade
        de atraso, integrável a sistemas de booking e operações</li>
</ol>
"""

# ----------------------------------------------------------------------------
# Anexo técnico
# ----------------------------------------------------------------------------
ANEXO_STACK = """
<ul>
    <li><strong>Processamento de dados</strong>: Polars (DataFrame de alto desempenho)</li>
    <li><strong>Modelagem supervisionada</strong>: scikit-learn, LightGBM</li>
    <li><strong>Modelagem não supervisionada</strong>: scikit-learn (KMeans, IsolationForest, LOF, PCA)</li>
    <li><strong>Visualização</strong>: matplotlib, seaborn, plotly, folium</li>
    <li><strong>Geração do relatório</strong>: Python + HTML/CSS</li>
</ul>
"""
