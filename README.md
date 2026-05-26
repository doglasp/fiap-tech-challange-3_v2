# ✈️ Flight Delay ML — Tech Challenge 3

Pipeline completo de Machine Learning sobre o dataset público **2015 Flight Delays and Cancellations** (U.S. DOT), aplicando aprendizado supervisionado (classificação e regressão) e não-supervisionado (clusterização + PCA) para entender e prever atrasos em voos comerciais nos EUA.

## Integrantes do Grupo

| Nome | E-mail | Perfil Google Skills |
| :--- | :--- | :---: |
| **Doglas Parise** | [doglasparise@gmail.com](mailto:doglasparise@gmail.com) | [🔗 Skills Google](https://www.skills.google/public_profiles/c73ebebd-15ad-4883-97f3-02551573d9b9) |
| **Mariana Teixeira Dornelles Parise** | [m.dornelles19@gmail.com](mailto:m.dornelles19@gmail.com) | [🔗 Skills Google](https://www.skills.google/public_profiles/c71a2add-704b-450f-9eba-2ebb17f39191) |
| **Ricardo Gomes de Souza** | [ricardo_g_souza@yahoo.com](mailto:ricardo_g_souza@yahoo.com) | - |
| **Silvio José Meirelles** | [professorsilviomeireles@gmail.com](mailto:professorsilviomeireles@gmail.com) | - |

## Vídeo de apresentação

[Link para vídeo de apresentação](#) <!-- preencher quando gravado -->

## 🚀 Dashboard Interativo

Dashboard interativo do projeto disponível publicamente via Streamlit Cloud:

🔗 **[fiap-mlet8-g37-tc3.streamlit.app](https://fiap-mlet8-g37-tc3.streamlit.app/)**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fiap-mlet8-g37-tc3.streamlit.app/)

## 📋 Sobre o Projeto

Este projeto realiza uma análise end-to-end de atrasos em **5,7 milhões de voos domésticos** nos Estados Unidos no ano de 2015, cobrindo todas as etapas de um projeto de ciência de dados:

- **Análise Exploratória de Dados (EDA)**
- **Modelagem Supervisionada** (Classificação e Regressão)
- **Modelagem Não Supervisionada** (Clusterização e Detecção de Anomalias)
- **Visualização Geográfica** interativa
- **Análise de Sazonalidade**
- **Dashboard interativo** com Streamlit
- **Relatório HTML** estático consolidado

---

## 📊 Dataset

| Atributo       | Valor                                     |
|----------------|-------------------------------------------|
| **Fonte**      | U.S. Department of Transportation (DOT)   |
| **Ano**        | 2015                                      |
| **Volume**     | 5.714.008 voos                            |
| **Companhias** | 14 distintas                              |
| **Aeroportos** | 628                                       |
| **Formato raw**| CSV (`data/raw/flights.csv`)              |

---

## 🏗️ Arquitetura do Projeto

```
flight-delay-ml/
├── data/
│   ├── raw/
│   │   └── flights.csv
│   └── processed/
│       └── flights_model.parquet
│
├── docs/
│   └── arquitetura.md
│
├── src/                          # Classes do pipeline
│   ├── eda.py                    # FlightEDA
│   ├── classificacao.py          # FlightClassifier
│   ├── regressao.py              # FlightRegressor
│   ├── clusterizacao.py          # FlightClusterer
│   ├── mapa_geografico.py        # FlightGeoMapper
│   ├── anomalias.py              # FlightAnomalyDetector
│   └── sazonalidade.py           # FlightSeasonality
│
├── notebooks/                    # Notebooks de teste/exploração
│   ├── test_eda.ipynb
│   ├── test_classificacao.ipynb
│   ├── test_regressao.ipynb
│   ├── test_clusterizacao.ipynb
│   ├── test_mapa_geografico.ipynb
│   ├── test_anomalias.ipynb
│   └── test_sazonalidade.ipynb
│
├── reports/
│   ├── __init__.py
│   ├── report_generator.py
│   └── templates.py
│
├── outputs/
│   ├── relatorio.html
│   ├── mapa_atrasos_heatmap.html
│   └── pickles/
│       ├── eda.pkl
│       ├── clf.pkl
│       ├── reg.pkl
│       ├── clusterer.pkl
│       ├── mapper.pkl
│       ├── detector.pkl
│       ├── season.pkl
│       └── folium_map.html
│
├── main.py                       # Orquestrador do pipeline
├── dashboard.py                  # Streamlit
├── test_main.ipynb               # Notebook de teste do pipeline completo
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🔬 Pipeline de Análise (7 Etapas)

### Etapa 1 — Análise Exploratória (`eda.py`)
- Carregamento e limpeza do CSV (~5,7M linhas)
- Filtragem, imputação e engenharia de features (ex: `periodo_dia`)
- Agregações: atrasos por companhia, aeroporto, período do dia
- Exporta `flights_model.parquet` para as demais etapas

### Etapa 2 — Classificação (`classificacao.py`)
**Pergunta:** *O voo vai atrasar?* (target binário: `is_delayed`)

| Modelo              | Estratégia                     |
|---------------------|-------------------------------|
| Logistic Regression | Linear, regularizado           |
| Random Forest       | Ensemble, 150 árvores, depth=12|

- Pré-processamento: `ColumnTransformer` com `StandardScaler` + `OneHotEncoder`
- Métricas: Accuracy, Precision, Recall, F1, **ROC-AUC**
- Amostragem de 10% para treino ágil (`sample_frac=0.1`)

### Etapa 3 — Regressão (`regressao.py`)
**Pergunta:** *Quantos minutos de atraso?* (target contínuo: `ARRIVAL_DELAY`)

| Modelo             | Observação                             |
|--------------------|----------------------------------------|
| Linear Regression  | Baseline linear                        |
| Decision Tree      | Árvore única (max_depth=8)             |
| LightGBM           | Gradient boosting (100 estimadores) 🏆 |

- Amostragem de 30% (`sample_frac=0.3`)
- Métricas: MAE, RMSE, **R²**

### Etapa 4 — Clusterização (`clusterizacao.py`)
- KMeans para agrupar aeroportos por perfil operacional
- Redução de dimensionalidade com PCA (2D) para visualização
- Identifica perfis: *hubs grandes*, *regionais tranquilos*, *intermediários*

### Etapa 5 — Visualização Geográfica (`mapa_geografico.py`)
5 tipos de visualização:
- 🔵 Aeroportos por taxa de atraso (Plotly)
- 🟠 Aeroportos por cluster (Plotly)
- 🟢 Top rotas com atraso médio (Plotly)
- 🗺️ Choropleth por estado (Plotly)
- 🔥 Heatmap interativo (Folium)

### Etapa 6 — Detecção de Anomalias (`anomalias.py`)
Consenso entre **3 métodos** para identificar aeroportos com perfil atípico:
- Isolation Forest
- Local Outlier Factor (LOF)
- Silhouette individual

### Etapa 7 — Sazonalidade (`sazonalidade.py`)
Padrões de atraso em múltiplas dimensões temporais:
- 📅 Mensal
- 📆 Semanal (dia da semana)
- ⏰ Por período do dia (madrugada, manhã, tarde, noite)
- 💥 Cenários críticos (combinação mês × dia × período)

---

## 🚀 Como Executar

### 1. Pré-requisitos

```bash
# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt
```

> **⚠️ Nota:** O Streamlit (`streamlit`) não está no `requirements.txt` padrão. Instale separadamente se quiser rodar o dashboard:
> ```bash
> pip install streamlit
> ```

### 2. Dataset

Coloque o arquivo `flights.csv` em `data/raw/flights.csv`.  
O dataset pode ser obtido no [Kaggle — 2015 Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays).

### 3. Executar o Pipeline Completo

```bash
python main.py
```

Isso irá:
1. Processar os dados e gerar `data/processed/flights_model.parquet`
2. Treinar todos os modelos (EDA → Classificação → Regressão → Clusterização → Mapa → Anomalias → Sazonalidade)
3. Gerar o relatório HTML em `outputs/relatorio.html`
4. Serializar os objetos treinados em `outputs/pickles/`

> **⏱️ Tempo estimado:** 15–40 minutos dependendo da CPU (dataset ~5,7M linhas).

### 4. Executar o Dashboard Interativo

```bash
streamlit run dashboard.py
```

O dashboard carrega automaticamente os pickles gerados. Se não existirem, executa o pipeline completo na primeira inicialização.

### 5. Visualizar o Relatório HTML

Abra o arquivo `outputs/relatorio.html` diretamente no navegador.

---

## 🧰 Stack Tecnológica

| Categoria              | Tecnologias                                      |
|------------------------|--------------------------------------------------|
| **Processamento**      | Polars, Pandas, PyArrow                          |
| **Modelagem**          | scikit-learn, LightGBM                           |
| **Visualização**       | Matplotlib, Seaborn, Plotly, Folium              |
| **Dashboard**          | Streamlit                                        |
| **Serialização**       | Pickle, Joblib                                   |
| **Relatório**          | Python + HTML/CSS                                |

---

## 📈 Principais Resultados

| Análise          | Resultado                                                                 |
|------------------|---------------------------------------------------------------------------|
| **Classificação**| ROC-AUC ~0.66 — previsibilidade moderada do atraso binário                |
| **Regressão**    | R² ≤ 0.06 — tempo exato de atraso não previsível sem dados externos       |
| **Clusterização**| 3 perfis distintos de aeroportos identificados                            |
| **Sazonalidade** | Junho e dezembro, noites de quinta e sexta são os piores cenários         |
| **Anomalias**    | Aeroportos remotos (Alasca, Havaí) emergem como operacionalmente atípicos |

### Por que a regressão tem R² tão baixo?
O tempo exato de atraso depende fortemente de **variáveis externas** não disponíveis no dataset:
- 🌦️ Condições climáticas
- 🔄 Efeito cascata (atraso anterior da mesma aeronave)
- 🛫 Congestionamento de tráfego aéreo em tempo real

---

## ⚙️ Configurações de Amostragem

Para execução mais rápida durante desenvolvimento, os modelos usam frações do dataset:

| Módulo          | `sample_frac` padrão | Linhas aproximadas |
|-----------------|---------------------|--------------------|
| Classificação   | 10%                 | ~570 mil           |
| Regressão       | 30%                 | ~1,7 milhão        |
| Demais módulos  | 100%                | ~5,7 milhões       |

Ajuste os parâmetros em `main.py` conforme a capacidade da sua máquina.

---

## 🗂️ Arquivos Gerados (não versionados)

```
data/processed/flights_model.parquet   # ~500 MB
outputs/relatorio.html                 # ~2 MB
outputs/pickles/eda.pkl
outputs/pickles/clf.pkl                # ~XX MB (Random Forest)
outputs/pickles/reg.pkl
outputs/pickles/clusterer.pkl
outputs/pickles/mapper.pkl
outputs/pickles/detector.pkl
outputs/pickles/season.pkl
outputs/pickles/folium_map.html
mapa_atrasos_heatmap.html
```

---

## 📚 Referências

- [U.S. DOT — Bureau of Transportation Statistics](https://www.transtats.bts.gov/)
- [Kaggle — 2015 Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays)
- [Polars Documentation](https://docs.pola.rs/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

---

## 🎓 Contexto Acadêmico

**Curso:** FIAP Postech — Engenharia de Machine Learning  
**Desafio:** Tech Challenge — Fase 3  
**Tema:** Análise preditiva de atrasos em aviação civil

---

*Desenvolvido como parte do Tech Challenge da FIAP Postech — 2025/2026*
