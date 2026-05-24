import polars as pl
import pandas as pd
import numpy as np
import joblib
from io import StringIO
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")


# Coordenadas embutidas dos principais aeroportos dos EUA (IATA, name, lat, lon, state)
AIRPORT_COORDS = """iata,name,lat,lon,state
ATL,Hartsfield-Jackson Atlanta,33.6367,-84.4281,GA
LAX,Los Angeles International,33.9425,-118.4081,CA
ORD,Chicago O'Hare,41.9742,-87.9073,IL
DFW,Dallas/Fort Worth,32.8998,-97.0403,TX
DEN,Denver International,39.8561,-104.6737,CO
JFK,John F. Kennedy,40.6413,-73.7781,NY
SFO,San Francisco,37.6213,-122.3790,CA
SEA,Seattle-Tacoma,47.4502,-122.3088,WA
LAS,Las Vegas Harry Reid,36.0840,-115.1537,NV
MCO,Orlando International,28.4294,-81.3089,FL
EWR,Newark Liberty,40.6925,-74.1687,NJ
MIA,Miami International,25.7959,-80.2870,FL
PHX,Phoenix Sky Harbor,33.4373,-112.0078,AZ
IAH,Houston George Bush,29.9902,-95.3368,TX
BOS,Boston Logan,42.3656,-71.0096,MA
MSP,Minneapolis-Saint Paul,44.8848,-93.2223,MN
DTW,Detroit Metropolitan,42.2124,-83.3534,MI
FLL,Fort Lauderdale,26.0726,-80.1527,FL
PHL,Philadelphia International,39.8729,-75.2437,PA
LGA,LaGuardia,40.7772,-73.8726,NY
BWI,Baltimore/Washington,39.1754,-76.6683,MD
DCA,Ronald Reagan Washington,38.8521,-77.0377,VA
MDW,Chicago Midway,41.7868,-87.7522,IL
TPA,Tampa International,27.9755,-82.5332,FL
SLC,Salt Lake City,40.7884,-111.9778,UT
IAD,Washington Dulles,38.9445,-77.4558,VA
HNL,Daniel K. Inouye Honolulu,21.3245,-157.9251,HI
PDX,Portland International,45.5898,-122.5951,OR
DAL,Dallas Love Field,32.8471,-96.8518,TX
HOU,Houston William P. Hobby,29.6454,-95.2789,TX
STL,St. Louis Lambert,38.7487,-90.3700,MO
CLE,Cleveland Hopkins,41.4117,-81.8498,OH
MCI,Kansas City,39.2976,-94.7139,MO
SAN,San Diego International,32.7338,-117.1933,CA
AUS,Austin-Bergstrom,30.1945,-97.6699,TX
BNA,Nashville International,36.1245,-86.6782,TN
RDU,Raleigh-Durham,35.8776,-78.7875,NC
MEM,Memphis International,35.0424,-89.9767,TN
SMF,Sacramento International,38.6954,-121.5908,CA
MSY,Louis Armstrong New Orleans,29.9934,-90.2580,LA
SJC,San Jose International,37.3626,-121.9290,CA
OAK,Oakland International,37.7213,-122.2208,CA
RSW,Southwest Florida,26.5362,-81.7552,FL
PIT,Pittsburgh International,40.4915,-80.2329,PA
CVG,Cincinnati/Northern Kentucky,39.0488,-84.6678,KY
IND,Indianapolis International,39.7173,-86.2944,IN
CMH,John Glenn Columbus,39.9980,-82.8919,OH
JAX,Jacksonville International,30.4941,-81.6879,FL
MKE,Milwaukee Mitchell,42.9472,-87.8966,WI
OMA,Eppley Airfield Omaha,41.3032,-95.8941,NE
ABQ,Albuquerque International,35.0402,-106.6090,NM
BUF,Buffalo Niagara,42.9405,-78.7322,NY
SNA,John Wayne Orange County,33.6757,-117.8682,CA
BOI,Boise Airport,43.5644,-116.2228,ID
TUS,Tucson International,32.1161,-110.9410,AZ
ELP,El Paso International,31.8072,-106.3779,TX
OKC,Will Rogers Oklahoma City,35.3931,-97.6007,OK
ONT,Ontario International,34.0560,-117.6012,CA
ANC,Ted Stevens Anchorage,61.1744,-149.9964,AK
GEG,Spokane International,47.6199,-117.5338,WA
BDL,Bradley International,41.9389,-72.6832,CT
PVD,T.F. Green Providence,41.7325,-71.4282,RI
ALB,Albany International,42.7483,-73.8017,NY
CHS,Charleston International,32.8986,-80.0405,SC
SAV,Savannah/Hilton Head,32.1276,-81.2021,GA
GSO,Piedmont Triad Greensboro,36.0978,-79.9373,NC
SDF,Louisville Muhammad Ali,38.1744,-85.7360,KY
ORF,Norfolk International,36.8976,-76.0133,VA
RIC,Richmond International,37.5052,-77.3197,VA
ROC,Greater Rochester,43.1189,-77.6724,NY
SYR,Syracuse Hancock,43.1112,-76.1063,NY
TUL,Tulsa International,36.1984,-95.8881,OK
LIT,Bill and Hillary Clinton,34.7294,-92.2243,AR
PBI,Palm Beach International,26.6832,-80.0956,FL
DAY,Dayton International,39.9024,-84.2194,OH
GRR,Gerald R. Ford Grand Rapids,42.8808,-85.5228,MI
LGB,Long Beach,33.8177,-118.1516,CA
BUR,Hollywood Burbank,34.2007,-118.3585,CA
SJU,Luis Munoz Marin San Juan,18.4394,-66.0018,PR
FAT,Fresno Yosemite,36.7762,-119.7182,CA
TYS,McGhee Tyson Knoxville,35.8110,-83.9940,TN
BHM,Birmingham-Shuttlesworth,33.5629,-86.7535,AL
GSP,Greenville-Spartanburg,34.8957,-82.2189,SC
ICT,Wichita Dwight D. Eisenhower,37.6499,-97.4331,KS
DSM,Des Moines International,41.5340,-93.6631,IA
FSD,Sioux Falls Regional,43.5820,-96.7419,SD
MSO,Missoula Montana,46.9163,-114.0906,MT
BZN,Bozeman Yellowstone,45.7777,-111.1530,MT
BIL,Billings Logan,45.8077,-108.5428,MT
JAC,Jackson Hole,43.6073,-110.7377,WY
COS,Colorado Springs,38.8059,-104.7009,CO
SGF,Springfield-Branson,37.2457,-93.3886,MO
EUG,Eugene Airport,44.1246,-123.2119,OR
MFR,Rogue Valley Medford,42.3742,-122.8735,OR
SBN,South Bend International,41.7087,-86.3173,IN
FWA,Fort Wayne International,40.9785,-85.1951,IN
EVV,Evansville Regional,38.0369,-87.5324,IN
MLI,Quad City International,41.4485,-90.5075,IL
FNT,Bishop International Flint,42.9654,-83.7436,MI
TVC,Cherry Capital,44.7414,-85.5822,MI
GRB,Austin Straubel Green Bay,44.4851,-88.1296,WI
MSN,Dane County Regional,43.1399,-89.3375,WI
FAR,Hector International Fargo,46.9207,-96.8158,ND
RAP,Rapid City Regional,43.9723,-103.0574,SD
CPR,Casper-Natrona County,42.9080,-106.4644,WY
XNA,Northwest Arkansas National,36.2819,-94.3068,AR
FSM,Fort Smith Regional,35.3366,-94.3675,AR
PNS,Pensacola International,30.4734,-87.1866,FL
TLH,Tallahassee International,30.3965,-84.3503,FL
SRQ,Sarasota-Bradenton,27.3954,-82.5544,FL
MLB,Melbourne Orlando International,28.1028,-80.6453,FL
PIE,St. Pete-Clearwater,27.9102,-82.6874,FL
VPS,Destin-Fort Walton Beach,30.4832,-86.5254,FL
RST,Rochester International,43.9083,-92.5000,MN
DLH,Duluth International,46.8421,-92.1936,MN
AVL,Asheville Regional,35.4362,-82.5418,NC
FAY,Fayetteville Regional,34.9912,-78.8803,NC
OAJ,Albert J. Ellis Jacksonville NC,34.8292,-77.6121,NC
ROA,Roanoke-Blacksburg,37.3255,-79.9754,VA
CHO,Charlottesville-Albemarle,38.1386,-78.4529,VA
PWM,Portland International Jetport,43.6462,-70.3093,ME
BGR,Bangor International,44.8074,-68.8281,ME
BTV,Burlington International,44.4720,-73.1533,VT
MOB,Mobile Regional,30.6912,-88.2428,AL
HSV,Huntsville International,34.6372,-86.7751,AL
JAN,Jackson-Medgar Wiley Evers,32.3112,-90.0759,MS
GPT,Gulfport-Biloxi,30.4073,-89.0701,MS
BTR,Baton Rouge Metro,30.5332,-91.1496,LA
SHV,Shreveport Regional,32.4466,-93.8256,LA
LFT,Lafayette Regional,30.2053,-91.9876,LA
CID,The Eastern Iowa,41.8847,-91.7108,IA
SGU,St. George Regional,37.0363,-113.5103,UT
RDM,Redmond Airport,44.2541,-121.1500,OR
LWS,Lewiston-Nez Perce County,46.3745,-117.0153,ID
TWF,Magic Valley Regional Twin Falls,42.4818,-114.4877,ID
PIH,Pocatello Regional,42.9098,-112.5960,ID
GJT,Grand Junction Regional,39.1224,-108.5267,CO
DRO,Durango-La Plata,37.1515,-107.7538,CO
ABR,Aberdeen Regional,45.4491,-98.4218,SD
GTF,Great Falls International,47.4820,-111.3707,MT
LBB,Lubbock Preston Smith,33.6636,-101.8228,TX
MAF,Midland International,31.9425,-102.2019,TX
CRP,Corpus Christi International,27.7704,-97.5026,TX
SAT,San Antonio International,29.5337,-98.4698,TX
HRL,Valley International,26.2285,-97.6644,TX
AMA,Rick Husband Amarillo,35.2194,-101.7059,TX
LRD,Laredo International,27.5438,-99.4615,TX
GRK,Killeen-Fort Hood,31.0672,-97.8291,TX
ACT,Waco Regional,31.6113,-97.2305,TX
SPS,Wichita Falls Regional,33.9888,-98.4919,TX
TXK,Texarkana Regional,33.4537,-93.9910,TX
TYR,Tyler Pounds Regional,32.3541,-95.4024,TX
GGG,East Texas Regional,32.3840,-94.7115,TX
LCH,Lake Charles Regional,30.1261,-93.2233,LA
MLU,Monroe Regional,32.5109,-92.0377,LA
AEX,Alexandria International,31.3274,-92.5498,LA
MEI,Key Field Meridian,32.3326,-88.7519,MS
DHN,Dothan Regional,31.3213,-85.4496,AL
VLD,Valdosta Regional,30.7825,-83.2767,GA
AGS,Augusta Regional,33.3699,-81.9645,SC
CAE,Columbia Metropolitan,33.9388,-81.1195,SC
PHF,Newport News/Williamsburg,37.1319,-76.4930,VA
HPN,Westchester County,41.0670,-73.7076,NY
ISP,Long Island MacArthur,40.7952,-73.1002,NY
ACK,Nantucket Memorial,41.2531,-70.0603,MA
EYW,Key West International,24.5561,-81.7596,FL
GNV,Gainesville Regional,29.6900,-82.2717,FL
MCN,Middle Georgia Regional,32.6928,-83.6492,GA
CSG,Columbus Metropolitan,32.5163,-84.9389,GA
PIA,General Wayne A. Downing Peoria,40.6642,-89.6933,IL
BMI,Central Illinois Regional,40.4771,-88.9159,IL
MBS,MBS International,43.5329,-84.0797,MI
LAN,Capital Region International,42.7787,-84.5874,MI
LSE,La Crosse Regional,43.8790,-91.2567,WI
LNK,Lincoln Airport,40.8510,-96.7592,NE
GRI,Central Nebraska Regional,40.9675,-98.3096,NE
BIS,Bismarck Municipal,46.7727,-100.7457,ND
GFK,Grand Forks International,47.9493,-97.1761,ND
LAR,Laramie Regional,41.3121,-105.6750,WY
RKS,Southwest Wyoming Regional,41.5942,-109.0652,WY
PUB,Pueblo Memorial,38.2891,-104.4966,CO
"""


class FlightGeoMapper:
    """
    Visualizações geográficas de aeroportos, rotas e atrasos nos EUA.

    Pipeline:
        1. Carrega coordenadas embutidas dos aeroportos
        2. Lê parquet processado e agrega perfil por aeroporto de origem
        3. Junta com coordenadas (inner join — aeroportos sem coords são excluídos)
        4. Carrega/recria clusters (do notebook 04 ou via fallback)
        5. Gera mapas Plotly e Folium

    Mapas gerados:
        - map_delay_rate: scatter geo por taxa de atraso (cor = delay, tamanho = volume)
        - map_clusters: scatter geo por cluster
        - map_routes: linhas das top rotas, coloridas por delay médio
        - map_heatmap: heatmap folium dos atrasos graves
        - map_state_choropleth: choropleth por estado
    """

    AGG_FEATURES = [
        "media_arrival_delay",
        "taxa_atraso",
        "taxa_atraso_grave",
        "std_arrival_delay",
        "n_destinos",
        "qtd_voos",
    ]

    def __init__(
        self,
        input_path=None,
        kmeans_path=None,
        scaler_path=None,
        min_voos=500,
        n_clusters_fallback=4,
        top_n_routes=150,
        random_state=42,
    ):
        """
        Parameters:
        - input_path (str or Path): Path to the processed flights parquet file.
        - kmeans_path (str or Path): Caminho opcional pro KMeans salvo (notebook 04).
        - scaler_path (str or Path): Caminho opcional pro StandardScaler salvo (notebook 04).
        - min_voos (int): Mínimo de voos para um aeroporto entrar no mapa.
        - n_clusters_fallback (int): k usado se não houver modelo salvo.
        - top_n_routes (int): Quantas rotas mostrar no mapa de linhas.
        - random_state (int): Semente para reprodutibilidade.
        """
        self.input_path = Path(input_path) if input_path else None
        self.kmeans_path = Path(kmeans_path) if kmeans_path else None
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.min_voos = min_voos
        self.n_clusters_fallback = n_clusters_fallback
        self.top_n_routes = top_n_routes
        self.random_state = random_state

        # Dados
        self.coords = None
        self.df = None
        self.airport_stats = None
        self.ap_map = None
        self.routes = None
        self.state_stats = None
        self.cluster_labels = None

        # Mapas gerados
        self.fig_delay_rate = None
        self.fig_clusters = None
        self.fig_routes = None
        self.fig_state_choropleth = None
        self.folium_map = None

    # ------------------------------------------------------------------
    # 1. Coordenadas
    # ------------------------------------------------------------------
    def load_coords(self):
        """Carrega a tabela embutida de coordenadas dos aeroportos."""
        self.coords = pd.read_csv(StringIO(AIRPORT_COORDS)).drop_duplicates(subset="iata")
        print(f"Aeroportos com coordenadas: {len(self.coords)}")
        return self.coords

    # ------------------------------------------------------------------
    # 2. Carregamento e agregação
    # ------------------------------------------------------------------
    def load_data(self):
        """Carrega o parquet processado."""
        print(f"Carregando dados de: {self.input_path} ...")
        print(f"Existe? {self.input_path.exists()}")

        self.df = pl.read_parquet(self.input_path)
        print(f"Shape: {self.df.shape}")
        return self.df

    def aggregate_airports(self):
        """Agrega estatísticas por aeroporto de origem e junta com coordenadas."""
        if self.df is None:
            self.load_data()
        if self.coords is None:
            self.load_coords()

        self.airport_stats = (
            self.df.filter(pl.col("ARRIVAL_DELAY").is_not_null())
            .group_by("ORIGIN_AIRPORT")
            .agg(
                [
                    pl.len().alias("qtd_voos"),
                    pl.col("ARRIVAL_DELAY").mean().alias("media_arrival_delay"),
                    (pl.col("ARRIVAL_DELAY") > 0).mean().alias("taxa_atraso"),
                    (pl.col("ARRIVAL_DELAY") > 15).mean().alias("taxa_atraso_grave"),
                    pl.col("ARRIVAL_DELAY").std().alias("std_arrival_delay"),
                    pl.col("DESTINATION_AIRPORT").n_unique().alias("n_destinos"),
                ]
            )
            .filter(pl.col("qtd_voos") >= self.min_voos)
            .to_pandas()
        )

        # Inner join com coordenadas
        self.ap_map = self.airport_stats.merge(
            self.coords, left_on="ORIGIN_AIRPORT", right_on="iata", how="inner"
        )

        print(
            f"Aeroportos com coords: {len(self.ap_map)} "
            f"de {len(self.airport_stats)} no dataset"
        )
        return self.ap_map

    # ------------------------------------------------------------------
    # 3. Clusters (modelo salvo ou fallback)
    # ------------------------------------------------------------------
    def assign_clusters(self):
        """Atribui cluster a cada aeroporto. Tenta carregar modelo salvo;
        se não houver, recluster com KMeans local."""
        if self.ap_map is None:
            self.aggregate_airports()

        try:
            if self.kmeans_path is None or self.scaler_path is None:
                raise FileNotFoundError("Caminhos de modelo não informados")

            km = joblib.load(self.kmeans_path)
            scaler = joblib.load(self.scaler_path)

            feature_cols = [
                "media_arrival_delay",
                "taxa_atraso",
                "taxa_atraso_grave",
                "std_arrival_delay",
                "n_destinos",
            ]
            X = self.ap_map[feature_cols].fillna(self.ap_map[feature_cols].median())
            self.ap_map["cluster"] = km.predict(scaler.transform(X))
            print("✓ Clusters carregados do modelo salvo")

        except (FileNotFoundError, OSError):
            print("Modelo não encontrado — reclusterizando localmente...")
            X = self.ap_map[self.AGG_FEATURES].fillna(
                self.ap_map[self.AGG_FEATURES].median()
            )
            scaler_local = StandardScaler()
            X_scaled = scaler_local.fit_transform(X)

            km_local = KMeans(
                n_clusters=self.n_clusters_fallback,
                random_state=self.random_state,
                n_init="auto",
            )
            self.ap_map["cluster"] = km_local.fit_predict(X_scaled)
            print(f"✓ Reclusterizado com k={self.n_clusters_fallback}")

        self.ap_map["cluster_str"] = self.ap_map["cluster"].astype(str)

        # Rótulos padrão (podem ser sobrescritos via set_cluster_labels)
        self.cluster_labels = {
            str(c): f"Cluster {c}" for c in sorted(self.ap_map["cluster"].unique())
        }
        self.ap_map["perfil"] = self.ap_map["cluster_str"].map(self.cluster_labels)

        print(self.ap_map["cluster"].value_counts().sort_index())
        return self.ap_map

    def set_cluster_labels(self, labels_dict):
        """Permite renomear os clusters após inspeção.
        Ex: mapper.set_cluster_labels({'0': 'Hubs problemáticos', '1': 'Regionais', ...})
        """
        if self.ap_map is None:
            self.assign_clusters()
        self.cluster_labels = labels_dict
        self.ap_map["perfil"] = self.ap_map["cluster_str"].map(self.cluster_labels)
        print("Rótulos atualizados:", self.cluster_labels)

    # ------------------------------------------------------------------
    # 4. Mapa 1: Aeroportos por taxa de atraso
    # ------------------------------------------------------------------
    def build_map_delay_rate(self):
        """Scatter geo: aeroportos coloridos por taxa de atraso, tamanho = volume."""
        if self.ap_map is None:
            self.assign_clusters()

        self.fig_delay_rate = px.scatter_geo(
            self.ap_map,
            lat="lat",
            lon="lon",
            color="taxa_atraso",
            size="qtd_voos",
            size_max=40,
            hover_name="ORIGIN_AIRPORT",
            hover_data={
                "name": True,
                "taxa_atraso": ":.1%",
                "media_arrival_delay": ":.1f",
                "qtd_voos": ":,",
                "lat": False,
                "lon": False,
            },
            color_continuous_scale="RdYlGn_r",
            scope="usa",
            title="Taxa de Atraso por Aeroporto (tamanho = volume de voos)",
            labels={"taxa_atraso": "Taxa de atraso"},
            template="plotly_white",
        )
        self.fig_delay_rate.update_layout(
            coloraxis_colorbar=dict(tickformat=".0%")
        )
        return self.fig_delay_rate

    # ------------------------------------------------------------------
    # 5. Mapa 2: Aeroportos por cluster
    # ------------------------------------------------------------------
    def build_map_clusters(self):
        """Scatter geo: aeroportos coloridos pelo cluster (perfil operacional)."""
        if self.ap_map is None:
            self.assign_clusters()

        self.fig_clusters = px.scatter_geo(
            self.ap_map,
            lat="lat",
            lon="lon",
            color="perfil",
            size="qtd_voos",
            size_max=40,
            hover_name="ORIGIN_AIRPORT",
            hover_data={
                "name": True,
                "taxa_atraso": ":.1%",
                "media_arrival_delay": ":.1f",
                "qtd_voos": ":,",
                "lat": False,
                "lon": False,
            },
            scope="usa",
            title="Clusters de Aeroportos — Perfil Operacional",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        return self.fig_clusters

    # ------------------------------------------------------------------
    # 6. Mapa 3: Rotas mais movimentadas coloridas por delay
    # ------------------------------------------------------------------
    def build_routes_dataframe(self):
        """Constrói o dataframe das top N rotas com coordenadas de origem e destino."""
        if self.df is None:
            self.load_data()
        if self.coords is None:
            self.load_coords()

        routes = (
            self.df.filter(pl.col("ARRIVAL_DELAY").is_not_null())
            .group_by(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])
            .agg(
                [
                    pl.len().alias("qtd_voos"),
                    pl.col("ARRIVAL_DELAY").mean().alias("media_delay"),
                ]
            )
            .sort("qtd_voos", descending=True)
            .head(self.top_n_routes)
            .to_pandas()
        )

        # Juntar coords da origem e do destino
        routes = routes.merge(
            self.coords[["iata", "lat", "lon"]],
            left_on="ORIGIN_AIRPORT",
            right_on="iata",
            how="inner",
        ).rename(columns={"lat": "lat_orig", "lon": "lon_orig"})

        routes = routes.merge(
            self.coords[["iata", "lat", "lon"]],
            left_on="DESTINATION_AIRPORT",
            right_on="iata",
            how="inner",
        ).rename(columns={"lat": "lat_dest", "lon": "lon_dest"})

        self.routes = routes
        print(f"Rotas com coords completas: {len(self.routes)}")
        return self.routes

    @staticmethod
    def _delay_to_color(norm_val):
        """Vermelho = muito atraso, verde = pouco atraso."""
        r = int(220 * norm_val)
        g = int(200 * (1 - norm_val))
        return f"rgba({r},{g},60,0.55)"

    def build_map_routes(self):
        """Mapa de linhas com as top rotas coloridas por delay médio."""
        if self.routes is None:
            self.build_routes_dataframe()
        if self.ap_map is None:
            self.assign_clusters()

        # Normalizar delay para cor
        delay_min = self.routes["media_delay"].min()
        delay_max = self.routes["media_delay"].max()
        self.routes["delay_norm"] = (
            self.routes["media_delay"] - delay_min
        ) / (delay_max - delay_min + 1e-9)

        max_voos = self.routes["qtd_voos"].max()

        self.fig_routes = go.Figure()

        for _, row in self.routes.iterrows():
            self.fig_routes.add_trace(
                go.Scattergeo(
                    lon=[row["lon_orig"], row["lon_dest"]],
                    lat=[row["lat_orig"], row["lat_dest"]],
                    mode="lines",
                    line=dict(
                        width=max(0.5, row["qtd_voos"] / max_voos * 4),
                        color=self._delay_to_color(row["delay_norm"]),
                    ),
                    hoverinfo="text",
                    text=(
                        f"{row['ORIGIN_AIRPORT']} → {row['DESTINATION_AIRPORT']}<br>"
                        f"Voos: {row['qtd_voos']:,}<br>"
                        f"Delay médio: {row['media_delay']:.1f} min"
                    ),
                    showlegend=False,
                )
            )

        # Pontos dos aeroportos por cima
        self.fig_routes.add_trace(
            go.Scattergeo(
                lon=self.ap_map["lon"],
                lat=self.ap_map["lat"],
                mode="markers",
                marker=dict(
                    size=5, color="white", line=dict(width=0.5, color="gray")
                ),
                hoverinfo="text",
                text=self.ap_map["ORIGIN_AIRPORT"] + " — " + self.ap_map["name"],
                showlegend=False,
            )
        )

        self.fig_routes.update_layout(
            title="Top rotas mais movimentadas — cor por delay médio "
            "(vermelho=pior, verde=melhor)",
            geo=dict(
                scope="usa",
                projection_type="albers usa",
                showland=True,
                landcolor="rgb(30,30,40)",
                showlakes=True,
                lakecolor="rgb(20,20,30)",
                bgcolor="rgb(15,15,25)",
            ),
            paper_bgcolor="rgb(15,15,25)",
            font_color="white",
            height=600,
        )
        return self.fig_routes

    # ------------------------------------------------------------------
    # 7. Mapa 4: Heatmap interativo (Folium)
    # ------------------------------------------------------------------
    def build_map_heatmap(self, save_path="mapa_atrasos_heatmap.html"):
        """Mapa Folium com camada de calor por taxa de atraso grave."""
        if self.ap_map is None:
            self.assign_clusters()

        m = folium.Map(
            location=[37.0, -95.7], zoom_start=4, tiles="CartoDB dark_matter"
        )

        # Heatmap pesado por taxa de atraso grave
        heat_data = [
            [row["lat"], row["lon"], row["taxa_atraso_grave"]]
            for _, row in self.ap_map.iterrows()
        ]

        HeatMap(
            heat_data,
            min_opacity=0.3,
            radius=25,
            blur=20,
            gradient={0.3: "blue", 0.6: "yellow", 1.0: "red"},
        ).add_to(m)

        # Marcadores individuais clicáveis
        for _, row in self.ap_map.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                color="white",
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{row['ORIGIN_AIRPORT']}</b> — {row['name']}<br>"
                    f"Taxa atraso: {row['taxa_atraso']:.1%}<br>"
                    f"Atraso grave (>15min): {row['taxa_atraso_grave']:.1%}<br>"
                    f"Delay médio: {row['media_arrival_delay']:.1f} min<br>"
                    f"Voos: {int(row['qtd_voos']):,}",
                    max_width=220,
                ),
                tooltip=row["ORIGIN_AIRPORT"],
            ).add_to(m)

        if save_path:
            m.save(save_path)
            print(f"Mapa salvo em {save_path}")

        self.folium_map = m
        return self.folium_map

    # ------------------------------------------------------------------
    # 8. Mapa 5: Choropleth por estado (bônus)
    # ------------------------------------------------------------------
    def build_map_state_choropleth(self):
        """Choropleth com taxa de atraso média agregada por estado."""
        if self.ap_map is None:
            self.assign_clusters()

        self.state_stats = (
            self.ap_map.groupby("state")
            .agg(
                taxa_atraso=("taxa_atraso", "mean"),
                media_delay=("media_arrival_delay", "mean"),
                n_aeroportos=("ORIGIN_AIRPORT", "count"),
            )
            .reset_index()
        )

        self.fig_state_choropleth = px.choropleth(
            self.state_stats,
            locations="state",
            locationmode="USA-states",
            color="taxa_atraso",
            scope="usa",
            color_continuous_scale="RdYlGn_r",
            hover_data={
                "taxa_atraso": ":.1%",
                "media_delay": ":.1f",
                "n_aeroportos": True,
            },
            title="Taxa de Atraso Média por Estado",
            labels={
                "taxa_atraso": "Taxa de atraso",
                "media_delay": "Delay médio (min)",
            },
            template="plotly_white",
        )
        self.fig_state_choropleth.update_layout(
            coloraxis_colorbar=dict(tickformat=".0%")
        )
        return self.fig_state_choropleth

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------
    def run_all(self, show_plots=True):
        """Constrói todos os mapas de uma vez. Os figs ficam acessíveis via atributos."""
        self.load_coords()
        self.load_data()
        self.aggregate_airports()
        self.assign_clusters()
        self.build_map_delay_rate()
        self.build_map_clusters()
        self.build_routes_dataframe()
        self.build_map_routes()
        self.build_map_heatmap()
        self.build_map_state_choropleth()

        if show_plots:
            self.fig_delay_rate.show()
            self.fig_clusters.show()
            self.fig_routes.show()
            self.fig_state_choropleth.show()
            # Folium map é exibido inline ao retornar do método


if __name__ == "__main__":
    mapper = FlightGeoMapper(
        input_path="../data/processed/flights_model.parquet",
    )
    mapper.run_all(show_plots=True)
