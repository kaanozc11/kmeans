# streamlit_app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import unidecode
import matplotlib.colors as mcolors
import logging
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)


st.title("🇹🇷 K-Ortalamalar Algoritması ile Türkiye Haritasının Kümelenmesi")

# Read the Dataset
df = pd.read_excel("turkiyemm.xlsx")

#Missing Data Imputation
missing_values = df.isna().sum()
print(missing_values)
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(df.isna().sum())

#Converting Categorical Data to Numerical with Label Encoder
label_encoder = LabelEncoder()
df['most_edu_level'].unique()
df['most_edu_level']= label_encoder.fit_transform(df['most_edu_level'])
df['most_edu_level'].unique()

df['most_voted_party'].unique()
df['most_voted_party']= label_encoder.fit_transform(df['most_voted_party'])
df['most_voted_party'].unique()

df["region"].unique()
df['region']= label_encoder.fit_transform(df['region'])
df['region'].unique()


numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_df = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
X = numeric_df.dropna()
df_clean = df.loc[X.index].copy()



#Cluster Slider
k = st.slider("Kaç küme (k) olsun?", 2, 6, 5)

# --- Normalize the Data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- K-means Clustering ---
kmeans = KMeans(n_clusters=k, random_state=42)
df_clean["cluster"] = kmeans.fit_predict(X_scaled)

# --- Harita Verisi (GeoJSON) ---
gdf = gpd.read_file("turkiye_il.geojson")

# --- İsimleri Normalize Et ---
gdf["name_lower"] = gdf["shapeName"].apply(lambda x: unidecode.unidecode(x.lower()))
df_clean["country_name_lower"] = df_clean["country_name"].apply(lambda x: unidecode.unidecode(x.lower()))

# --- Birleştir ---
merged = gdf.merge(df_clean, left_on="name_lower", right_on="country_name_lower")

# --- Renk Ayarları ---
colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3', '#ffff33']  # En fazla 6 küme için
cmap = mcolors.ListedColormap(colors[:k])

# --- Harita Çizimi ---
fig, ax = plt.subplots(figsize=(10, 10))
merged.plot(column="cluster", cmap=cmap, legend=True, ax=ax, edgecolor='black', linewidth=0.5)
ax.set_title(f"K-means Türkiye Haritası (k={k})", fontsize=14)
ax.axis("off")

st.pyplot(fig)

cluster_summary = df_clean.groupby("cluster")[numerical_cols].mean()
st.dataframe(cluster_summary)


from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, df_clean["cluster"])
st.write(f"Silhouette Skoru: {score:.2f}")
print(f"Silhouette Skoru: {score:.2f}")
