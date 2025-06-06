# streamlit_app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import unidecode
import matplotlib.colors as mcolors

st.title("🇹🇷 Türkiye K-means Kümeleme Haritası")

# --- Veri Yükleme ---
df = pd.read_excel("turkiyemm.xlsx")
missing_values = df.isna().sum()
print(missing_values)
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(df.isna().sum())

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_df = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
X = numeric_df.dropna()
df_clean = df.loc[X.index].copy()

numeric_df.isna().sum()

# --- Küme Sayısı Ayarı ---
k = st.slider("Kaç küme (k) olsun?", 2, 6, 3)

# --- Veriyi Ölçeklendir ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- K-means Kümeleme ---
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


# Harita verisindeki tüm iller
geo_names = set(gdf["name_lower"])

# Excel verisindeki tüm iller
data_names = set(df_clean["country_name_lower"])

# Eksik kalanlar
missing = geo_names - data_names
print("Haritada gösterilemeyen iller:", missing)
print("Excel'deki il isimleri:")
print(sorted(df["country_name"].unique()))

df["country_name_lower"] = df["country_name"].apply(lambda x: unidecode.unidecode(str(x).lower()))
geo_names = set(gdf["shapeName"].apply(lambda x: unidecode.unidecode(x.lower())))
excel_names = set(df["country_name_lower"])

missing_from_excel = geo_names - excel_names

print("Excel'de bulunmayan il isimleri:", missing_from_excel)
