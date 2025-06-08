# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 12:43:33 2025

@author: Kaan
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import unidecode
import matplotlib.colors as mcolors
import logging

# Log ayarları
logging.getLogger('streamlit.runtime.scriptrunner').setLevel(logging.ERROR)

st.set_page_config(layout="wide")
st.title("🇹🇷 Türkiye'de K-Means Kümelenme Analizi")

# --- 1. Veri Yükleme ---
df = pd.read_excel("turkiyemm.xlsx")
st.success("Veri başarıyla yüklendi.")

# --- 2. Eksik Veri İmputasyonu ---
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

# --- 3. Label Encoding ---
label_encoder = LabelEncoder()
for col in ['most_edu_level', 'most_voted_party', 'region']:
    df[col] = label_encoder.fit_transform(df[col])

# --- 4. Sayısal Veriler Seçimi ---
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numerical_cols].apply(pd.to_numeric, errors='coerce').dropna()
df_clean = df.loc[X.index].copy()

# --- 5. Normalize Et ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 6. Elbow Yöntemi ile En Uygun k Belirleme ---
st.subheader("📌 En Uygun Küme Sayısı (Elbow Yöntemi)")
inertia_list = []
K_range = range(1, 11)
for i in K_range:
    km = KMeans(n_clusters=i, random_state=42, n_init='auto')
    km.fit(X_scaled)
    inertia_list.append(km.inertia_)

# Elbow grafiği çizimi
fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(K_range, inertia_list, marker='o')
ax_elbow.set_xlabel("Küme Sayısı (k)")
ax_elbow.set_ylabel("Inertia")
ax_elbow.set_title("Elbow Yöntemi ile En Uygun k")
st.pyplot(fig_elbow)


# --- 7. Kullanıcıdan k değeri al ---
k = st.slider("Kaç küme (k) olsun?", 2, 10, 3)

# --- 8. K-means Uygulama ---
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
df_clean["cluster"] = kmeans.fit_predict(X_scaled)

# --- 9. Silhouette Skoru ---
score = silhouette_score(X_scaled, df_clean["cluster"])
st.write(f"📈 **Silhouette Skoru (k={k})**: `{score:.2f}`")

# --- 10. Küme Özet Tablosu ---
st.subheader("🔍 Küme Özeti")
cluster_summary = df_clean.groupby("cluster")[numerical_cols].mean().round(2)
st.dataframe(cluster_summary)

# --- 11. Harita Verisi ve Birleştirme ---
gdf = gpd.read_file("turkiye_il.geojson")
gdf["name_lower"] = gdf["shapeName"].apply(lambda x: unidecode.unidecode(x.lower()))
df_clean["country_name_lower"] = df_clean["country_name"].apply(lambda x: unidecode.unidecode(x.lower()))

merged = gdf.merge(df_clean, left_on="name_lower", right_on="country_name_lower")

# --- 12. Harita Çizimi ---
colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3', '#ffff33']
cmap = mcolors.ListedColormap(colors[:k])

fig, ax = plt.subplots(figsize=(10, 10))
merged.plot(column="cluster", cmap=cmap, legend=True, ax=ax, edgecolor='black', linewidth=0.5)
ax.set_title(f"K-means Türkiye Haritası (k={k})", fontsize=14)
ax.axis("off")
st.pyplot(fig)
