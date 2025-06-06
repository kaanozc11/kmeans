# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 21:13:37 2025

@author: Kaan
"""

# app.py
from shiny import App, render, ui, reactive
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import unidecode

# Veri yükle
df = pd.read_excel("turkiyemm.xlsx")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_df = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
df["country_name_lower"] = df["country_name"].apply(lambda x: unidecode.unidecode(str(x).lower()))
numeric_df = numeric_df.fillna(numeric_df.mean())  # Eksik veri düzeltme
X = numeric_df
df_clean = df.copy()

# Harita verisi
gdf = gpd.read_file("turkiye_il.geojson")
gdf["name_lower"] = gdf["shapeName"].apply(lambda x: unidecode.unidecode(x.lower()))

# Arayüz
app_ui = ui.page_fluid(
    ui.h2("🇹🇷 K-Ortalamalar Algoritması ile Kümelenmiş Türkiye Haritası"),
    ui.input_slider("k", "Kaç küme (k)?", min=2, max=6, value=3),
    ui.output_plot("kmeans_plot", width="800px", height="800px")
)

# Sunucu
def server(input, output, session):

    @reactive.Calc
    def clustered_data():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=input.k(), random_state=42)
        df_clean["cluster"] = kmeans.fit_predict(X_scaled)
        merged = gdf.merge(df_clean, left_on="name_lower", right_on="country_name_lower", how="left")
        return merged

    @output
    @render.plot
    def kmeans_plot():
        merged = clustered_data()
        fig, ax = plt.subplots(figsize=(10, 10))
        cmap = plt.get_cmap("Set1", input.k())
        merged.plot(column="cluster", cmap=cmap, legend=True, ax=ax, edgecolor='black', missing_kwds={
            "color": "lightgrey", "hatch": "///", "label": "Veri Yok"
        })
        ax.set_title(f"K-means Türkiye Haritası (k={input.k()})", fontsize=14)
        ax.axis("off")
        return fig

app = App(app_ui, server)
