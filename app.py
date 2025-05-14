import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from tools import (
    available_subsets,
    load_subset,
    convert_to_gpd,
    convert_to_metric,
    get_area_center,
)
from settings import CONSTANTS

st.set_page_config(page_title="Building Orientation Explorer", layout="wide")

st.title("🏗️ Building Subset Viewer")

# Sidebar
st.sidebar.header("Options")
subset_options = available_subsets()
selected_subset = st.sidebar.selectbox("Select a subset", subset_options)

if selected_subset:
    st.sidebar.success(f"Selected: {selected_subset}")
    
    # Load data
    raw_df = load_subset(selected_subset)
    gdf = convert_to_gpd(raw_df)
    gdf_metric = convert_to_metric(gdf)

    # Display metrics
    st.subheader("📊 Dataset Info")
    st.markdown(f"- **Buildings**: {len(gdf)}")
    st.markdown(f"- **CRS**: {gdf.crs.to_string()}")
    st.markdown(f"- **Area Center**: {get_area_center(gdf)}")

    # Show map
    st.subheader("🗺️ Buildings Map")
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color="lightblue", edgecolor="black", alpha=0.5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    st.pyplot(fig)
