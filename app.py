import streamlit as st
import matplotlib.pyplot as plt
from tools import available_subsets, load_subset, convert_to_gpd

st.set_page_config(page_title="Building Subset Viewer", layout="wide")
st.title("🏗️ Explorador de Subsets de Edificios")

# Panel lateral
st.sidebar.header("Opciones")
subsets = available_subsets()

if not subsets:
    st.error("⚠️ No se encontraron subsets en la carpeta definida.")
    st.stop()

selected_subset = st.sidebar.selectbox("Seleccioná un subset", subsets)

if selected_subset:
    st.sidebar.success(f"Subset seleccionado: `{selected_subset}`")

    # Cargar datos
    df = load_subset(selected_subset)
    gdf = convert_to_gpd(df)

    # Mostrar métricas básicas
    st.subheader("📊 Información del Dataset")
    st.markdown(f"- Edificios: **{len(gdf)}**")
    st.markdown(f"- Sistema de referencia (CRS): **{gdf.crs.to_string()}**")

    # Mapa de edificios
    st.subheader("🗺️ Mapa de Edificios")
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color="lightblue", edgecolor="black", alpha=0.6)
    ax.set_title(f"Subset: {selected_subset}")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    st.pyplot(fig)
