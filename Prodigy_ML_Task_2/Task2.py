import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Hide Streamlit menu/footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🛍️ Customer Segmentation Dashboard")
st.write("Interactive K-Means clustering for retail customers")

# ------------------------
# Hardcoded dataset
# ------------------------
data = pd.DataFrame({
    "CustomerID": [1,2,3,4,5,6,7,8,9,10],
    "TotalPurchase": [500, 1500, 2000, 800, 1200, 2500, 3000, 700, 1800, 2200],
    "PurchaseFrequency": [5, 15, 20, 8, 12, 25, 30, 7, 18, 22],
    "AverageSpend": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
})

# ------------------------
# Sidebar: Number of clusters
# ------------------------
st.sidebar.header("⚙️ Cluster Settings")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 5, 3)

# ------------------------
# K-Means clustering
# ------------------------
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
features = ['TotalPurchase', 'PurchaseFrequency']  # 2D for plotting
data['Cluster'] = kmeans.fit_predict(data[features])

# ------------------------
# Metrics
# ------------------------
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("🎯 Cluster Summary")
    st.dataframe(data.groupby('Cluster')[features].mean().round(2))
with col2:
    st.subheader("📊 Cluster Sizes")
    st.bar_chart(data['Cluster'].value_counts().sort_index())

# ------------------------
# Side-by-side plots
# ------------------------
col1, col2 = st.columns(2)

# Scatter plot of clusters
with col1:
    fig_scatter = px.scatter(data, x='TotalPurchase', y='PurchaseFrequency', 
                             color='Cluster', hover_data=['CustomerID'], 
                             title="📌 Customer Clusters", width=600, height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

# Optional: 3D Scatter if 3 features
with col2:
    fig_3d = px.scatter_3d(data, x='TotalPurchase', y='PurchaseFrequency', z='AverageSpend',
                           color='Cluster', hover_data=['CustomerID'], title="🌐 3D Cluster View",
                           width=600, height=400)
    st.plotly_chart(fig_3d, use_container_width=True)

st.caption("Built with Streamlit & Plotly. Compact layout for single screenshot.")
