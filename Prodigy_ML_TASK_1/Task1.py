import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Page config & clean UI
# -------------------------
st.set_page_config(page_title="House Price Prediction", layout="wide")

# Hide Streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -------------------------
# Title
# -------------------------
st.title("🏠 House Price Prediction Dashboard")
st.write("Interactive Linear Regression model to predict house prices")

# -------------------------
# Hardcoded dataset
# -------------------------
data = pd.DataFrame({
    "SquareFeet": [800,1000,1200,1500,1800,2000,2200,2500],
    "Bedrooms": [1,2,2,3,3,4,4,5],
    "Bathrooms": [1,1,2,2,3,3,4,4],
    "Price": [4000000,5200000,6500000,8200000,9800000,11000000,12500000,15000000]
})

# -------------------------
# Train Linear Regression
# -------------------------
X = data[['SquareFeet', 'Bedrooms', 'Bathrooms']]
y = data['Price']
model = LinearRegression()
model.fit(X, y)
data['PredictedPrice'] = model.predict(X)

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("🏡 Enter House Details")
sqft = st.sidebar.slider("Square Feet", 500, 3000, 1200)
beds = st.sidebar.selectbox("Bedrooms", sorted(data['Bedrooms'].unique()), index=1)
baths = st.sidebar.selectbox("Bathrooms", sorted(data['Bathrooms'].unique()), index=1)

# Prediction
pred_price = model.predict([[sqft, beds, baths]])[0]

# -------------------------
# Metric card + Random Sample
# -------------------------
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("💰 Predicted Price")
    st.metric(label="Estimated House Price", value=f"₹ {pred_price:,.0f}")
with col2:
    if st.button("🎲 Show Random House Sample"):
        st.write(data.sample(1))

# -------------------------
# Side-by-side plots (2x2)
# -------------------------
col1, col2 = st.columns(2)

# Column 1: Scatter + Regression
with col1:
    fig_scatter = px.scatter(data, x='SquareFeet', y='Price', size='Bedrooms', color='Bathrooms',
                             title="📊 Price vs Square Feet", width=600, height=400)
    fig_scatter.add_traces(go.Scatter(x=data['SquareFeet'], y=data['PredictedPrice'],
                                      mode='lines', name='Regression Line', line=dict(color='red')))
    st.plotly_chart(fig_scatter, use_container_width=True)

# Column 2: Histogram
with col2:
    fig_hist = px.histogram(data, x='Price', nbins=8, title="📈 Distribution of Prices", 
                            color_discrete_sequence=['orange'], width=600, height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

# Next row: 3D Scatter + Predicted vs Actual
col3, col4 = st.columns(2)

# 3D Scatter
with col3:
    fig_3d = px.scatter_3d(data, x='SquareFeet', y='Bedrooms', z='Price',
                           size='Bathrooms', color='Price', title="🌐 3D View", width=600, height=400)
    st.plotly_chart(fig_3d, use_container_width=True)

# Predicted vs Actual Line Chart
with col4:
    fig_line = px.line(data, y=['Price','PredictedPrice'], markers=True, 
                       title="📉 Predicted vs Actual", width=600, height=400)
    st.plotly_chart(fig_line, use_container_width=True)

st.caption("Built with Streamlit & Plotly. Compact layout for single screenshot.")
