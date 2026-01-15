# =====================================================
# Ames Housing Price Prediction - Styled Dashboard
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -----------------------------------------------------
# Page Config
# -----------------------------------------------------
st.set_page_config(
    page_title="Ames Housing Price Predictor",
    layout="wide"
)

# -----------------------------------------------------
# Custom CSS Styling
# -----------------------------------------------------
st.markdown("""
<style>
/* Background */
.main {
    background: linear-gradient(135deg, #f4f7fb, #e6ecf5);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white;
}

/* Title */
h1 {
    color: #1f3b4d;
    text-align: center;
}

/* KPI cards */
.metric-container {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Section headers */
h2, h3 {
    color: #243a5e;
}

/* Dataframe */
.dataframe {
    border-radius: 10px;
}

/* Plot containers */
.stPlotlyChart {
    background: white;
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Title
# -----------------------------------------------------
st.title("🏡 Ames Housing Price Prediction")

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

data = load_data()

# -----------------------------------------------------
# Prepare Data
# -----------------------------------------------------
X = data.drop(["SalePrice", "Id"], axis=1)
y = data["SalePrice"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# -----------------------------------------------------
# Preprocessing
# -----------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------------------------------
# Model
# -----------------------------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# -----------------------------------------------------
# Train-Test Split
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# -----------------------------------------------------
# Sidebar Inputs
# -----------------------------------------------------
st.sidebar.header("🏠 House Details")

overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.sidebar.slider("Living Area (sqft)", 300, 5000, 1500)
garage_cars = st.sidebar.slider("Garage Cars", 0, 4, 2)
total_bsmt_sf = st.sidebar.slider("Basement Area (sqft)", 0, 3000, 800)
year_built = st.sidebar.slider("Year Built", 1870, 2010, 2000)

input_data = X.iloc[[0]].copy()
input_data["OverallQual"] = overall_qual
input_data["GrLivArea"] = gr_liv_area
input_data["GarageCars"] = garage_cars
input_data["TotalBsmtSF"] = total_bsmt_sf
input_data["YearBuilt"] = year_built

prediction = model.predict(input_data)[0]

# -----------------------------------------------------
# KPI Cards
# -----------------------------------------------------
st.markdown("### 📌 Key Metrics")

k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(f"""
    <div class="metric-container">
        <h3>💰 Estimated Price</h3>
        <h2>${prediction:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-container">
        <h3>📈 R² Score</h3>
        <h2>{r2:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-container">
        <h3>📉 RMSE</h3>
        <h2>${rmse:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------
# Visualizations
# -----------------------------------------------------
st.markdown("---")
st.markdown("## 📊 Market Insights")

c1, c2 = st.columns(2)

with c1:
    fig = px.scatter(
        data,
        x="GrLivArea",
        y="SalePrice",
        trendline="ols",
        title="Living Area vs Sale Price",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.box(
        data,
        x="OverallQual",
        y="SalePrice",
        title="Overall Quality vs Sale Price",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# Actual vs Predicted
# -----------------------------------------------------
st.markdown("---")
st.markdown("## 🎯 Model Performance")

perf_df = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

fig = px.scatter(
    perf_df,
    x="Actual Price",
    y="Predicted Price",
    trendline="ols",
    title="Actual vs Predicted Prices",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# Correlation Chart
# -----------------------------------------------------
st.markdown("---")
st.markdown("## 🔥 Top Correlated Features")

top_corr = data.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)[1:11]
corr_df = top_corr.reset_index()
corr_df.columns = ["Feature", "Correlation"]

fig = px.bar(
    corr_df,
    x="Correlation",
    y="Feature",
    orientation="h",
    title="Top 10 Correlated Features",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# Raw Data
# -----------------------------------------------------
with st.expander("📂 View Dataset"):
    st.dataframe(data)
