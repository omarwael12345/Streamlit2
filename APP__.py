import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import io
import os

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("Mall Customer Segmentation")
st.caption("K-Means clustering on customer age, income, spending score and gender")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload Mall_Customers.csv", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.info("Upload Mall_Customers.csv to begin.")
    st.stop()

required_cols = {"CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing expected columns: {sorted(missing)}")
    st.stop()

st.subheader("Raw data preview")
st.dataframe(df.head(), use_container_width=True)

st.subheader("Missing values")
st.dataframe(df.isnull().sum().rename("nulls"), use_container_width=True)

numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

st.subheader("Outlier check")
fig0, axes0 = plt.subplots(1, 3, figsize=(12, 4))
for ax, col in zip(axes0, numeric_cols):
    ax.boxplot(df[col])
    ax.set_title(col)
st.pyplot(fig0)

df = df.copy()
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

feature_cols = numeric_cols + ["Gender"]

scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
pretrained_scaler = None
if os.path.exists(scaler_path):
    pretrained_scaler = joblib.load(scaler_path)

with st.sidebar:
    st.header("Clustering settings")
    if pretrained_scaler is not None:
        st.caption("Using pre-fit scaler.pkl")
        use_scaling = True
    else:
        use_scaling = st.checkbox("Scale features", value=False)
    k = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=5)
    show_elbow = st.checkbox("Show elbow curve", value=True)

if pretrained_scaler is not None:
    scaler = pretrained_scaler
    X = df.copy()
    X[numeric_cols] = scaler.transform(df[numeric_cols])
    X = X[feature_cols]
elif use_scaling:
    scaler = StandardScaler()
    X = df.copy()
    X[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    X = X[feature_cols]
else:
    scaler = None
    X = df[feature_cols]

if show_elbow:
    st.subheader("Elbow method")
    inertias = []
    k_range = range(1, 11)
    for i in k_range:
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(k_range), inertias, marker="o")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(list(k_range))
    ax.grid(True)
    st.pyplot(fig)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

col1, col2 = st.columns(2)
with col1:
    st.metric("Customers", len(df))
with col2:
    st.metric("Clusters", k)

st.subheader("Cluster sizes")
st.bar_chart(df["Cluster"].value_counts().sort_index())

st.subheader("Average feature values per cluster")
st.dataframe(df.groupby("Cluster")[feature_cols].mean().round(2), use_container_width=True)

st.subheader("Income vs spending score")
fig2, ax2 = plt.subplots(figsize=(8, 6))
scatter = ax2.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
)
ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score (1-100)")
fig2.colorbar(scatter, label="Cluster")
st.pyplot(fig2)

st.subheader("Customers by cluster")
selected_cluster = st.selectbox("Filter by cluster", sorted(df["Cluster"].unique()))
st.dataframe(df[df["Cluster"] == selected_cluster], use_container_width=True)

st.subheader("Download results")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download clustered data as CSV", csv, "Mall_Customers_with_clusters.csv", "text/csv")

excel_buffer = io.BytesIO()
df.to_excel(excel_buffer, index=False, engine="openpyxl")
st.download_button(
    "Download clustered data as Excel",
    excel_buffer.getvalue(),
    "Mall_Customers_with_clusters.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

if scaler is not None and pretrained_scaler is None:
    scaler_buffer = io.BytesIO()
    joblib.dump(scaler, scaler_buffer)
    st.download_button("Download fitted scaler (.pkl)", scaler_buffer.getvalue(), "scaler.pkl", "application/octet-stream")
