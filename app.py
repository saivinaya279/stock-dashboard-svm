import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# ---------------- Page Config ----------------
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("📈 Stock Dashboard")

# ---------------- Load Companies ----------------
DATA_FOLDER = "data"

companies = sorted([
    f.replace(".csv", "")
    for f in os.listdir(DATA_FOLDER)
    if f.endswith(".csv")
])

selected_company = st.selectbox("Select Company", companies)

# ---------------- Load CSV ----------------
file_path = os.path.join(DATA_FOLDER, f"{selected_company}.csv")
df = pd.read_csv(file_path)

# Force first column to be Date
df.rename(columns={df.columns[0]: "Date"}, inplace=True)

# Convert Date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# ---------------- Ensure Numeric Columns ----------------
price_cols = ["Open", "High", "Low", "Close", "Volume"]
for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# ---------------- Date Features ----------------
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month_name()

# ---------------- Sidebar Filters ----------------
st.sidebar.header("📅 Filters")

selected_years = st.sidebar.multiselect(
    "Select Year",
    options=sorted(df["Year"].unique()),
    default=sorted(df["Year"].unique())
)

selected_months = st.sidebar.multiselect(
    "Select Month",
    options=df["Month"].unique(),
    default=df["Month"].unique()
)

filtered_df = df[
    (df["Year"].isin(selected_years)) &
    (df["Month"].isin(selected_months))
]

if filtered_df.empty:
    st.warning("No data available for selected filters")
    st.stop()

# ---------------- KPI Cards ----------------
latest = filtered_df.iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Open", f"${latest['Open']:.2f}")
col2.metric("High", f"${latest['High']:.2f}")
col3.metric("Low", f"${latest['Low']:.2f}")
col4.metric("Close", f"${latest['Close']:.2f}")

# ---------------- ML: SVM Prediction ----------------
st.subheader("🤖 ML Prediction (SVM)")

# Target: 1 = UP, 0 = DOWN
df_ml = df.copy()
df_ml["Target"] = (df_ml["Close"].shift(-1) > df_ml["Close"]).astype(int)
df_ml = df_ml.dropna()

features = ["Open", "High", "Low", "Close", "Volume"]
X = df_ml[features]
y = df_ml["Target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

svm_model = SVC(kernel="rbf", probability=True)
svm_model.fit(X_train, y_train)

latest_features = scaler.transform(df_ml[features].iloc[[-1]])
prediction = svm_model.predict(latest_features)[0]
confidence = svm_model.predict_proba(latest_features).max()

if prediction == 1:
    st.success(f"📈 Trend: UP  | Confidence: {confidence*100:.2f}%")
else:
    st.error(f"📉 Trend: DOWN | Confidence: {confidence*100:.2f}%")
# ---------------- Save Model ----------------
os.makedirs("models", exist_ok=True)

joblib.dump(svm_model, f"models/{selected_company}_svm_model.pkl")
joblib.dump(scaler, f"models/{selected_company}_scaler.pkl")

# ---------------- Model Accuracy ----------------
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.info(f"📊 SVM Model Accuracy: {accuracy * 100:.2f}%")


# ---------------- Candlestick Chart ----------------
st.subheader("📈 Historical Candlestick Chart")

candlestick_fig = go.Figure(
    data=[
        go.Candlestick(
            x=filtered_df["Date"],
            open=filtered_df["Open"],
            high=filtered_df["High"],
            low=filtered_df["Low"],
            close=filtered_df["Close"],
            name=selected_company
        )
    ]
)

candlestick_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    height=500
)

st.plotly_chart(candlestick_fig, use_container_width=True)

# ---------------- Volume Chart ----------------
st.subheader("📊 Trading Volume")

volume_fig = go.Figure()
volume_fig.add_trace(
    go.Bar(
        x=filtered_df["Date"],
        y=filtered_df["Volume"],
        name="Volume"
    )
)

volume_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Volume",
    height=300
)

st.plotly_chart(volume_fig, use_container_width=True)

# ---------------- Data Preview ----------------
st.subheader(f"📊 Data Preview — {selected_company}")
st.dataframe(filtered_df.tail(), use_container_width=True)

# ---------------- Dataset Info ----------------
with st.expander("ℹ️ Dataset Info"):
    st.write("Columns:", list(filtered_df.columns))
    st.write("Rows:", len(filtered_df))
    st.write(
        "Date range:",
        filtered_df["Date"].min().date(),
        "→",
        filtered_df["Date"].max().date()
    )
