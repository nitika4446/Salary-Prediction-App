import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="360 Financial EDA Engine",
    page_icon="📈",
    layout="wide"
)

st.title("📈 360° Financial Exploratory Data Analysis Engine")
st.markdown("""
Upload your stock dataset and get:

✅ Technical Analysis  
✅ Risk Analysis  
✅ Statistical EDA  
✅ AI Stock Score  
✅ Candlestick Visualization  
""")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Dataset uploaded successfully")

        # ---------------------------
        # Validate Required Columns
        # ---------------------------
        required_cols = ["Date", "Open", "High", "Low", "Close"]

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            st.stop()

        # ---------------------------
        # Data Cleaning
        # ---------------------------
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df.set_index("Date", inplace=True)

        numeric_cols = ["Open", "High", "Low", "Close"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(inplace=True)

        # ---------------------------
        # Raw Data
        # ---------------------------
        if st.checkbox("Show Raw Data"):
            st.dataframe(df.tail())

        # ---------------------------
        # Technical Indicators
        # ---------------------------
        st.header("Technical Analysis")

        close = df["Close"]

        df["SMA_20"] = close.rolling(20).mean()
        df["EMA_20"] = close.ewm(span=20).mean()

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        rolling_std = close.rolling(20).std()

        df["Upper_Band"] = df["SMA_20"] + (rolling_std * 2)
        df["Lower_Band"] = df["SMA_20"] - (rolling_std * 2)

        df["EMA12"] = close.ewm(span=12).mean()
        df["EMA26"] = close.ewm(span=26).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]

        st.line_chart(df[["Close", "SMA_20", "EMA_20"]])
        st.line_chart(df[["RSI"]])
        st.line_chart(df[["MACD"]])

        # ---------------------------
        # Risk Analysis
        # ---------------------------
        st.header("Risk Analysis")

        df["Returns"] = close.pct_change()

        volatility = df["Returns"].std() * np.sqrt(252)

        sharpe_ratio = 0
        if df["Returns"].std() != 0:
            sharpe_ratio = (
                df["Returns"].mean() / df["Returns"].std()
            ) * np.sqrt(252)

        rolling_max = close.cummax()
        drawdown = (close - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        c1, c2, c3 = st.columns(3)

        c1.metric("Volatility", round(volatility, 2))
        c2.metric("Sharpe Ratio", round(sharpe_ratio, 2))
        c3.metric("Max Drawdown", round(max_drawdown, 2))

        # ---------------------------
        # Statistical EDA
        # ---------------------------
        st.header("Statistical EDA")

        st.dataframe(df.describe())

        numeric_df = df.select_dtypes(include=np.number)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )
        st.pyplot(fig)

        # ---------------------------
        # AI Stock Score
        # ---------------------------
        st.header("AI Stock Score")

        score = 0

        if volatility < 0.3:
            score += 25

        if sharpe_ratio > 1:
            score += 25

        latest_rsi = df["RSI"].dropna()

        if not latest_rsi.empty:
            if latest_rsi.iloc[-1] < 70:
                score += 25

        if max_drawdown > -0.2:
            score += 25

        st.subheader(f"Final Score: {score}/100")

        if score >= 75:
            st.success("Strong Buy Candidate")
        elif score >= 50:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk Stock")

        # ---------------------------
        # Candlestick Chart
        # ---------------------------
        st.header("Candlestick Chart")

        fig = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"]
            )
        ])

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------
        # Download Report
        # ---------------------------
        csv = df.to_csv().encode("utf-8")

        st.download_button(
            "Download Report",
            csv,
            "financial_report.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("Upload a CSV/Excel file to begin")