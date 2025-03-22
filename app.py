import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from prophet import Prophet
import io

st.set_page_config(page_title="AI Sprzedaż - Dashboard", layout="wide")

st.title("📊 AI Dashboard Sprzedaży e-Commerce")

st.markdown("""
Wgraj dane sprzedaży i reklam, a my:
- przeanalizujemy Twoje wyniki,
- wygenerujemy prognozę sprzedaży na 30 dni,
- podpowiemy, co możesz poprawić.
""")

# 1. Upload plików
sales_file = st.file_uploader("📄 Wgraj plik CSV ze sprzedażą", type=["csv"], key="sales")
ad_file = st.file_uploader("📄 Wgraj plik CSV z danymi reklamowymi", type=["csv"], key="ads")

if sales_file and ad_file:
    # 2. Wczytaj dane
    sales_df = pd.read_csv(sales_file)
    ad_df = pd.read_csv(ad_file)

    sales_df["Date"] = pd.to_datetime(sales_df["Date"])
    ad_df["Date"] = pd.to_datetime(ad_df["Date"])

    st.success("Pliki zostały poprawnie wczytane!")

    # 3. Analiza sprzedaży
    st.subheader("🧾 Łączna sprzedaż według źródła")
    total_sales = sales_df.groupby("Source")["Total_Value"].sum().sort_values(ascending=False)
    st.bar_chart(total_sales)

    # 4. Analiza ROI
    st.subheader("📈 ROI według platformy reklamowej")
    roi_df = ad_df.groupby("Platform").agg({"Cost": "sum", "Conversions": "sum"}).reset_index()
    roi_df["ROI"] = roi_df["Conversions"] / roi_df["Cost"]
    st.dataframe(roi_df)
    st.bar_chart(roi_df.set_index("Platform")["ROI"])

    # 5. Prognoza sprzedaży z Prophet
    st.subheader("🔮 Prognoza sprzedaży na 30 dni")
    daily_sales = sales_df.groupby("Date")["Total_Value"].sum().reset_index()
    daily_sales.rename(columns={"Date": "ds", "Total_Value": "y"}, inplace=True)

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(daily_sales)

    # Bierzemy średnie z ostatnich 7 dni (lub mniej jeśli danych mało)
    recent_days = ad_features.sort_values("ds").tail(7)
    averages = recent_days[["Cost", "CTR", "Conversions", "Impressions"]].mean()

    # Tworzymy DataFrame z brakującymi datami
    missing = future[~future["ds"].isin(ad_features["ds"])]
    for col in ["Cost", "CTR", "Conversions", "Impressions"]:
        missing[col] = averages[col]

    # Łączymy aktualne + brakujące
    future = pd.concat([
        future[future["ds"].isin(ad_features["ds"])].merge(ad_features, on="ds", how="left"),missing
    ], ignore_index=True).sort_values("ds")

    forecast = model.predict(future)
    forecast_plot = model.plot(forecast)
    st.pyplot(forecast_plot)

    # 6. Rekomendacje
    st.subheader("💡 Rekomendacje działań")
    top_roi = roi_df.sort_values("ROI", ascending=False).iloc[0]
    low_roi = roi_df.sort_values("ROI", ascending=True).iloc[0]
    if forecast["yhat"].iloc[-1] < forecast["yhat"].iloc[0]:
        st.warning("Prognoza wskazuje na spadek sprzedaży – warto wzmocnić kampanię marketingową.")

    st.info(f"Zwiększ budżet na {top_roi['Platform']}, ROI = {top_roi['ROI']:.2f}")
    st.info(f"Ogranicz wydatki na {low_roi['Platform']}, ROI = {low_roi['ROI']:.2f}")

    # 7. Mapa ciepła sprzedaży
    st.subheader("🔥 Mapa ciepła sprzedaży")
    sales_df["Weekday"] = sales_df["Date"].dt.weekday
    sales_df["Hour"] = np.random.randint(8, 22, len(sales_df))
    heatmap_data = sales_df.pivot_table(values="Total_Value", index="Hour", columns="Weekday", aggfunc="sum")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".0f", ax=ax)
    st.pyplot(fig)
else:
    st.info("Proszę wgrać dwa pliki CSV, aby rozpocząć analizę.")
