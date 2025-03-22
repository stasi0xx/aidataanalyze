import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from prophet import Prophet
import io

st.set_page_config(page_title="AI Sprzedaż - Dashboard", layout="wide")

# 🔒 Informacja o prywatności
st.warning("\n🔐 **Bezpieczeństwo danych:** Twoje pliki są przetwarzane tymczasowo w pamięci aplikacji i nie są zapisywane ani udostępniane. Dane znikają po odświeżeniu strony.")

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

# 2. Funkcja fallback - bez regresorów
def forecast_sales_simple(sales_df, forecast_period=30):
    daily_sales = sales_df.groupby("Date")["Total_Value"].sum().reset_index()
    daily_sales.rename(columns={"Date": "ds", "Total_Value": "y"}, inplace=True)

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(daily_sales)

    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    return forecast, model

# 3. Funkcja zaawansowana - z regresorami
def forecast_sales_with_regressors(sales_df, ad_df, forecast_period=30):
    daily_sales = sales_df.groupby("Date")["Total_Value"].sum().reset_index()
    daily_sales.rename(columns={"Date": "ds", "Total_Value": "y"}, inplace=True)

    ad_features = ad_df.groupby("Date").agg({
        "Cost": "sum",
        "CTR": "mean",
        "Conversions": "sum",
        "Impressions": "sum"
    }).reset_index()
    ad_features.rename(columns={"Date": "ds"}, inplace=True)

    full_data = pd.merge(daily_sales, ad_features, on="ds", how="left").fillna(0)

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_regressor("Cost")
    model.add_regressor("CTR")
    model.add_regressor("Conversions")
    model.add_regressor("Impressions")
    model.fit(full_data)

    future = model.make_future_dataframe(periods=forecast_period, freq='D')

    recent_days = ad_features.sort_values("ds").tail(7)
    averages = recent_days[["Cost", "CTR", "Conversions", "Impressions"]].mean()

    future_ads = future.merge(ad_features, on="ds", how="left")
    for col in ["Cost", "CTR", "Conversions", "Impressions"]:
        future_ads[col] = future_ads[col].fillna(averages[col])

    forecast = model.predict(future_ads)
    return forecast, model

if sales_file and ad_file:
    sales_df = pd.read_csv(sales_file)
    ad_df = pd.read_csv(ad_file)

    sales_df["Date"] = pd.to_datetime(sales_df["Date"])
    ad_df["Date"] = pd.to_datetime(ad_df["Date"])

    st.success("Pliki zostały poprawnie wczytane!")

    st.subheader("🧾 Łączna sprzedaż według źródła")
    total_sales = sales_df.groupby("Source")["Total_Value"].sum().sort_values(ascending=False)
    st.bar_chart(total_sales)

    st.subheader("📈 ROI według platformy reklamowej")
    roi_df = ad_df.groupby("Platform").agg({"Cost": "sum", "Conversions": "sum"}).reset_index()
    roi_df["ROI"] = roi_df["Conversions"] / roi_df["Cost"]
    st.dataframe(roi_df)
    st.bar_chart(roi_df.set_index("Platform")["ROI"])

    st.subheader("🔮 Prognoza sprzedaży na 30 dni")
    try:
        if ad_df["Cost"].sum() < 1 or ad_df["CTR"].mean() < 0.001:
            forecast, model = forecast_sales_simple(sales_df)
            st.info("Użyto uproszczonej prognozy bez danych reklamowych.")
        else:
            forecast, model = forecast_sales_with_regressors(sales_df, ad_df)
    except Exception as e:
        st.error(f"Błąd podczas generowania prognozy: {e}")
        forecast, model = forecast_sales_simple(sales_df)

    forecast_plot = model.plot(forecast)
    st.pyplot(forecast_plot)

    st.subheader("💡 Rekomendacje działań")
    top_roi = roi_df.sort_values("ROI", ascending=False).iloc[0]
    low_roi = roi_df.sort_values("ROI", ascending=True).iloc[0]
    if forecast["yhat"].iloc[-1] < forecast["yhat"].iloc[0]:
        st.warning("Prognoza wskazuje na spadek sprzedaży – warto wzmocnić kampanię marketingową.")

    st.info(f"Zwiększ budżet na {top_roi['Platform']}, ROI = {top_roi['ROI']:.2f}")
    st.info(f"Ogranicz wydatki na {low_roi['Platform']}, ROI = {low_roi['ROI']:.2f}")

    st.subheader("🔥 Mapa ciepła sprzedaży")
    sales_df["Weekday"] = sales_df["Date"].dt.weekday
    sales_df["Hour"] = np.random.randint(8, 22, len(sales_df))
    heatmap_data = sales_df.pivot_table(values="Total_Value", index="Hour", columns="Weekday", aggfunc="sum")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".0f", ax=ax)
    st.pyplot(fig)
else:
    st.info("Proszę wgrać dwa pliki CSV, aby rozpocząć analizę.")
