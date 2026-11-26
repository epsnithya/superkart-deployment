import os
import requests
import pandas as pd
import streamlit as st

# Backend URL. Inside the Docker network the Flask container is reachable by its name.
BACKEND_URL = os.getenv("BACKEND_URL", "http://superkart-backend:7860")

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="centered")
st.title("🛒 SuperKart Sales Forecasting")
st.write("Predict the total sales revenue of a product in a store.")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

# ---------------- Online prediction ----------------
with tab_single:
    st.subheader("Product details")
    c1, c2 = st.columns(2)
    with c1:
        product_weight = st.number_input("Product Weight", 4.0, 22.0, 12.66, step=0.1)
        sugar = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        area = st.number_input("Allocated Display Area (ratio)", 0.004, 0.300, 0.056,
                               step=0.001, format="%.3f")
        mrp = st.number_input("Product MRP", 30.0, 270.0, 146.74, step=0.5)
        id_char = st.selectbox("Product Id Prefix", ["FD", "DR", "NC"])
        ptype = st.selectbox("Product Category", ["Perishables", "Non Perishables"])
    with c2:
        st.subheader("Store details")
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Departmental Store", "Food Mart",
                                                 "Supermarket Type1", "Supermarket Type2"])
        store_age = st.number_input("Store Age (years)", 0, 60, 16, step=1)

    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": area,
        "Product_MRP": mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": city,
        "Store_Type": store_type,
        "Product_Id_char": id_char,
        "Store_Age_Years": store_age,
        "Product_Type_Category": ptype,
    }

    if st.button("Predict Sales", type="primary"):
        try:
            r = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            if r.status_code == 200:
                st.success(f"Predicted Sales: **{r.json()['Predicted_Sales']:,.2f}**")
            else:
                st.error(f"API error {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Could not reach the backend: {e}")

# ---------------- Batch prediction ----------------
with tab_batch:
    st.subheader("Upload a CSV file")
    st.caption("Required columns: " + ", ".join(payload.keys()))
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.dataframe(batch_df.head())
        if st.button("Predict for all rows"):
            try:
                r = requests.post(f"{BACKEND_URL}/v1/predictbatch",
                                  files={"file": batch_df.to_csv(index=False).encode("utf-8")},
                                  timeout=120)
                if r.status_code == 200:
                    preds = pd.Series(r.json()).rename("Predicted_Sales")
                    preds.index = preds.index.astype(int)
                    result = batch_df.join(preds.sort_index())
                    st.dataframe(result)
                    st.download_button("Download predictions", result.to_csv(index=False),
                                       "superkart_predictions.csv", "text/csv")
                else:
                    st.error(f"API error {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Could not reach the backend: {e}")
