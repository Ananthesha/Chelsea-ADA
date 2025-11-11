import streamlit as st
import pandas as pd
import joblib
import os, glob

st.set_page_config(page_title="⚽ Player Fatigue & Performance Dashboard", layout="wide")

# ================================
# 🧠 AUTO DATA LOADER
# ================================
@st.cache_data
def load_data():
    # --- Locate all required files automatically ---
# --- Fixed, direct file paths (no searching) ---
    cleaned_path = "../data/final/cleaned_player_data.csv"

    preds_path = "../ML-delivarables/predictions.csv"
    shap_path = "../ML-delivarables/shap_values.csv"
    st.info(f"📂 Loaded dataset: {os.path.abspath(cleaned_path)}")



    if not cleaned_path:
        st.error("❌ cleaned_player_data.csv missing — please generate it using build_fatigue_index.py.")
        st.stop()

    # --- Load and parse date column safely ---
    df = pd.read_csv(cleaned_path)
    date_col = next((c for c in df.columns if c.lower() in ["match_date", "date", "game_date", "matchday"]), None)
    if date_col:
        df["match_date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        st.warning("⚠️ No explicit date column found; using index instead.")
        df["match_date"] = pd.to_datetime(df.index, errors="coerce")

    if "player_name" not in df.columns:
        df["player_name"] = df["player_id"]

    preds = pd.read_csv(preds_path) if preds_path else pd.DataFrame()
    shap_global = pd.read_csv(shap_path) if shap_path else pd.DataFrame()

    fi = df.groupby("player_id").tail(10)[["player_id", "player_name", "match_date", "fatigue_index"]] \
        if "fatigue_index" in df.columns else pd.DataFrame()

    return fi, preds, shap_global, df


# ================================
# ⚙️ LOAD DATA
# ================================
fi, preds, shap_global, df_full = load_data()

# Sidebar controls
st.sidebar.header("⚙️ Filters")
player_name = st.sidebar.selectbox("Select Player", sorted(df_full["player_name"].unique()))
date_range = st.sidebar.date_input(
    "Select Date Range",
    [df_full["match_date"].min(), df_full["match_date"].max()]
)
fatigue_thresh = st.sidebar.slider("Fatigue Threshold", 0.0, 1.0, 0.6)

# ================================
# 🧭 TABS
# ================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Fatigue", "🎯 Predictions", "🔍 SHAP Importance", "🏅 Top 10 Fatigued Players", "🤖 Live Prediction"]
)

# ================================
# 📈 TAB 1: Fatigue Trends
# ================================
with tab1:
    st.subheader("📈 Fatigue Index Over Time")
    if fi.empty:
        st.warning("No fatigue index data available in this dataset.")
    else:
        pdata = fi[(fi["player_name"] == player_name) &
                   (fi["match_date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))]
        if pdata.empty:
            st.warning("No data for selected player/date range.")
        else:
            st.line_chart(pdata.set_index("match_date")["fatigue_index"])
            st.caption(f"⚠️ {len(pdata[pdata['fatigue_index'] > fatigue_thresh])} matches exceeded FI threshold ({fatigue_thresh})")

# ================================
# 🎯 TAB 2: Model Predictions
# ================================
with tab2:
    st.subheader("🎯 Model Predictions (from ML Developer)")
    if preds.empty:
        st.info("No predictions.csv found.")
    else:
        st.dataframe(preds.head(20))

# ================================
# 🔍 TAB 3: SHAP Feature Importance
# ================================
with tab3:
    st.subheader("🔍 SHAP Feature Importance (Explainability)")

    if shap_global.empty:
        st.info("No SHAP data found.")
    else:
        shap_cols = [c for c in shap_global.columns if c.startswith("SHAP_")]
        if shap_cols:
            mean_abs = shap_global[shap_cols].abs().mean().sort_values(ascending=False)
            st.bar_chart(mean_abs.head(10))
            st.caption("🔹 Showing Top 10 features ranked by mean |SHAP| value.")
            top_feat = mean_abs.head(5)
            st.write("**Top Influential Features:**")
            for name, value in top_feat.items():
                st.write(f"• {name.replace('SHAP_', '')}: {value:.4f}")
        else:
            st.warning("⚠️ SHAP columns not recognized in uploaded CSV.")

# ================================
# 🏅 TAB 4: Top 10 Most Fatigued Players
# ================================
with tab4:
    st.subheader("🏅 Top 10 Most Fatigued Players")
    if "fatigue_index" not in df_full.columns:
        st.info("Fatigue Index not found in dataset.")
    else:
        top10 = (df_full.groupby("player_name")["fatigue_index"]
                 .mean()
                 .sort_values(ascending=False)
                 .head(10))
        st.bar_chart(top10)
        st.caption("🔹 Based on average fatigue index over available matches.")

# ================================
# 🤖 TAB 5: Live Prediction
# ================================
with tab5:
    st.subheader("🤖 Run Live Prediction")
    if st.checkbox("Run prediction using trained model"):
        try:
            matches = glob.glob("../**/trained_fatigue_model.pkl", recursive=True)
            if matches:
                model_path = matches[0]
                model = joblib.load(model_path)
                st.success(f"✅ Loaded trained model from: {model_path}")
            else:
                st.error("❌ trained_fatigue_model.pkl not found.")
                st.stop()

            # Required features for prediction
            required_features = [
                "sharpness_decay", "sprint_speed", "potential", "rating_rolling", "stamina",
                "days_since_last_match", "sprint_norm", "strength", "overall_rating",
                "reactions", "agility", "recovery_factor", "stamina_norm"
            ]

            missing = [f for f in required_features if f not in df_full.columns]

            if missing:
                st.error(f"❌ Missing required features in cleaned_player_data.csv: {missing}")
                st.info("➡️ Run build_fatigue_index.py again to regenerate full dataset.")
                st.stop()

            latest = df_full[df_full["player_name"] == player_name].sort_values("match_date").tail(1)
            if not latest.empty:
                pred = model.predict(latest[required_features].fillna(0))[0]
                st.success(f"🎯 Predicted next-match performance metric: **{pred:.2f}**")
            else:
                st.warning("No recent data available for this player.")

        except ModuleNotFoundError as e:
            st.error(f"⚠️ Missing dependency: {e}. Please install required libraries (e.g., `pip install xgboost`).")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Developed by Member 3 – Visualization & Integration Lead")
