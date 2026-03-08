import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from codecarbon import EmissionsTracker
import time

# ------------------------------
# App Title
# ------------------------------
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Responsible AI Evaluation Toolkit (RAI-Toolkit)</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #566573;'>Transparency, Fairness & Energy Efficiency Analyzer</h4>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------
# 📂 Upload Dataset
# ------------------------------
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Dataset:")
    st.dataframe(data.head())

    # ------------------------------
    # Target Selection
    # ------------------------------
    target_column = st.selectbox("Select Target Column", data.columns)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Convert categorical features to dummy variables
    X = pd.get_dummies(X)

    # ------------------------------
    # Detect target type
    # ------------------------------
    if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 10:
        target_type = "regression"
        st.info("🔹 Continuous numeric target detected. Using RandomForestRegressor.")
    else:
        target_type = "classification"
        st.info("🔹 Categorical target detected. Using RandomForestClassifier.")

    # ------------------------------
    # Train-Test Split
    # ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------
    # ⚡ Train Model with Progress
    # ------------------------------
    st.subheader("⚡ Train Model")
    st.write("Training model...")
    progress_bar = st.progress(0)
    for i in range(101):
        time.sleep(0.01)
        progress_bar.progress(i)

    tracker = EmissionsTracker()
    tracker.start()

    # ------------------------------
    # Train classifier or regressor
    # ------------------------------
    if target_type == "classification":
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
    else:
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)  # regression metric

    emissions = tracker.stop()
    st.success("✅ Model Trained Successfully!")

    # ------------------------------
    # 📊 Model Metrics
    # ------------------------------
    st.subheader("📊 Model Metrics")
    col1, col2 = st.columns(2)
    with col1:
        metric_label = "Model Accuracy" if target_type=="classification" else "R² Score"
        st.metric(metric_label, f"{round(accuracy*100,2)} %")
    with col2:
        st.metric("CO₂ Emission (kg)", f"{round(emissions,6)}")

    efficiency = accuracy / (emissions + 1e-6)
    st.markdown(f"<h3 style='color: #28B463;'>Efficiency Score: {round(efficiency,2)}</h3>", unsafe_allow_html=True)

    # ------------------------------
    # 🔍 Feature Insights
    # ------------------------------
    st.subheader("🔍 Feature Insights")
    importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    st.bar_chart(importance.set_index("Feature"))
    st.write("📌 Shows which features influence AI decisions the most.")

    # ------------------------------
    # 🏆 Responsible AI Score
    # ------------------------------
    st.subheader("🏆 Responsible AI Score")
    score = (accuracy*80) + (20*(1/(1+emissions)))
    st.markdown(f"<h2 style='color: #D68910;'>Score: {round(score,2)} / 100</h2>", unsafe_allow_html=True)
    st.write("📌 Combines performance, transparency, and energy efficiency.")

    # ------------------------------
    # 💡 Suggestions / Recommendations
    # ------------------------------
    st.subheader("💡 Suggestions / Recommendations")
    if target_type=="classification":
        if accuracy < 0.8:
            st.warning("🔹 Accuracy is low. Consider adding more data, cleaning data, or feature engineering.")
        elif efficiency < 1000:
            st.info("🔹 Efficiency is low. Try reducing model complexity or optimizing hyperparameters.")
        elif score < 70:
            st.error("🔹 Responsible AI Score is low. Check fairness, transparency, and energy usage.")
        else:
            st.success("✅ Model performance is good! You can deploy the model or explore advanced features.")
    else:  # regression suggestions
        if accuracy < 0.5:
            st.warning("🔹 R² Score is low. Consider improving features or adding more data.")
        elif efficiency < 1000:
            st.info("🔹 Efficiency is low. Try reducing model complexity or optimizing hyperparameters.")
        elif score < 70:
            st.error("🔹 Responsible AI Score is low. Check energy usage and model transparency.")
        else:
            st.success("✅ Model prediction is good! You can deploy the model or explore advanced features.")