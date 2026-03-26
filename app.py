import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@st.cache_data(show_spinner=False)
def load_data(path: str = "heart.csv") -> pd.DataFrame:
    """Load the heart disease dataset."""
    df = pd.read_csv(path)

    # Replace zeros in Cholesterol and RestingBP columns with the column mean (as in the notebook)
    if "Cholesterol" in df.columns:
        cholesterol_mean = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean()
        df["Cholesterol"] = df["Cholesterol"].replace(0, cholesterol_mean).round(2)

    if "RestingBP" in df.columns:
        restingbp_mean = df.loc[df["RestingBP"] != 0, "RestingBP"].mean()
        df["RestingBP"] = df["RestingBP"].replace(0, restingbp_mean).round(2)

    return df


@st.cache_data(show_spinner=False)
def train_model(df: pd.DataFrame) -> tuple[LogisticRegression, StandardScaler, list]:
    """Train a logistic regression model and return the trained model along with scaler and expected columns."""
    df_encoded = pd.get_dummies(df, drop_first=False)
    df_encoded = df_encoded.astype(int)

    # Standardize numeric columns the same way as the notebook
    scaler = StandardScaler()
    numeric_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR"]
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    X = df_encoded.drop(columns=["HeartDisease"])
    y = df_encoded["HeartDisease"]

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    return model, scaler, X.columns.tolist()


def build_input_dataframe(user_input: dict, feature_columns: list, scaler: StandardScaler) -> pd.DataFrame:
    """Convert the user input into the same encoded feature vector used by the model."""
    # Start with a single-row dataframe of the raw input
    row = pd.DataFrame([user_input])

    # Apply the same zero-replacement logic for robustness
    if "Cholesterol" in row.columns:
        row["Cholesterol"] = row["Cholesterol"].replace(0, np.nan)
        if row["Cholesterol"].isna().any():
            row["Cholesterol"] = row["Cholesterol"].fillna(user_input.get("Cholesterol", 0))
    if "RestingBP" in row.columns:
        row["RestingBP"] = row["RestingBP"].replace(0, np.nan)
        if row["RestingBP"].isna().any():
            row["RestingBP"] = row["RestingBP"].fillna(user_input.get("RestingBP", 0))

    # One-hot encode categorical fields using get_dummies
    row_encoded = pd.get_dummies(row, drop_first=False)

    # Add missing columns that the trained model expects (fill missing with 0)
    for c in feature_columns:
        if c not in row_encoded.columns:
            row_encoded[c] = 0

    # Keep only the columns in the same order
    row_encoded = row_encoded[feature_columns]

    # Scale the numeric inputs using the same scaler
    numeric_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR"]
    row_encoded[numeric_cols] = scaler.transform(row_encoded[numeric_cols])

    return row_encoded


def main():
    st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

    st.title("Heart Disease Risk Predictor")
    st.markdown(
        "Use this simple app to estimate the likelihood of heart disease based on common clinical measurements and symptoms."
    )

    df = load_data("heart.csv")
    model, scaler, feature_columns = train_model(df)

    st.subheader("Enter your details")
    st.markdown("Provide your measurements and symptoms below, then tap **Check for Heart Disease**.")

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", min_value=18, max_value=100, value=int(df["Age"].median()))
            sex = st.selectbox("Sex", options=["M", "F"], index=0)
            chest_pain = st.selectbox(
                "Chest Pain Type",
                options=["ATA", "NAP", "TA", "ASY"],
                index=0,
                help="ATA = Typical Angina, NAP = Non-anginal Pain, TA = Atypical Angina, ASY = Asymptomatic",
            )
            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=50,
                max_value=250,
                value=int(df["RestingBP"].median()),
            )
            cholesterol = st.number_input(
                "Cholesterol (mg/dL)",
                min_value=100,
                max_value=600,
                value=int(df["Cholesterol"].median()),
            )
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], index=0)
            rest_ecg = st.selectbox(
                "Resting ECG",
                options=["Normal", "ST", "LVH"],
                index=0,
                help="ST = ST-T wave abnormality, LVH = Left ventricular hypertrophy",
            )

        with col2:
            max_hr = st.number_input(
                "Max Heart Rate Achieved",
                min_value=60,
                max_value=250,
                value=int(df["MaxHR"].median()),
            )
            exercise_angina = st.selectbox("Exercise Induced Angina", options=["Y", "N"], index=1)
            oldpeak = st.number_input(
                "Oldpeak (ST depression induced by exercise)",
                min_value=0.0,
                max_value=10.0,
                value=float(df["Oldpeak"].median()),
                step=0.1,
            )
            slope = st.selectbox("Slope of ST segment", options=["Up", "Flat", "Down"], index=0)
            ca = st.selectbox("Number of major vessels (0-3) colored by fluoroscopy", options=[0, 1, 2, 3], index=0)
            thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"], index=0)

        submitted = st.form_submit_button("Check for Heart Disease")

    if submitted:
        user_input = {
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": rest_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": slope,
            "Ca": ca,
            "Thal": thal,
        }

        row = build_input_dataframe(user_input, feature_columns, scaler)
        prediction = model.predict(row)[0]
        probability = model.predict_proba(row)[0][1]

        if prediction == 1:
            st.error("⚠️ The model predicts a high likelihood of heart disease.")
        else:
            st.success("✅ The model predicts a low likelihood of heart disease.")

        st.markdown(f"**Predicted probability of heart disease:** {probability:.2%}")

    st.markdown("---")
    with st.expander("View raw dataset (for reference)"):
        st.dataframe(df.head(10))


if __name__ == "__main__":
    main()
