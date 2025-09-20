import streamlit as st
import pandas as pd
import joblib
import datetime
import os

# =================================================================================================
# Load Models and Preprocessing Objects
# =================================================================================================
st.cache_data.clear()
@st.cache_resource()
def load_models():
    """Load the pre-trained XGBoost models and LabelEncoders."""
    try:
        BASE_DIR = os.path.dirname(__file__)  # directory of app.py
        model_path1 = os.path.join(BASE_DIR, "xgb_best_proceed_to_mediation.pkl")
        model_path2 = os.path.join(BASE_DIR, "xgb_best_settled_undersampled.pkl")
        model_proceed = joblib.load(model_path1)
        model_settled = joblib.load(model_path2)
        le_intake = joblib.load("label_encoder_intake.pkl")
        le_dispute = joblib.load("label_encoder_dispute.pkl")
        return model_proceed, model_settled, le_intake, le_dispute
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Ensure all models and encoders are in the directory.")
        return None, None, None, None

model_proceed, model_settled, le_intake, le_dispute = load_models()


# =================================================================================================
# Preprocessing Function
# =================================================================================================

def preprocess_input(input_data):
    """Preprocess user input for prediction."""
    df = pd.DataFrame([input_data])

    # Convert date to features
    df['date_registered'] = pd.to_datetime(df['date_registered'])
    df['is_weekend'] = df['date_registered'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    # Encode categorical features
    try:
        df['type_of_intake'] = le_intake.transform(df['type_of_intake'])
        df['type_of_dispute'] = le_dispute.transform(df['type_of_dispute'])
    except ValueError as e:
        st.error(f"Input contains categories not seen during training: {e}")
        return None

    features = ['is_weekend', 'type_of_intake', 'type_of_dispute']
    return df[features]

# =================================================================================================
# Streamlit App
# =================================================================================================

def main():
    st.set_page_config(page_title="Mediation Case Predictor", layout="centered")
    st.title("Mediation Case Predictor")
    st.markdown("""
        Predict the likelihood of a case proceeding to mediation and being settled
        based on case attributes.
    """)
    st.markdown("---")

    if le_intake is None or le_dispute is None:
        st.warning("Encoders failed to load. Please check the .pkl files.")
        st.stop()

    # Get categories from encoders
    unique_intake_types = le_intake.classes_
    unique_dispute_types = le_dispute.classes_

    # User Input Form
    with st.form(key='prediction_form'):
        st.subheader("Case Details")
        date_registered = st.date_input("Date of Registration", datetime.date.today())
        type_of_intake = st.selectbox("Type of Intake", options=unique_intake_types)
        type_of_dispute = st.selectbox("Type of Dispute", options=unique_dispute_types)
        submit_button = st.form_submit_button(label='Predict Outcome')

    if submit_button and model_proceed and model_settled:
        user_input = {
            'date_registered': date_registered,
            'type_of_intake': type_of_intake,
            'type_of_dispute': type_of_dispute
        }

        input_df = preprocess_input(user_input)
        if input_df is None:
            return

        st.subheader("Prediction Results")

        # Predict probability 
        proceed_proba = model_proceed.predict_proba(input_df)[:, 1][0]

        if 0.40 <= proceed_proba <= 0.50:
            # Box 1: Mediation probability (borderline)
            settled_proba = model_settled.predict_proba(input_df)[:, 1][0]
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; border: 2px solid #FF9800; background-color: #FFF3E0; margin-bottom:10px;">
                <h4 style="color: #F57C00; margin: 0 0 10px 0;">⚠️ Borderline chance to proceed to mediation</h4>
                <p style="margin: 0; color: #E65100">
                    Probability of proceeding to mediation: <strong>{proceed_proba*100:.2f}%</strong>
                </p>
                <p style="margin: 5px 0 0 0; color: #E65100;">
                    Staff may consider following up, as the first model may underestimate attendance.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Box 2: Settlement probability
            settled_text = "✅ Likely to be settled" if settled_proba >= 0.5 else "❌ Unlikely to be settled"
            settled_border = "#4CAF50" if settled_proba >= 0.5 else "#F44336"
            settled_bg = "#E8F5E9" if settled_proba >= 0.5 else "#FFEBEE"
            settled_text_color = "#2E7D32" if settled_proba >= 0.5 else "#D32F2F"

            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; border: 2px solid {settled_border}; background-color: {settled_bg}; margin-top:10px;">
                <h4 style="color: {settled_text_color}; margin: 0 0 10px 0;">{settled_text}</h4>
                <p style="margin: 0; color: {settled_text_color}">
                    Probability of settlement if mediated: <strong>{settled_proba*100:.2f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)


        elif proceed_proba > 0.50:
            settled_proba = model_settled.predict_proba(input_df)[:, 1][0]
            # Box 1: Mediation probability
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; border: 2px solid #4CAF50; background-color: #E8F5E9;">
                    <h4 style="color: #2E7D32; margin: 0 0 10px 0;">✅ Case likely to proceed to mediation</h4>
                    <p style="margin: 0; color: #388E3C;">
                        Probability of proceeding to mediation: <strong>{proceed_proba * 100:.2f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True
            )

            # Box 2: Settlement probability
            if settled_proba >= 0.5:
                settled_text = "✅ Likely to be settled"
                settled_border = "#4CAF50"
                settled_bg = "#E8F5E9"
                settled_text_color = "#2E7D32"
            else:
                settled_text = "❌ Unlikely to be settled"
                settled_border = "#F44336"
                settled_bg = "#FFEBEE"
                settled_text_color = "#D32F2F"

            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; border: 2px solid {settled_border}; background-color: {settled_bg}; margin-top:10px;">
                    <h4 style="color: {settled_text_color}; margin: 0 0 10px 0;">{settled_text}</h4>
                    <p style="margin: 0; color: {settled_text_color};">
                        Probability of settlement: <strong>{settled_proba * 100:.2f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True
            )

        else:
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; border: 2px solid #F44336; background-color: #FFEBEE;">
                    <h4 style="color: #D32F2F; margin: 0 0 10px 0;">❌ Case unlikely to proceed to mediation</h4>
                    <p style="margin: 0; color: #B71C1C;">
                        Probability of proceeding to mediation: <strong>{proceed_proba * 100:.2f}%</strong>
                    </p>
                    <p style="margin: 5px 0 0 0; color: #B71C1C;">
                        If a case does not proceed to mediation, settlement is <strong>not applicable</strong>.
                    </p>
                </div>
                """, unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
