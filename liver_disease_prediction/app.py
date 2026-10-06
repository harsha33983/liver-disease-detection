import streamlit as st
import pandas as pd
import joblib
import os

def main():
    st.set_page_config(page_title="Liver Disease Prediction System", layout="centered")

    st.title("Liver Disease Prediction System")
    st.subheader("Machine Learning Based Liver Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("1. Age", min_value=1, max_value=120, value=40)
        gender = st.selectbox("2. Gender", ["Male", "Female"])
        total_bilirubin = st.number_input("3. Total Bilirubin", min_value=0.0, value=1.0, format="%.2f")
        direct_bilirubin = st.number_input("4. Direct Bilirubin", min_value=0.0, value=0.1, format="%.2f")
        alkaline_phosphotase = st.number_input("5. Alkaline Phosphotase", min_value=0.0, value=100.0, format="%.2f")

    with col2:
        alamine_aminotransferase = st.number_input("6. Alamine Aminotransferase", min_value=0.0, value=50.0, format="%.2f")
        aspartate_aminotransferase = st.number_input("7. Aspartate Aminotransferase", min_value=0.0, value=50.0, format="%.2f")
        total_proteins = st.number_input("8. Total Proteins", min_value=0.0, value=6.0, format="%.2f")
        albumin = st.number_input("9. Albumin", min_value=0.0, value=3.0, format="%.2f")
        ag_ratio = st.number_input("10. Albumin and Globulin Ratio", min_value=0.0, value=1.0, format="%.2f")

    if st.button("Predict"):
        model_path = "models/liver_disease_model.joblib"
        if not os.path.exists(model_path):
            st.error("Trained model not found. Please run the training script first.")
            return

        try:
            model = joblib.load(model_path)

            gender_encoded = 1 if gender == "Male" else 0

            input_data = {
                'Age': [age],
                'Gender': [gender_encoded],
                'Total_Bilirubin': [total_bilirubin],
                'Direct_Bilirubin': [direct_bilirubin],
                'Alkaline_Phosphotase': [alkaline_phosphotase],
                'Alamine_Aminotransferase': [alamine_aminotransferase],
                'Aspartate_Aminotransferase': [aspartate_aminotransferase],
                'Total_Proteins': [total_proteins],
                'Albumin': [albumin],
                'Albumin_and_Globulin_Ratio': [ag_ratio]
            }

            df = pd.DataFrame(input_data)

            prediction = model.predict(df)[0]

            st.markdown("---")
            if prediction == 1:
                st.error("### Liver Disease Detected")
            else:
                st.success("### No Liver Disease Detected")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

   

if __name__ == "__main__":
    main()
