import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the pre-trained model and pipeline
try:
    model_pipeline = joblib.load('Dragon.joblib')
except FileNotFoundError:
    st.error("Error: 'Dragon.joblib' not found. Please ensure the model file is in the same directory.")
    st.stop()

# Define the pipeline (from Real Estates.ipynb)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

try:
    housing_raw = pd.read_csv('data.csv')
    housing_features = housing_raw.drop("MEDV", axis=1).copy()
    my_pipeline.fit(housing_features)

except FileNotFoundError:
    st.error("Error: 'data.csv' not found. Cannot fit the preprocessing pipeline.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading or processing data for pipeline fitting: {e}")
    st.stop()


st.set_page_config(layout="wide")

st.title("🏡 Dragon Real Estate Price Predictor")
st.markdown("Enter the details of the property to get an estimated median value.")

feature_info = {
    "CRIM": "Per capita crime rate by town",
    "ZN": "Proportion of residential land zoned for lots over 25,000 sq.ft.",
    "INDUS": "Proportion of non-retail business acres per town",
    "CHAS": "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)",
    "NOX": "Nitric oxides concentration (parts per 10 million)",
    "RM": "Average number of rooms per dwelling",
    "AGE": "Proportion of owner-occupied units built prior to 1940",
    "DIS": "Weighted distances to five Boston employment centres",
    "RAD": "Index of accessibility to radial highways",
    "TAX": "Full-value property-tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B": "1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town",
    "LSTAT": "% lower status of the population"
}

input_data = {}
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("General Information")
    input_data["CRIM"] = st.number_input(f"CRIM ({feature_info['CRIM']})", value=0.00632, format="%.5f")
    input_data["ZN"] = st.number_input(f"ZN ({feature_info['ZN']})", value=18.0, format="%.1f")
    input_data["INDUS"] = st.number_input(f"INDUS ({feature_info['INDUS']})", value=2.31, format="%.2f")
    input_data["CHAS"] = st.selectbox(f"CHAS ({feature_info['CHAS']})", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

with col2:
    st.subheader("Environmental & Social Factors")
    input_data["NOX"] = st.number_input(f"NOX ({feature_info['NOX']})", value=0.538, format="%.3f")
    input_data["RM"] = st.number_input(f"RM ({feature_info['RM']})", value=6.575, format="%.3f")
    input_data["AGE"] = st.number_input(f"AGE ({feature_info['AGE']})", value=65.2, format="%.1f")
    input_data["DIS"] = st.number_input(f"DIS ({feature_info['DIS']})", value=4.0900, format="%.4f")

with col3:
    st.subheader("Accessibility & Demographics")
    input_data["RAD"] = st.number_input(f"RAD ({feature_info['RAD']})", value=1, format="%d")
    input_data["TAX"] = st.number_input(f"TAX ({feature_info['TAX']})", value=296, format="%d")
    input_data["PTRATIO"] = st.number_input(f"PTRATIO ({feature_info['PTRATIO']})", value=15.3, format="%.1f")
    input_data["B"] = st.number_input(f"B ({feature_info['B']})", value=396.90, format="%.2f")
    input_data["LSTAT"] = st.number_input(f"LSTAT ({feature_info['LSTAT']})", value=4.98, format="%.2f")

input_df = pd.DataFrame([input_data])
input_df["TAXRM"] = input_df['TAX'] / input_df['RM']

ordered_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
input_df = input_df[ordered_features]

if st.button("Predict Price"):
    try:
        prepared_input = my_pipeline.transform(input_df)
        prediction = model_pipeline.predict(prepared_input)[0]
        st.success(f"The estimated median value of the home is: ${prediction*1000:.2f}")
        st.info("Note: The prediction is in $1000's as per the dataset's MEDV attribute.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input fields are filled correctly.")

st.markdown("---")
st.markdown("### About the Features:")
for feature, desc in feature_info.items():
    st.markdown(f"**{feature}**: {desc}")

st.markdown("---")
st.markdown("This app uses a pre-trained RandomForestRegressor model to predict Boston housing prices.")
