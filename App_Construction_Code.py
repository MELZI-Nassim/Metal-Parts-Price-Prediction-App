import streamlit as st
import pandas as pd
import numpy as np
import os
import gzip
import pickle
from io import BytesIO

# Set the working directory to the specified path
os.chdir("C:\\Users\\Fateh-Nassim MELZI\\Documents\\AI_Projects\\Metal_Parts_Price_Prediction_Project\\App_Construction")

# Cache the model loading function to avoid reloading on every interaction
@st.cache_resource
def load_model():
    with gzip.open("Model.pkl.gz", 'rb') as file:
        return pickle.load(file)

# Cache the encoders loading function to avoid reloading on every interaction
@st.cache_resource
def load_encoders():
    with gzip.open("Encoders.pkl.gz", 'rb') as file:
        return pickle.load(file)

# Load the trained model and encoders
model = load_model()
encoders = load_encoders()

# Function to predict the cost of new part or parts
def predict(encoders, model, covariates: pd.DataFrame, target_name: str) -> pd.DataFrame:
    data = covariates.copy().dropna(axis=0)
    for covariate in data.columns.tolist():
        if covariate in encoders.keys():
            encoded = encoders[covariate].transform(data[covariate])
            data[covariate] = encoded

    estimators_prediction = (np.array([tree.predict(data.values) for tree in model.estimators_])).T
    y_hat = np.round(np.mean(estimators_prediction, axis=1), 2)
    y_hat_std = np.round(np.std(estimators_prediction, axis=1), 2)
    ci = 1.96 * (y_hat_std / np.sqrt(len(model.estimators_)))
    mean_deviation = (ci / y_hat) * 100

    data[target_name] = y_hat
    data['Std'] = y_hat_std
    data['Mean_deviation (%)'] = np.round(mean_deviation, 2)
    data['95%'] = np.round(ci, 2)
    return data

### Export the prediction results into excel format ###
def to_excel_data(data: pd.DataFrame, index:bool=False) -> None:        
    template_io = BytesIO()
    data.to_excel(template_io, index=index)
    template_io.seek(0,0)
    return template_io.read()



# Streamlit app title
st.title('Metal Parts Price Prediction App')

# Add explanation and image to the sidebar

st.sidebar.title("About:")
st.sidebar.write("""
This AI-powered application predicts the price of metal parts based on various features.
Leveraging advanced machine learning algorithms, it provides accurate price estimations to help you make informed decisions.
You can choose to predict the price for a single part or multiple parts by uploading an Excel file.
""")

st.sidebar.image("Metal_Parts.png", caption="", use_container_width=True)


# Initialize session state for user input
if 'user_input' not in st.session_state:
    st.session_state.user_input = {}

# Create a single radio button group for prediction mode
prediction_mode = st.radio("Choose the prediction mode:", ("One part", "Several parts"), index=0, horizontal=True)

# If the prediction mode is "One part"
if prediction_mode == "One part":
    st.write("Please fill the following fields to get the cost prediction for one part:")   
    for feature in model.feature_names_in_:
        if feature in encoders.keys():
            st.session_state.user_input[feature] = st.selectbox(label=f"Select {feature}", options=encoders[feature].classes_, help=f"Select a value for the {feature} variable")
        else:
            st.session_state.user_input[feature] = st.number_input(label=f"Enter {feature}", step=0.01, help=f"Enter a value for the {feature} variable")

    # Create a button to get the prediction
    if st.button("Get the cost prediction"):
        # Check if all the required fields are filled
        if None in st.session_state.user_input.values():
            st.error("Please fill all the required fields.")
        else:
            # Create a DataFrame from the user inputs
            user_input_df = pd.DataFrame(st.session_state.user_input, index=[0])

            # Get the prediction
            prediction = predict(encoders, model, covariates=user_input_df, target_name='Cost (euro)')

            # Display the prediction
            st.write(f"**Cost (euro)**: **{prediction['Cost (euro)'].values[0]}** ± **{prediction['95%'].values[0]}**.")
            st.write(f"The **mean deviation** represents **{prediction['Mean_deviation (%)'].values[0]}%** of the predicted value.")

# If the prediction mode is "Several parts"
else:
    cols = st.columns(2)
    excel_data = to_excel_data(data=pd.DataFrame(columns=model.feature_names_in_, index=None))
    cols[0].download_button('Download an Excel template', excel_data, file_name='template.xlsx', mime='application/excel', help='Download an Excel file with the features to be filled in')
    uploaded_file = st.file_uploader(label=f"Select a file with the parts to predict their {'Cost (euro)'}", type=["csv", "xls", "xlsx"], accept_multiple_files=False, key="pred_parts", help="Click on browser to upload your file")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            parts_df = pd.read_excel(uploaded_file)

            # Check if the necessary columns are present
            missing_columns = [col for col in model.feature_names_in_ if col not in parts_df.columns]
            if missing_columns:
                st.warning(f"The uploaded file is missing the following columns: {', '.join(missing_columns)}")
            else:
                # Get the predictions
                predictions = predict(encoders, model, covariates=parts_df, target_name='Cost (euro)')

                # Display the predictions
                st.write(predictions)

                # Convert predictions to Excel format
                excel_data = to_excel_data(data=predictions, index=True)

                # Create a download button for the predictions
                st.download_button('⬇️ Download the predictions', excel_data, file_name='predictions.xlsx', mime='application/excel', help='Download an Excel file with the predictions')

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")