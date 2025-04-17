### Import the necessary libraries ### 
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# Constants
TEMPLATE_PATH = "Template.xlsx"
UPLOAD_SIZE_LIMIT_MB = 5120  # 5 Go = 5120 Mo

# Set the working directory to the specified path
os.chdir("C:\\Users\\Fateh-Nassim MELZI\\Documents\\AI_Projects\\Metal_Parts_Price_Prediction_Project\\App_Construction")

# Set page configuration
st.set_page_config(page_title="Metal Parts Price Prediction", page_icon="🛠️", layout="centered")

# Set max upload size
os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = str(UPLOAD_SIZE_LIMIT_MB)

# Cache the model loading function to avoid reloading on every interaction
@st.cache_resource
def load_model():
    try:
        return joblib.load("Model.pkl.xz")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'Model.pkl.xz' is in the correct directory.")
        return None

# Cache the encoders loading function to avoid reloading on every interaction
@st.cache_resource
def load_encoders():
    try:
        return joblib.load("Encoders.pkl.xz")
    except FileNotFoundError:
        st.error("Encoders file not found. Please ensure 'Encoders.pkl.xz' is in the correct directory.")
        return None

def validate_input(input_data):
    for key, value in input_data.items():
        if value is None or (isinstance(value, (int, float)) and value < 0):
            return False
    return True

# Load the trained model and encoders
model = load_model()
encoders = load_encoders()

# Function to predict the cost of new part or parts
def predict(encoders, model, covariates: pd.DataFrame, target_name: str) -> pd.DataFrame:
    original_data = covariates.copy().dropna(axis=0)  # Keep a copy of the original data
    data = covariates.dropna(axis=0)
    for covariate in data.columns.tolist():
        if covariate in encoders.keys():
            encoded = encoders[covariate].transform(data[covariate])
            data[covariate] = encoded

    estimators_prediction = (np.array([tree.predict(data.values) for tree in model.estimators_])).T
    y_hat = np.round(np.mean(estimators_prediction, axis=1), 2)
    y_hat_std = np.round(np.std(estimators_prediction, axis=1), 2)
    ci = 1.96 * (y_hat_std / np.sqrt(len(model.estimators_)))
    mean_deviation = (ci / y_hat) * 100

    original_data[target_name] = y_hat
    original_data['± Confidence interval (euro)'] = np.round(ci, 2)
    original_data['Mean_deviation (%)'] = np.round(mean_deviation, 2)
    
    return original_data

# Export the prediction results into excel format
def to_excel_data(data: pd.DataFrame, index: bool = False) -> bytes:
    template_io = BytesIO()
    data.to_excel(template_io, index=index)
    template_io.seek(0, 0)
    return template_io.read()

# Create a template with headers and categorical variable modalities
def create_template(encoders):
    template = pd.DataFrame(columns=model.feature_names_in_)
    for col, encoder in encoders.items():
        template[col] = [', '.join(map(str, encoder.classes_))]
    return template

# Load the pre-prepared Excel template
def load_template(file_path: str) -> bytes:
    with open(file_path, "rb") as file:
        return file.read()

# Function to validate the uploaded file
def validate_uploaded_file(parts_df: pd.DataFrame) -> bool:
    # Check for missing columns
    missing_columns = [col for col in model.feature_names_in_ if col not in parts_df.columns]
    if missing_columns:
        st.warning(f"The uploaded file is missing the following columns: {', '.join(missing_columns)}")
        return False

    # Validate categorical variables
    invalid_categories = {}
    for feature in encoders.keys():
        if feature in parts_df.columns:
            invalid_values = parts_df[~parts_df[feature].isin(encoders[feature].classes_)][feature].unique()
            if len(invalid_values) > 0:
                invalid_categories[feature] = invalid_values

    if invalid_categories:
        for feature, values in invalid_categories.items():
            st.warning(f"The following values for {feature} are invalid: {', '.join(map(str, values))}")
        return False

    # Validate numerical variables
    invalid_numerics = {}
    for feature in numerical_bounds.keys():
        if feature in parts_df.columns:
            invalid_values = parts_df[parts_df[feature].isnull() | ~parts_df[feature].apply(lambda x: isinstance(x, (int, float)))][feature].unique()
            if len(invalid_values) > 0:
                invalid_numerics[feature] = invalid_values

    if invalid_numerics:
        for feature, values in invalid_numerics.items():
            st.warning(f"The following values for {feature} are invalid or missing: {', '.join(map(str, values))}")
        return False

    # Validate numerical values within bounds
    out_of_bounds = {}
    for feature, bounds in numerical_bounds.items():
        if feature in parts_df.columns:
            out_of_bounds_values = parts_df[(parts_df[feature] < bounds["min"]) | (parts_df[feature] > bounds["max"])][feature].unique()
            if len(out_of_bounds_values) > 0:
                out_of_bounds[feature] = out_of_bounds_values

    if out_of_bounds:
        for feature, values in out_of_bounds.items():
            st.warning(f"The following values for {feature} are out of bounds: {', '.join(map(str, values))}")
        return False

    return True

# Streamlit app title
st.header('🛠️ Metal Parts Price Prediction')

# Add explanation and image to the sidebar
st.sidebar.title("ℹ️ About:")
st.sidebar.write("""
This AI-powered application 🤖 predicts the price of metal parts based on various characteristics. 
Leveraging advanced machine learning algorithms 🧠, it provides accurate price estimations to help you make informed decisions 📊. 
You can choose to predict the price for a single part or multiple parts by uploading an Excel file. 
""")

st.sidebar.image("Metal_Parts_Image.png", caption="", use_container_width=True)

# Define min and max values for each numerical feature
numerical_bounds = {
    'Batch size': {"min": 1.00, "max": 4618.00, "default": 5.00},
    'Finish mass (kg)': {"min": 0.01, "max": 97.00, "default": 1.00},
    'Rough mass (kg)': {"min": 0.013, "max": 99.00, "default": 1.76},
    'Thickness (mm)': {"min": 0.1, "max": 9.8, "default": 0.1},
    # Add more covariates as needed
}

# Initialize session state for user input
if 'user_input' not in st.session_state:
    st.session_state.user_input = {}

# Create a single radio button group for prediction mode
prediction_mode = st.radio("Choose the prediction mode:", ("One part", "Several parts"), index=0, horizontal=True)

# If the prediction mode is "One part"
if prediction_mode == "One part":
    st.write("Please fill the following fields 👇 to get the price for one part:")   
    for feature in model.feature_names_in_:
        if feature in encoders.keys():
            st.session_state.user_input[feature] = st.selectbox(
                label=feature, 
                options=encoders[feature].classes_, 
                help=f"Select a value for the {feature}"
            )
        else:
            bounds = numerical_bounds.get(feature, {"min": 0.0, "max": 100.0, "default": 50.0})
            st.session_state.user_input[feature] = st.number_input(
                label=feature, 
                value=bounds["default"], 
                min_value=bounds["min"], 
                max_value=bounds["max"], 
                step=0.01, 
                help=f"Enter a value for the {feature}"
            )

    # Create a button to get the prediction
    if st.button("Get the price", help="Click to get the price of the part"):
        # Check if all the required fields are filled
        if not validate_input(st.session_state.user_input):
            st.error("Please fill all the required fields correctly.")
        else:
            # Create a DataFrame from the user inputs
            user_input_df = pd.DataFrame(st.session_state.user_input, index=[0])

            # Get the prediction
            prediction = predict(encoders, model, covariates=user_input_df, target_name='Price (euro)')

            # Display the prediction
            st.success(f"**Price (euro)**: **{prediction['Price (euro)'].values[0]}** ± **{prediction['± Confidence interval (euro)'].values[0]}**.") 
            st.success(f"The **mean deviation** represents **{prediction['Mean_deviation (%)'].values[0]}%** of the predicted price.")

# If the prediction mode is "Several parts"
else:
    cols = st.columns(2)
    template = load_template(TEMPLATE_PATH)
    cols[0].download_button('Download a template', template, file_name='template.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', help='Download an Excel file template')
    
    # Add an information note
    st.info("""
    ⚠️ **Note:** please ensure that the headers and categorical variable values in your uploaded file match exactly as specified in the template. The order of the headers is also important. This will help avoid errors during the prediction process.
    """)
    
    uploaded_file = st.file_uploader(label="Select a file with the parts to predict their price (euro)", type=["csv", "xls", "xlsx"], accept_multiple_files=False, key="pred_parts", help="Click on browser to upload your file")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            parts_df = pd.read_excel(uploaded_file)

            # Validate the uploaded file
            if validate_uploaded_file(parts_df):
                # Get the predictions
                predictions = predict(encoders, model, covariates=parts_df, target_name='Price (euro)')

                # Display the predictions
                st.write(predictions)

                # Convert predictions to Excel format
                excel_data = to_excel_data(data=predictions, index=True)

                # Create a download button for the predictions
                st.download_button('⬇️ Download the predictions', excel_data, file_name='predictions.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', help='Download an Excel file with the predictions')

        except Exception as e:
            st.error("An error occurred while processing the file.")