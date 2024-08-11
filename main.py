# Import necessary libraries
import pandas as pd
import streamlit as st
import joblib

# Load the saved model and encoders
best_xgb = joblib.load('best_xgb_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')
one_hot_encoder = joblib.load('one_hot_encoder.joblib')

# Define feature columns
numerical_columns = ['gf', 'ga', 'xg', 'xga', 'sh', 'sot', 'fk', 'pk', 'pkatt', 'attendance', 'poss', 'dist']
categorical_columns = ['venue', 'team', 'opponent', 'formation', 'round', 'season', 'referee']

# Extract categories from the one-hot encoder
categories = {col: cat_list for col, cat_list in zip(categorical_columns, one_hot_encoder.categories_)}

# Streamlit app
st.title("Match Outcome Predictor")

# Get user input for numerical features
numerical_input = {}
for feature in numerical_columns:
    numerical_input[feature] = st.number_input(feature)

# Get user input for categorical features
categorical_input = {}
for feature in categorical_columns:
    categorical_input[feature] = st.selectbox(f"Select {feature}", options=categories[feature])

# Predict the match outcome when the user clicks the button
if st.button("Predict"):
    # Combine inputs into a DataFrame
    input_data = {**numerical_input, **categorical_input}
    input_df = pd.DataFrame([input_data])

    # Scale numerical features using the StandardScaler from the pipeline
    input_df[numerical_columns] = best_xgb.named_steps['preprocessor'].named_transformers_['num'].transform(input_df[numerical_columns])

    # One-hot encode categorical features using the loaded one-hot encoder
    cat_features_encoded = one_hot_encoder.transform(input_df[categorical_columns])
    cat_features_df = pd.DataFrame(cat_features_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_columns))

    # Combine numerical and categorical features
    input_df = pd.concat([input_df[numerical_columns], cat_features_df], axis=1)

    # Get the feature names after one-hot encoding
    feature_names = list(input_df.columns)

    # Ensure the columns match the training data
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Make the prediction
    predicted_label = best_xgb.named_steps['classifier'].predict(input_df)
    predicted_label_text = label_encoder.inverse_transform(predicted_label)[0]
    st.write("Predicted Match Outcome:", predicted_label_text)

    # Load your document as a string
    with open('Streamlit Doc.txt', 'r') as f:
        document_text = f.read()

    # Display the document as text
    st.text(document_text)

    # Create a download button
    st.download_button('Download Document', document_text, 'Streamlit Doc.txt', 'text/plain')