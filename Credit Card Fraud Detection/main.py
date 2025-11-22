import base64
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
from PIL import Image
from io import StringIO, BytesIO

# Page Configuration
st.set_page_config(
    page_title='Simple Prediction App',
    layout='wide',
    initial_sidebar_state='auto',
)

# Download dataset if not exists
def download_dataset():
    from kaggle.api.kaggle_api_extended import KaggleApi
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/creditcard.csv"):
        api = KaggleApi()
        api.set_config_value('username', st.secrets["kaggle"]["username"])
        api.set_config_value("key", st.secrets["kaggle"]["key"])
        api.authenticate()
        api.dataset_download_files("mlg-ulb/creditcardfraud", path="data", unzip=True)

download_dataset()

# Load image
try:
    img = Image.open('img/overview_photo.jpg')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_bs64 = base64.b64encode(buffered.getvalue()).decode()
    body = f"""
    <div style="text-align: center;">
        <img src="data:image/jpeg;base64,{img_bs64}" width="250" height="200"/>
    </div>
    """
    st.markdown(body, unsafe_allow_html=True)
except Exception as e:
    st.warning("Image not found or failed to load.")

# Title
st.title('🔓Credit Card Fraud Detection App')
st.write("It contains only numerical input variables which are the result of a PCA transformation. \
         Unfortunately, due to confidentiality issues, we cannot provide the original features and more \
         background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, \
         the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the \
         seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the \
         transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the \
         response variable and it takes value 1 in case of fraud and 0 otherwise.")

# Load dataset 
credit_card_df = pd.read_csv("data/creditcard.csv")
credit_card_df.rename(columns={'Amount': 'Transaction_Amount',
                               'Class': 'Is_Fraudulent'}, inplace=True)

# Data Overview
st.subheader("📊Data Overview")
st.dataframe(data=credit_card_df.head(), height=200)

# EDA
with st.expander("📌Credit Card Fraud EDA"):
    st.write("**Describe:**")
    st.dataframe(credit_card_df.describe(), height=250)
    st.write("**Info:**")
    buffer = StringIO()
    credit_card_df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.write("**Shape:**", credit_card_df.shape)
    st.write("**Missing Values:**")
    st.dataframe(credit_card_df.isna().sum())

# Correlation Matrix
with st.expander("📈Correlation Matrix"):
    corr = credit_card_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="Blues", annot=False)
    st.pyplot(fig)

# ML Model
st.subheader("🤖Machine Learning Model")
X = credit_card_df.drop(['Is_Fraudulent'], axis=1)
y = credit_card_df['Is_Fraudulent']

# Remove outliers using IQR method
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
X = X[mask]
y = y[mask]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Scale the data
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

with st.expander("🔍Logistic Regression Model"):
    # Logistic Regression with custom threshold
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_resampled_scaled, y_resampled)
    
    # Get predictions and apply custom threshold
    threshold = 0.3  # Example: Set the threshold to 0.3
    log_y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]  # Get probabilities for class 1
    log_y_pred_custom = (log_y_pred_prob >= threshold).astype(int)  # Apply threshold
    
    # Evaluation
    accuracy = accuracy_score(y_test, log_y_pred_custom)
    precision = precision_score(y_test, log_y_pred_custom)
    recall = recall_score(y_test, log_y_pred_custom)
    f1 = f1_score(y_test, log_y_pred_custom)
    roc_auc = roc_auc_score(y_test, log_y_pred_custom)
    
    st.success(f"✅Training Score: {log_reg.score(X_resampled_scaled, y_resampled):.2f}")
    st.success(f"✅Testing Score: {log_reg.score(X_test_scaled, y_test):.2f}")
    
    # Display metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
        'Score': [accuracy, precision, recall, f1, roc_auc]
    })
    st.dataframe(metrics_df.style.format({"Score": "{:.2f}"}), height=250)
