# Phishing Email Detection

Team project developing a machine learning pipeline to detect phishing emails. The system classifies email components as legitimate or suspicious, with a current focus on **subject lines** and **URLs**. Planned expansions will incorporate additional features for improved detection.

## Authors
- **Kali Ferguson** – responsible for **subject line classification** pipeline, including preprocessing and feature engineering – `@Arial1000`
- Siddi Ahmed Brahim – responsible for **URL classification** – `@Sidiahmedde`
- Mansi Patel – responsible for **sender classification**
- Sagar Bhandari– responsible for **body classification**

## Project Overview
- **Model:** Logistic Regression  
- **Dataset:** Public email dataset from [Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?resource=download)  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

## Features
- **Subject Line Classification:** Preprocessing, feature extraction, and logistic regression modeling to classify subject lines  
- **URL Classification:** Separate logistic regression model for URLs  
- **Evaluation:** Measures model performance using standard classification metrics  
- **Extensible Pipeline:** Designed to add more email features (body text, email header, sender info, etc.) in future iterations

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/phishing-detector.git
2. Install dependencies:   
   `pip install -r requirements.txt`
3. Run the model pipeline:   
   `python predict_subject.py  # for subject classification`   
   `python train_url_model.py      # for URL classification`

