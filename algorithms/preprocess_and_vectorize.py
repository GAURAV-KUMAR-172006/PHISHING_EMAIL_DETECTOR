import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from algorithms.feature_utils import preprocess_text, extract_email_features

def preprocess_and_vectorize():
    # Read the dataset
    df = pd.read_csv('mail_data.csv')
    
    # Preprocess text
    df['processed_text'] = df['Message'].apply(preprocess_text)
    
    # Extract TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words='english'
    )
    tfidf_features = tfidf.fit_transform(df['processed_text']).toarray()
    
    # Extract engineered features
    engineered_features = []
    for text in df['Message']:
        features = extract_email_features(text)
        engineered_features.append(features)
    
    # Convert to DataFrame
    engineered_df = pd.DataFrame(engineered_features)
    
    # Save feature columns for later use
    feature_columns = engineered_df.columns.tolist()
    joblib.dump(feature_columns, 'feature_columns.joblib')
    
    # Combine features
    X = np.hstack([tfidf_features, engineered_df.values])
    y = (df['Category'] == 'spam').astype(int)  # Convert 'spam'/'ham' to 1/0
    
    # Save features and labels
    joblib.dump((X, y), 'features_labels.joblib')
    
    # Save vectorizer
    joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
    
    print("Preprocessing complete. Features, vectorizer, and columns saved.")

if __name__ == '__main__':
    preprocess_and_vectorize() 