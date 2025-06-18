import numpy as np
import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os


def load_data():
    df = pd.read_csv('mail_data.csv')
    return df


def preprocess_text(text):
    # Simple preprocessing, can be expanded
    return text.lower()


def extract_features(df):
    # Use TF-IDF for text features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    X_tfidf = vectorizer.fit_transform(df['Message'].apply(preprocess_text))
    return X_tfidf, vectorizer


def train_naive_bayes():
    # Load features and labels
    X, y = joblib.load('features_labels.joblib')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Train Naive Bayes
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = nb_model.predict(X_test)
    print("Accuracy:", nb_model.score(X_test, y_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(nb_model, X, y, cv=5)
    print("\nCross-validation scores:", cv_scores)
    
    # Save model
    joblib.dump(nb_model, 'naive_bayes_model.joblib')
    print("Model saved.")


def load_model():
    """Load the Naive Bayes model and related components."""
    try:
        model = joblib.load('naive_bayes_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        feature_columns = joblib.load('feature_columns.joblib')
        return model, vectorizer, feature_columns
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None


def predict_email(text):
    model, vectorizer, feature_columns = load_model()
    from algorithms.preprocess_and_vectorize import preprocess_text, extract_email_features
    tfidf_features = vectorizer.transform([preprocess_text(text)]).toarray()
    engineered_features = extract_email_features(text)
    engineered_array = np.array([engineered_features[col] for col in feature_columns]).reshape(1, -1)
    combined_features = np.hstack([tfidf_features, engineered_array])
    pred = model.predict(combined_features)[0]
    prob = model.predict_proba(combined_features)[0].max()
    return pred, prob


if __name__ == '__main__':
    train_naive_bayes() 