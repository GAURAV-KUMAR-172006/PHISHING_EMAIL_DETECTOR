import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_model():
    """Load the trained Random Forest model."""
    try:
        model = joblib.load('model.pkl')
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def train_and_evaluate():
    # Load features and labels
    X, y = joblib.load('features_labels.joblib')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    # Train model
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    print("Accuracy:", rf_model.score(X_test, y_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=5)
    print("\nCross-validation scores:", cv_scores)
    
    # Save model
    joblib.dump(rf_model, 'model.pkl')
    print("Model saved.")

if __name__ == '__main__':
    train_and_evaluate() 