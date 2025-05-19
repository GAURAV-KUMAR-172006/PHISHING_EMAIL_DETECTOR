import pandas as pd
import joblib

def main():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    df = pd.read_csv('unseen_emails.csv')
    X = vectorizer.transform(df['text'].str.lower())
    preds = model.predict(X)
    df['prediction'] = preds
    df.to_csv('predictions.csv', index=False)
    print('Predictions saved to predictions.csv')

if __name__ == '__main__':
    main() 