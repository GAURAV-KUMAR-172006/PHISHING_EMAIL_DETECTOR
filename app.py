import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template
import logging
import re
from algorithms.feature_utils import extract_features

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

def check_phishing(email_text):
    # Extract features
    features = extract_features(email_text)
    
    # Calculate phishing score based on features
    score = 0
    max_score = 10
    
    # Check for suspicious patterns
    if features.get('has_spam_words', 0):
        score += 2
    if features.get('has_urgency', 0):
        score += 2
    if features.get('has_money_words', 0):
        score += 2
    if features.get('has_url', 0):
        score += 1
    if not features.get('has_proper_formatting', 1):
        score += 1
    if features.get('has_special_chars', 0):
        score += 1
    if features.get('has_emoji', 0):
        score += 1
    
    # Calculate confidence scores
    phishing_confidence = score / max_score
    safe_confidence = 1 - phishing_confidence
    
    # Determine prediction
    is_phishing = phishing_confidence > 0.5
    
    return {
        'prediction': 'phishing' if is_phishing else 'safe',
        'confidence': {
            'random_forest': phishing_confidence,
            'naive_bayes': phishing_confidence,
            'weighted_score': phishing_confidence
        },
        'analysis': {
            'text_analysis': {
                'text_length': features.get('text_length', 0),
                'word_count': features.get('word_count', 0),
                'sentence_count': features.get('sentence_count', 0),
                'avg_word_length': features.get('avg_word_length', 0),
                'has_question_marks': bool(features.get('has_question_marks', 0)),
                'has_exclamation_marks': bool(features.get('has_exclamation_marks', 0)),
                'has_proper_formatting': bool(features.get('has_proper_formatting', 0)),
                'has_signature': bool(features.get('has_signature', 0))
            },
            'url_analysis': {
                'url_count': features.get('url_count', 0),
                'has_url': bool(features.get('has_url', 0))
            },
            'spam_indicators': {
                'has_spam_words': bool(features.get('has_spam_words', 0)),
                'has_click_here': bool(features.get('has_click_here', 0)),
                'has_urgency': bool(features.get('has_urgency', 0)),
                'has_money_words': bool(features.get('has_money_words', 0))
            },
            'legitimate_indicators': {
                'has_legitimate_patterns': bool(features.get('has_legitimate_patterns', 0))
            },
            'special_characters': {
                'has_emoji': bool(features.get('has_emoji', 0)),
                'has_arrow_symbols': bool(features.get('has_arrow_symbols', 0)),
                'has_special_chars': bool(features.get('has_special_chars', 0)),
                'has_numbers': bool(features.get('has_numbers', 0)),
                'has_currency_symbols': bool(features.get('has_currency_symbols', 0))
            }
        }
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'email_text' not in data:
            return jsonify({'error': 'No email text provided'}), 400

        email_text = data['email_text']
        result = check_phishing(email_text)
        
        # Add reasons for the prediction
        reasons = []
        if result['prediction'] == 'phishing':
            if result['analysis']['spam_indicators']['has_spam_words']:
                reasons.append("Contains suspicious words commonly found in phishing emails")
            if result['analysis']['spam_indicators']['has_urgency']:
                reasons.append("Uses urgent language to create pressure")
            if result['analysis']['spam_indicators']['has_money_words']:
                reasons.append("Contains financial terms often used in phishing attempts")
            if result['analysis']['url_analysis']['has_url']:
                reasons.append("Contains URLs that may be suspicious")
            if not result['analysis']['text_analysis']['has_proper_formatting']:
                reasons.append("Poor email formatting typical of phishing attempts")
        else:
            if result['analysis']['text_analysis']['has_proper_formatting']:
                reasons.append("Well-formatted email with proper structure")
            if result['analysis']['text_analysis']['has_signature']:
                reasons.append("Contains a professional signature")
            if result['analysis']['legitimate_indicators']['has_legitimate_patterns']:
                reasons.append("Contains patterns typical of legitimate emails")
            if not result['analysis']['spam_indicators']['has_spam_words']:
                reasons.append("No suspicious words detected")
        
        result['reasons'] = reasons
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Error processing email'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True) 