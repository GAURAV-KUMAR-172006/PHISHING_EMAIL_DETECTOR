import numpy as np
import re
import email
from email.parser import Parser
from urllib.parse import urlparse
import tldextract
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    """Preprocess text by removing special characters and converting to lowercase."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_url_features(url):
    """Extract features from a URL."""
    features = {}
    try:
        parsed = urlparse(url)
        ext = tldextract.extract(url)
        
        features['url_length'] = len(url)
        features['url_has_https'] = 1 if parsed.scheme == 'https' else 0
        features['url_has_www'] = 1 if 'www.' in url else 0
        features['url_has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
        features['url_has_port'] = 1 if ':' in parsed.netloc else 0
        features['url_has_query'] = 1 if parsed.query else 0
        features['url_has_fragment'] = 1 if parsed.fragment else 0
        features['url_has_at_symbol'] = 1 if '@' in url else 0
        features['url_has_dash'] = 1 if '-' in url else 0
        features['url_has_underscore'] = 1 if '_' in url else 0
        features['url_has_tilde'] = 1 if '~' in url else 0
        features['url_has_dot'] = 1 if '.' in parsed.path else 0
        features['url_has_slash'] = 1 if '/' in parsed.path else 0
        features['url_has_equal'] = 1 if '=' in url else 0
        features['url_has_question'] = 1 if '?' in url else 0
        features['url_has_hash'] = 1 if '#' in url else 0
        features['url_has_percent'] = 1 if '%' in url else 0
        features['url_has_ampersand'] = 1 if '&' in url else 0
        features['url_has_plus'] = 1 if '+' in url else 0
        features['url_has_comma'] = 1 if ',' in url else 0
        features['url_has_semicolon'] = 1 if ';' in url else 0
        features['url_has_colon'] = 1 if ':' in url else 0
        features['url_has_exclamation'] = 1 if '!' in url else 0
        features['url_has_dollar'] = 1 if '$' in url else 0
        features['url_has_space'] = 1 if ' ' in url else 0
        features['url_has_quotes'] = 1 if '"' in url or "'" in url else 0
        features['url_has_less_than'] = 1 if '<' in url else 0
        features['url_has_greater_than'] = 1 if '>' in url else 0
        features['url_has_brackets'] = 1 if '[' in url or ']' in url else 0
        features['url_has_braces'] = 1 if '{' in url or '}' in url else 0
        features['url_has_pipe'] = 1 if '|' in url else 0
        features['url_has_backslash'] = 1 if '\\' in url else 0
        features['url_has_caret'] = 1 if '^' in url else 0
        features['url_has_grave'] = 1 if '`' in url else 0
        features['url_has_asterisk'] = 1 if '*' in url else 0
        features['url_has_parentheses'] = 1 if '(' in url or ')' in url else 0
        
        # Domain features
        features['domain_length'] = len(ext.domain)
        features['subdomain_length'] = len(ext.subdomain)
        features['tld_length'] = len(ext.suffix)
        features['has_subdomain'] = 1 if ext.subdomain else 0
        features['is_free_domain'] = 1 if ext.domain in ['gmail', 'yahoo', 'hotmail', 'outlook', 'aol', 'mail', 'protonmail', 'tutanota', 'zoho', 'yandex'] else 0
        
        # Suspicious patterns
        features['has_suspicious_words'] = 1 if re.search(r'(verify|confirm|account|suspended|security|claim|prize|bank|login|signin|password|update|secure|action|click|here|now|urgent|important|free|trial|offer|discount|sale|limited|time|miracle|weight|loss|diet|pill|supplement)', url.lower()) else 0
        features['has_suspicious_tld'] = 1 if ext.suffix in ['xyz', 'top', 'loan', 'work', 'site', 'online', 'website', 'space', 'click', 'link', 'bid', 'win', 'download', 'stream', 'live', 'video', 'audio', 'music', 'movie', 'film', 'show', 'watch', 'view', 'see', 'look', 'check', 'verify', 'confirm', 'account', 'login', 'signin', 'password', 'security', 'secure', 'protect', 'safeguard', 'shield', 'guard', 'defend', 'secure', 'safe', 'protect', 'guard', 'shield', 'defend', 'safeguard', 'secure', 'safe', 'protect', 'guard', 'shield', 'defend', 'safeguard'] else 0
        
    except:
        features = {k: 0 for k in features.keys()}
    
    return features

def extract_features(text):
    """Extract features from email text."""
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    
    # Initialize features dictionary with all required features
    features = {
        # Basic text features
        'text_length': len(text),
        'word_count': len(text.split()),
        'sentence_count': len(re.split(r'[.!?]+', text)),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        'has_question_marks': 1 if '?' in text else 0,
        'has_exclamation_marks': 1 if '!' in text else 0,
        'has_caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'has_proper_formatting': 1 if re.search(r'\n\n|\r\n\r\n', text) else 0,
        'has_signature': 1 if re.search(r'\b(regards|sincerely|best|thanks|thank you|cheers|best regards|yours truly|yours sincerely)\b', text.lower()) else 0,
        
        # URL features
        'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
        'has_url': 1 if re.search(r'http[s]?://', text) else 0,
        
        # Spam/Phishing specific patterns
        'has_spam_words': 1 if re.search(r'\b(free|trial|offer|discount|sale|limited|time|miracle|weight|loss|diet|pill|supplement|guarantee|money|back|refund|cash|prize|winner|lottery|inheritance|fortune|investment|profit|earn|income|work from home|make money|quick cash|easy money|get rich|millionaire|billionaire)\b', text.lower()) else 0,
        'has_click_here': 1 if re.search(r'\b(click here|click below|click the link|click this link|click to|click now|claim now|get started|sign up|register|subscribe|order now|buy now|shop now|learn more|read more|find out more|discover more|see more|view more)\b', text.lower()) else 0,
        'has_urgency': 1 if re.search(r'\b(limited time|act now|don\'t miss|hurry|expires|ending soon|last chance|final offer|exclusive|special|one time|once in a lifetime|never again|limited supply|while supplies last|only|just|now|today|tomorrow|immediately|asap|right away|urgently|quickly)\b', text.lower()) else 0,
        'has_money_words': 1 if re.search(r'\b(money|payment|bank|account|transfer|fund|dollar|euro|pound|cash|deposit|withdraw|balance|credit|debit|refund|invoice|bill|payment|transaction|price|cost|fee|charge|payment|pay|paid|free|trial|offer|discount|sale|limited|time)\b', text.lower()) else 0,
        
        # Legitimate email patterns
        'has_legitimate_patterns': 1 if re.search(r'\b(meeting|appointment|schedule|calendar|event|conference|call|discussion|project|team|work|office|company|business|client|customer|service|support|help|assist|information|details|update|report|document|file|attachment|link|reference|regards|sincerely|best|thanks|thank you|cheers|best regards|yours truly|yours sincerely)\b', text.lower()) else 0,
        
        # Additional features
        'has_emoji': 1 if re.search(r'[\U0001F300-\U0001F9FF]', text) else 0,
        'has_arrow_symbols': 1 if re.search(r'[→←↑↓↔↕]', text) else 0,
        'has_special_chars': 1 if re.search(r'[!@#$%^&*()_+\-=\[\]{};\'"\\|,.<>\/?]', text) else 0,
        'has_numbers': 1 if re.search(r'\d+', text) else 0,
        'has_currency_symbols': 1 if re.search(r'[$€£¥]', text) else 0
    }
    
    # Header-aware features: sender-domain vs link-domain consistency and trust
    sender_domain_match = re.search(r'^from:\s*([^\n]+)$', text, re.IGNORECASE | re.MULTILINE)
    sender_domain_flag = 0
    link_domain_flag = 0
    sender_link_domain_match = 0
    is_trusted_sender = 0
    trusted_etld1 = {
        'github.com', 'google.com', 'microsoft.com', 'apple.com', 'amazon.com', 'facebook.com', 'meta.com',
        'stripe.com', 'paypal.com', 'adobe.com', 'slack.com', 'zoom.us', 'atlassian.com', 'jira.com', 'bitbucket.org',
        'gitlab.com', 'cloudflare.com', 'dropbox.com', 'notion.so', 'openai.com', 'yahoo.com', 'outlook.com'
    }

    # Extract URLs early for domain checks
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    primary_link_domain = None
    if urls:
        try:
            ext_primary = tldextract.extract(urls[0])
            primary_link_domain = f"{ext_primary.domain}.{ext_primary.suffix}" if ext_primary.suffix else ext_primary.domain
            link_domain_flag = 1 if primary_link_domain else 0
        except Exception:
            primary_link_domain = None
    # Extract sender domain from From: header
    if sender_domain_match:
        sender_field = sender_domain_match.group(1)
        # Try to extract email within angle brackets or as-is
        m = re.search(r'<([^>]+)>', sender_field)
        email_addr = m.group(1) if m else sender_field.strip()
        m2 = re.search(r'@([A-Za-z0-9.-]+)', email_addr)
        if m2:
            raw_domain = m2.group(1).lower()
            ext_sender = tldextract.extract(raw_domain)
            sender_etld1 = f"{ext_sender.domain}.{ext_sender.suffix}" if ext_sender.suffix else ext_sender.domain
            sender_domain_flag = 1 if sender_etld1 else 0
            if primary_link_domain and sender_etld1 == primary_link_domain:
                sender_link_domain_match = 1
            if sender_etld1 in trusted_etld1:
                is_trusted_sender = 1
    if urls:
        url_features = [extract_url_features(url) for url in urls]
        for feature in url_features[0].keys():
            features[f'url_{feature}'] = np.mean([f[feature] for f in url_features])
    else:
        # Add default URL features if no URLs found
        default_url_features = extract_url_features('http://example.com')
        for feature in default_url_features.keys():
            features[f'url_{feature}'] = 0

    # Inject header-aware numeric features (no strings to keep model pipelines stable)
    features['sender_domain_present'] = sender_domain_flag
    features['primary_link_domain_present'] = link_domain_flag
    features['sender_link_domain_match'] = sender_link_domain_match
    features['is_trusted_sender'] = is_trusted_sender
    
    # Convert boolean values to integers
    for key in features:
        if isinstance(features[key], bool):
            features[key] = 1 if features[key] else 0
    
    return features

def extract_email_features(text):
    """Extract features from email text (legacy function)."""
    return extract_features(text) 