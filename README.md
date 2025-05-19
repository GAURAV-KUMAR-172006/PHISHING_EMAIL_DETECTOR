# Phishing Email Detector

## Phase 1: Setup & Data Collection

### Environment Setup

1. Install Python 3.8 or higher.
2. Create a virtual environment:
   ```sh
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On Mac/Linux:
     ```sh
     source venv/bin/activate
     ```
4. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Collect Datasets

- **Phishing emails:**
  - [PhishTank](https://www.phishtank.com/)
  - [Kaggle Phishing Datasets](https://www.kaggle.com/datasets/search?search=phishing)
- **Legitimate emails:**
  - [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)
  - [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)

### Combine and Label Emails
- Assign label `0` for legitimate, `1` for phishing.
- Combine into a single CSV file with columns: `text`, `label`. 