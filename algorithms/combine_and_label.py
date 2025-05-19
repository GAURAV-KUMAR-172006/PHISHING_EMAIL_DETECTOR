import pandas as pd

def main():
    # Example: combine two CSVs and add a label column
    df1 = pd.read_csv('legit_emails.csv')
    df1['label'] = 0
    df2 = pd.read_csv('phishing_emails.csv')
    df2['label'] = 1
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv('all_emails_labeled.csv', index=False)
    print('Combined and labeled emails saved to all_emails_labeled.csv')

if __name__ == '__main__':
    main() 