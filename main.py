import pandas as pd
import string
from textblob import TextBlob

# Load the dataset
file_path = 'user_reviews.csv'
data = pd.read_csv(file_path)
data.dropna(inplace=True)
# Assuming the column containing reviews is named 'review'
columns_to_keep = ['review']
data = data[columns_to_keep]

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['cleaned_review'] = data['review'].apply(preprocess_text)

# Sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

data['sentiment'] = data['cleaned_review'].apply(analyze_sentiment)

# Summary report
sentiment_counts = data['sentiment'].value_counts()
summary_report = pd.DataFrame({'Sentiment': sentiment_counts.index, 'Count': sentiment_counts.values})

# Save the summary report to a CSV file
summary_output_file_path = 'sentiment_summary.csv'
summary_report.to_csv(summary_output_file_path, index=False)

# Print the summary report
print(summary_report)
