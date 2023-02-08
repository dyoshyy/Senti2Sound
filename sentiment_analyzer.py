import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# 単語の感情分析
def analyze_sentiment(word):
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(word)
    if sentiment['compound'] >= 0.5:
        return 'pos'
    elif sentiment['compound'] <= -0.5:
        return 'neg'
    elif sentiment['neu'] == 1:
        return 'neu'
    else:
        return 'mix'
