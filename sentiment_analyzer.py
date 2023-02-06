import nltk

nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

# 単語の感情分析
def analyze_sentiment(word):
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(word)
    if sentiment['compound'] >= 0.5:
        return 'positive'
    elif sentiment['compound'] <= -0.5:
        return 'negative'
    elif sentiment['neu'] == 1:
        return 'neutral'
    else:
        return 'mixed'

# 入力した単語の感情分析
word = input('Enter a word: ')
sentiment = analyze_sentiment(word)
print('Sentiment:', sentiment)