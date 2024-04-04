import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus.util import LazyCorpusLoader

# nltk.data.path.append('/opt/nltk_data')
nltk.download('vader_lexicon')

# 単語の感情分析
def analyze_sentiment(word):
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(word)
    if sentiment["compound"] >= 0.5:
        return "pos"
    elif sentiment["compound"] <= -0.5:
        return "neg"
    elif sentiment["neu"] == 1:
        return "neu"
    else:
        return "mix"
