from flask import Flask, render_template, request, send_from_directory
from sentiment_analyzer import analyze_sentiment
from music_generator import generate
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=["POST"])
def post_senti():
    if request.method == "POST":
        senti = request.form['senti']
        result = analyze_sentiment(senti)
        generate(result)
        return render_template('result.html', senti = result)

if __name__ == "__main__":
    app.run(debug=True)