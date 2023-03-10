from flask import Flask, render_template, request, send_from_directory, session
from sentiment_analyzer import analyze_sentiment
from music_generator import generate
import os
from random_msg import get_message
app = Flask(__name__)
app.secret_key = 'your_secret_key'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=["POST"])
def post_senti():
    if request.method == "POST":
        senti = request.form['senti']
        if request.form.get('length'):
            length = int(request.form.get('length')) * 5
        else:
            length = 50
        inst_id = int(request.form.get('font'))
        result = analyze_sentiment(senti)
        session['senti'] = result
        generate(result,length,inst_id)
        msg = get_message()
        return render_template('result.html', senti = result, msg = msg)

@app.route('/download')
def download():
    senti = session.get('senti')
    path = senti + '.mid'
    dir = 'static/generated'
    return send_from_directory(dir, path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
    #app.run(debug=True)