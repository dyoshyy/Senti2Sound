import os
import sys
import traceback

sys.path.insert(0, "/workspaces/Senti2Sound/src")

from flask import Flask, abort, render_template, request, send_from_directory, session

from .my_modules.music_generator import generate
from .my_modules.random_msg import get_message
from .my_modules.sentiment_analyzer import analyze_sentiment


app = Flask(__name__)
app.secret_key = "your_secret_key"


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/generate", methods=["POST"])
def post_senti():
    try:
        if request.method == "POST":
            
            senti = request.form["senti"]
            
            # 曲の長さを指定
            if request.form.get("length"):
                length = int(request.form.get("length")) * 5
            else:
                length = 30
                
            # 楽器の種類を指定
            inst_id = int(request.form.get("font"))
            
            # 単語から感情を分析
            result = analyze_sentiment(senti)
            session["senti"] = result
            
            # 楽曲生成
            generate(result, length, inst_id)
            
            # ひとことメッセージを生成
            msg = get_message()
            
            return render_template("result.html", senti=result, msg=msg)
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()


@app.route("/download", methods=["GET"])
def download():
    if request.method == "GET":
        senti = session.get("senti")
        path = senti + ".mid"
        dir = "./static/generated"
        return send_from_directory(dir, path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
    # app.run(debug=True)
