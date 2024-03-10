from flask import Flask, request, render_template
from predict import predict_sentiment
from youtube import get_video_info
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

def get_video(video_id):
    if not video_id:
        return {"error" : "video_id is required"}
    name, comments = get_video_info(video_id)
    predictions  = predict_sentiment(comments)

    positive = predictions.count("Positive")  
    negative = predictions.count("Negative")
    comments_data = list(zip(comments[:10], predictions[:10]))

    summary = {
        "name" : name,
        "positive" : positive,
        "negative" : negative,
        "num_comments" : len(comments),
        "rating" : round((positive/len(comments))*100, 2),
        "comments_data" : comments_data
    }

    return summary


@app.route("/", methods=["GET","POST"])
def index():
    # summary = []
    # comments = []
    data = []
    if request.method == "POST":
        video_urls = request.form.get("video_url")
        video_urls = video_urls.split(",")
        for video_url in video_urls:
            video_id = video_url.split("v=")[1]
            res = get_video(video_id)
            data.append(res)

    return render_template("index.html", data = data)

if __name__=="__main__":
    app.run(debug=True)