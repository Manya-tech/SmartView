from flask import Flask, request, render_template
from .predict import predict_sentiment
from .youtube import get_video_comments
from .video import get_video
from flask_cors import CORS


app = Flask(__name__)
CORS(app)



def get_stats(video):
    video_id = video['video_id']
    if not video_id:
        return {"error" : "video_id is required"}
    comments = get_video_comments(video_id)
    print("got info")
    predictions  = predict_sentiment(comments)
    print("predicted")

    positive = predictions.count("Positive")  
    negative = predictions.count("Negative")
    comments_data = list(zip(comments[:10], predictions[:10]))
    # comms = " "
    # for comment in comments:
    #     comms = comms+comment
    # genout = get_gemini_response(comms)

    summary = {
        "title" : video['title'],
        "channel_name" : video['channel_name'],
        "comment_count" : video['comment_count'],
        "view_count" : video['view_count'],
        "like_count" : video['like_count'],
        "video_id" : video_id,
        "positive" : positive,
        "negative" : negative,
        "num_comments" : len(comments),
        "rating" : round((positive/len(comments))*100, 2),
        "comments_data" : comments_data,
        "thumbnail" : video['thumbnail'],
    }

    return summary


@app.route("/", methods=["GET","POST"])
def index():
    data = []
    if request.method == "POST":
        video_topic = request.form.get("video_topic")
        print(video_topic)
        # points = request.form.get("points")
        videos_info = get_video(video_topic)
        for video in videos_info:
            # video_id = video['video_id']
            res = get_stats(video)
            print("sent req")
            data.append(res)

    return render_template("index.html", data = data)

if __name__=="__main__":
    app.run(debug=True)