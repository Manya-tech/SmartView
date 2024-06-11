from flask import Flask, request, render_template
from predict import predict_sentiment
from youtube import get_video_info
from video import get_video_ids
from flask_cors import CORS
import pathlib
import textwrap

import google.generativeai as genai


genai.configure(api_key="AIzaSyDfaFCEi4LXUv1USOkI66Q2ZXwMG9KExmM")


app = Flask(__name__)
CORS(app)


def get_gemini_response(comments,points):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f""" You will get a list of comments of a YouTube video as input and the the main points the user wants in that video. 
Your role is to analyze the comments and create a summary that first mentions how many points the user wants match the video, then what are the main points discussed in a video and 
what are main themes in the comments. Each heading should have atmost 5 points. Just provide a brief overview that will help the reader to uderstand whether the video has what it wants or what the commentators thought about the video.
For example:
Points matching : mention points that matched in bullet point form.
Viewer insights : mention what people think about the video in bullet point form.
Main points: mention the main points discussed in the video in bullet point form.
The points are: {points}. The comments are: {comments}
"""
    response = model.generate_content(prompt)
    result=response.text
    # result = result[0]['text']
    return result



def get_video(video_id, points):
    if not video_id:
        return {"error" : "video_id is required"}
    name, comments = get_video_info(video_id)
    print("got info")
    predictions  = predict_sentiment(comments)
    print("predicted")

    positive = predictions.count("Positive")  
    negative = predictions.count("Negative")
    comments_data = list(zip(comments[:10], predictions[:10]))
    comms = " "
    for comment in comments:
        comms = comms+comment
    genout = get_gemini_response(comms,points)

    summary = {
        "name" : name,
        "video_id" : video_id,
        "positive" : positive,
        "negative" : negative,
        "num_comments" : len(comments),
        "rating" : round((positive/len(comments))*100, 2),
        "comments_data" : comments_data
        # "genout" : genout
    }

    return summary


@app.route("/", methods=["GET","POST"])
def index():
    data = []
    if request.method == "POST":
        video_ids = get_video_ids(request.form.get("video_url"))
        points = request.form.get("points")
        # video_urls = request.form.get("video_url")
        # video_urls = video_urls.split(",")
        for video in video_ids:
            video_id = video['video_id']
            res = get_video(video_id,points)
            print("sent req")
            data.append(res)

    return render_template("index.html", data = data)

if __name__=="__main__":
    app.run(debug=True)