import googleapiclient.discovery
import googleapiclient.errors
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

def get_comments(youtube, **kwargs):
 
    comments = []
    results = youtube.commentThreads().list(**kwargs)
    results = results.execute()

    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

     
        # check if there are more comments
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = youtube.commentThreads().list(**kwargs).execute()
        else:
            break

    return comments

def get_name(youtube, video_id):
    name = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    return name['items'][0]['snippet']['title']

def main(video_id):
    # Disable OAuthlib's HTTPs verification when running locally.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    youtube = googleapiclient.discovery.build(
        "youtube", "v3", developerKey = api_key)

    comments = get_comments(youtube, part="snippet", textFormat="plainText", videoId=video_id)
    name = get_name(youtube, video_id)
    return (name,comments)

def get_video_comments(video_id):
    return main(video_id)

def get_video_info(video_id):

    return main(video_id)

