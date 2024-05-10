from googleapiclient.discovery import build
import googleapiclient.discovery
import googleapiclient.errors
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")


youtube = build('youtube', 'v3', developerKey=api_key)

def get_video_ids(query):
    search_response = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=3
    ).execute()

    videos_with_comments = []
    for search_result in search_response.get('items', []):
        video_id = search_result['id']['videoId']
        video_info = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        ).execute()

        # Check if comments are enabled for the video
        if 'commentCount' in video_info['items'][0]['statistics']:
            videos_with_comments.append({
                'title': search_result['snippet']['title'],
                'video_id': video_id
            })
    print(videos_with_comments)

    return videos_with_comments


# # Example usage
# user_query = input("Enter your search query: ")
# top_videos = search_videos(user_query)

# for idx, video in enumerate(top_videos, start=1):
#     print(f"{idx}. Title: {video['title']}")
#     print(f"   Description: {video['description']}")
#     print(f"   Video ID: {video['video_id']}")
#     print(f"   Thumbnail URL: {video['thumbnail']}")
#     print()
