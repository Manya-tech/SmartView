from googleapiclient.discovery import build
import googleapiclient.discovery
import googleapiclient.errors
import os
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()
api_key = os.getenv("API_KEY")


youtube = build('youtube', 'v3', developerKey=api_key)

def get_video(query):
    search_response = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=3
    ).execute()

    # search_response = youtube.videos().list(
    # part="snippet,contentDetails,statistics",
    # chart="mostPopular",
    # maxResults=3,
    # regionCode="US"
    # ).execute()

    # pprint(search_response)

    videos_info = []
    for search_result in search_response.get('items', []):
        video_id = search_result['id']['videoId']
        video_info = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        ).execute()

        # Check if comments are enabled for the video
        if 'commentCount' in video_info['items'][0]['statistics']:
            videos_info.append({
                'video_id': video_id,
                'title': search_result['snippet']['title'],
                'channel_name' : search_result['snippet']['channelTitle'],
                'description' : search_result['snippet']['description'],
                'comment_count': video_info['items'][0]['statistics']['commentCount'],
                'view_count' : video_info['items'][0]['statistics']['viewCount'],
                "like_count": video_info['items'][0]['statistics']['likeCount']
            })

    return videos_info


# # Example usage
# user_query = input("Enter your search query: ")
# top_videos = get_video(user_query)

# for idx, video in enumerate(top_videos, start=1):
#     print(f"{idx}. Title: {video['title']}")
#     print(f"   Description: {video['description']}")
#     print(f"   Video ID: {video['video_id']}")
#     print(f"   Channel Name: {video['channel_name']}")
#     print(f"   Comment Count: {video['comment_count']}")
#     print(f"   View Count: {video['view_count']}")
#     print(f"   Like Count: {video['like_count']}")
#     print()