import os
import re
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
import requests

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    parsed = urlparse(url)
    if parsed.hostname == 'youtu.be':
        return parsed.path[1:]
    if parsed.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed.path == '/watch':
            return parse_qs(parsed.query)['v'][0]
    return None

def get_video_info(video_id):
    """Get video information including transcript"""
    api_key = os.environ.get('YOUTUBE_API_KEY')
    if not api_key:
        raise Exception("YouTube APIキーが設定されていません")
    
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={api_key}"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("YouTube APIの呼び出しに失敗しました")
    
    data = response.json()
    if not data['items']:
        raise Exception("動画が見つかりません")
    
    video_data = data['items'][0]['snippet']
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ja', 'en'])
        transcript_text = ' '.join([entry['text'] for entry in transcript])
    except Exception:
        raise Exception("字幕を取得できませんでした")
    
    return {
        'id': video_id,
        'title': video_data['title'],
        'thumbnail': video_data['thumbnails']['high']['url'],
        'transcript': transcript_text
    }
