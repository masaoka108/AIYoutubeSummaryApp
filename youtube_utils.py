import os
import re
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
import requests
import logging
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from models import db, Transcript

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    parsed = urlparse(url)
    if parsed.hostname == 'youtu.be':
        return parsed.path[1:]
    if parsed.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed.path == '/watch':
            return parse_qs(parsed.query)['v'][0]
    return None

def get_cached_transcript(video_id):
    """Get transcript from cache if it exists"""
    cached = Transcript.query.filter_by(video_id=video_id).first()
    if cached:
        # Check if cache is less than 24 hours old
        if datetime.utcnow() - cached.created_at < timedelta(hours=24):
            logger.info(f"Using cached transcript for video {video_id}")
            return cached.transcript_text
        else:
            # Remove old cache
            db.session.delete(cached)
            db.session.commit()
    return None

def cache_transcript(video_id, transcript_text, language='ja'):
    """Cache transcript in database"""
    try:
        new_transcript = Transcript(
            video_id=video_id,
            transcript_text=transcript_text,
            language=language
        )
        db.session.add(new_transcript)
        db.session.commit()
        logger.info(f"Cached transcript for video {video_id}")
    except IntegrityError:
        db.session.rollback()
        logger.warning(f"Transcript for video {video_id} already exists")

def get_transcript(video_id):
    """Get transcript with improved language support and caching"""
    cached_transcript = get_cached_transcript(video_id)
    if cached_transcript:
        logger.info(f"Transcript content for video {video_id} (cached):")
        logger.info(cached_transcript[:500] + "..." if len(cached_transcript) > 500 else cached_transcript)
        return cached_transcript
        
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        language = None
        
        # First try to get original language transcript
        try:
            transcript = transcript_list.find_manually_created_transcript()
            language = transcript.language_code
        except NoTranscriptFound:
            pass
            
        # If no manual transcript, try English or Japanese
        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(['en', 'ja'])
                language = transcript.language_code
            except NoTranscriptFound:
                raise Exception("No suitable transcripts found")

        entries = transcript.fetch()
        transcript_text = ' '.join(entry['text'] for entry in entries)
        
        # Add debug logging
        logger.info(f"Transcript content for video {video_id}:")
        logger.info(transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text)
        
        # Cache the transcript
        cache_transcript(video_id, transcript_text, language)
        return transcript_text
            
    except VideoUnavailable:
        raise Exception("動画が非公開または削除されています")
    except TranscriptsDisabled:
        raise Exception("この動画では字幕が無効になっています")
    except Exception as e:
        logger.error(f"Transcript fetch error: {str(e)}")
        raise

def get_video_info(video_id):
    """Get video information including transcript"""
    api_key = os.environ.get('YOUTUBE_API_KEY')
    if not api_key:
        raise Exception("YouTube APIキーが設定されていません")
    
    # Get video metadata
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,status&id={video_id}&key={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        if not data['items']:
            raise Exception("動画が見つかりません")
        
        video_data = data['items'][0]
        
        # Check video privacy status
        if video_data['status']['privacyStatus'] == 'private':
            raise Exception("この動画は非公開です")
        
        # Get transcript
        transcript_text = get_transcript(video_id)
        
        return {
            'id': video_id,
            'title': video_data['snippet']['title'],
            'thumbnail': video_data['snippet']['thumbnails']['high']['url'],
            'transcript': transcript_text
        }
        
    except requests.exceptions.RequestException as e:
        if e.response is not None:
            if e.response.status_code == 403:
                raise Exception("YouTube APIのアクセス権限がありません")
            elif e.response.status_code == 404:
                raise Exception("動画が見つかりません")
            else:
                raise Exception(f"YouTube APIエラー: {e.response.status_code}")
        else:
            raise Exception("YouTube APIへの接続に失敗しました")
