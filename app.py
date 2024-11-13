import os
from flask import Flask, render_template, request, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from youtube_utils import get_video_info, extract_video_id
from llm_utils import summarize_text, answer_question, model_lock

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "dev_key_123"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///youtube_summary.db"
db.init_app(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    youtube_url = request.form.get('youtube_url')
    if not youtube_url:
        flash('URLを入力してください', 'error')
        return render_template('index.html')
    
    video_id = extract_video_id(youtube_url)
    if not video_id:
        flash('無効なYouTube URLです', 'error')
        return render_template('index.html')

    try:
        if not model_lock.acquire(blocking=False):
            flash('モデルを読み込み中です。しばらくお待ちください。', 'warning')
            return render_template('index.html')
        model_lock.release()
        
        video_info = get_video_info(video_id)
        summary = summarize_text(video_info['transcript'])
        return render_template('result.html', 
                             video=video_info,
                             summary=summary)
    except Exception as e:
        flash(f'エラーが発生しました: {str(e)}', 'error')
        return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')
    video_id = request.form.get('video_id')
    
    try:
        if not model_lock.acquire(blocking=False):
            return jsonify({'error': 'モデルを読み込み中です。しばらくお待ちください。'}), 503
        model_lock.release()
        
        video_info = get_video_info(video_id)
        answer = answer_question(question, video_info['transcript'])
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

with app.app_context():
    import models
    db.create_all()
