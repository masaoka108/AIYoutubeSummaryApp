import os
from flask import Flask, render_template, request, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from youtube_utils import get_video_info, extract_video_id
from llm_utils import summarize_text, answer_question
from flask_wtf.csrf import CSRFProtect

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "dev_key_123"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///youtube_summary.db"
csrf = CSRFProtect(app)
db.init_app(app)

@app.route('/', methods=['GET'])
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
        video_info = get_video_info(video_id)
        summary = summarize_text(video_info['transcript'])
        
        from models import Summary
        new_summary = Summary(video_id=video_id, summary=summary)
        db.session.add(new_summary)
        db.session.commit()
        
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
    
    if not question or not video_id:
        return jsonify({'error': '質問とビデオIDが必要です'}), 400
    
    try:
        video_info = get_video_info(video_id)
        answer = answer_question(question, video_info['transcript'])
        
        from models import Question
        new_question = Question(video_id=video_id, question=question, answer=answer)
        db.session.add(new_question)
        db.session.commit()
        
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.errorhandler(405)
def method_not_allowed(e):
    if request.path == '/summarize':
        flash('無効なリクエストメソッドです。フォームから送信してください。', 'error')
        return render_template('index.html'), 405
    return jsonify({'error': '無効なリクエストメソッドです'}), 405

with app.app_context():
    import models
    db.create_all()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
