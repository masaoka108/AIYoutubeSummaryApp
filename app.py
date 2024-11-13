import os
from flask import Flask, render_template, request, jsonify, flash
from flask_wtf.csrf import CSRFProtect
from youtube_utils import get_video_info, extract_video_id
from llm_utils import summarize_text, answer_question, get_progress
import threading
import queue
import time
from models import db, Summary, Question

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "dev_key_123"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///youtube_summary.db"
csrf = CSRFProtect(app)
db.init_app(app)

# Global queue for processing tasks
processing_queue = queue.Queue()
processing_results = {}

def process_video_summary(video_id, video_info):
    """Background task for processing video summary"""
    try:
        summary = summarize_text(video_info['transcript'])
        
        with app.app_context():
            new_summary = Summary(video_id=video_id, summary=summary)
            db.session.add(new_summary)
            db.session.commit()
        
        processing_results[video_id] = {
            'status': 'completed',
            'data': {
                'video': video_info,
                'summary': summary
            }
        }
    except Exception as e:
        processing_results[video_id] = {
            'status': 'error',
            'error': str(e)
        }

def process_queue():
    """Process tasks in the background queue"""
    while True:
        try:
            video_id, video_info = processing_queue.get(timeout=1)
            process_video_summary(video_id, video_info)
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Background processing error: {str(e)}")

# Start background processing thread
background_thread = threading.Thread(target=process_queue, daemon=True)
background_thread.start()

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
        processing_results[video_id] = {'status': 'processing'}
        processing_queue.put((video_id, video_info))
        
        return render_template('processing.html', 
                             video=video_info,
                             video_id=video_id)
    except Exception as e:
        flash(f'エラーが発生しました: {str(e)}', 'error')
        return render_template('index.html')

@app.route('/progress/<video_id>', methods=['GET'])
def check_progress(video_id):
    """Check the progress of video processing"""
    if video_id not in processing_results:
        return jsonify({'status': 'not_found'}), 404
        
    result = processing_results[video_id]
    if result['status'] == 'processing':
        progress = get_progress()
        return jsonify({
            'status': 'processing',
            'progress': progress
        })
    elif result['status'] == 'completed':
        return jsonify({
            'status': 'completed',
            'redirect': f'/result/{video_id}'
        })
    else:
        return jsonify({
            'status': 'error',
            'error': result.get('error', '不明なエラーが発生しました')
        })

@app.route('/result/<video_id>', methods=['GET'])
def show_result(video_id):
    """Show the processing result"""
    if video_id not in processing_results:
        flash('処理結果が見つかりません', 'error')
        return render_template('index.html')
        
    result = processing_results[video_id]
    if result['status'] != 'completed':
        return render_template('processing.html',
                             video=result.get('data', {}).get('video', {}),
                             video_id=video_id)
    
    return render_template('result.html',
                         video=result['data']['video'],
                         summary=result['data']['summary'])

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')
    video_id = request.form.get('video_id')
    
    if not question or not video_id:
        return jsonify({'error': '質問とビデオIDが必要です'}), 400
    
    try:
        video_info = get_video_info(video_id)
        answer = answer_question(question, video_info['transcript'])
        
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
    db.create_all()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
