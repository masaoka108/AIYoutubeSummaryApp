import re
import threading
import unicodedata
import logging
import time
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and progress tracking
summarizer = None
qa_model = None
summarization_progress = {"current": 0, "total": 0, "status": "idle"}
processing_lock = threading.Lock()
models_loaded = threading.Event()

def get_progress():
    """Get current summarization progress"""
    with processing_lock:
        return {
            "current": summarization_progress["current"],
            "total": summarization_progress["total"],
            "status": summarization_progress["status"],
            "percentage": int((summarization_progress["current"] / max(summarization_progress["total"], 1)) * 100)
        }

def update_progress(current, total, status="processing"):
    """Update summarization progress"""
    with processing_lock:
        summarization_progress["current"] = current
        summarization_progress["total"] = total
        summarization_progress["status"] = status
        logger.info(f"Progress: {current}/{total} - {status}")

def validate_japanese_output(text):
    """Validate if the text contains Japanese characters"""
    if not any(unicodedata.name(char).startswith('CJK UNIFIED') or 
               unicodedata.name(char).startswith('HIRAGANA') or 
               unicodedata.name(char).startswith('KATAKANA') 
               for char in text):
        raise ValueError("生成されたテキストに日本語が含まれていません")
    return text

def is_japanese_text(text):
    """Check if text contains Japanese characters"""
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
    return bool(japanese_pattern.search(text))

def load_models_async():
    """Load models asynchronously"""
    global summarizer, qa_model
    try:
        logger.info("Loading summarization model...")
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device="cpu"
        )
        
        logger.info("Loading QA model...")
        qa_model = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device="cpu"
        )
        
        models_loaded.set()
        logger.info("Models loaded successfully")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

def get_model():
    """Get or initialize summarizer"""
    global summarizer
    if summarizer is None:
        if not models_loaded.is_set():
            load_models_async()
            models_loaded.wait(timeout=30)  # Wait up to 30 seconds for models to load
    return summarizer

def get_qa_model():
    """Get or initialize QA model"""
    global qa_model
    if qa_model is None:
        if not models_loaded.is_set():
            load_models_async()
            models_loaded.wait(timeout=30)  # Wait up to 30 seconds for models to load
    return qa_model

def clean_text(text):
    """Clean and validate text for summarization"""
    if not isinstance(text, str):
        raise ValueError("入力テキストは文字列である必要があります")
    
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[｜│\s\u3000]+', ' ', text)
    text = re.sub(r'([。．.!?！？])\s*', r'\1\n', text)
    return text.strip()

def create_chunks(text, max_chunk_size=128, max_chunks=2):
    """Create properly sized chunks with Japanese text handling"""
    if not text:
        raise ValueError("入力テキストが空です")
    
    if not is_japanese_text(text):
        raise ValueError("日本語のテキストが含まれていません")
    
    text = clean_text(text)
    sentences = text.split('\n')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence_length = len(sentence)
        
        if current_length + sentence_length > max_chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    valid_chunks = [chunk for chunk in chunks if chunk.strip()]
    if len(valid_chunks) > max_chunks:
        logger.warning(f"チャンク数を {len(valid_chunks)} から {max_chunks} に削減します")
        valid_chunks = valid_chunks[:max_chunks]
    
    if not valid_chunks:
        raise ValueError("有効なチャンクを生成できませんでした")
    
    return valid_chunks

def summarize_chunk(chunk, index, total_chunks):
    try:
        if not chunk.strip():
            return ""
            
        model = get_model()
        
        # Add try-except for model call
        try:
            outputs = model(
                chunk,
                max_length=150,
                min_length=30,
                num_beams=2,
                do_sample=False,
                early_stopping=True
            )
            
            # Handle different output formats
            if isinstance(outputs, dict):
                summary = outputs.get('summary_text', '')
            elif isinstance(outputs, list) and outputs:
                summary = outputs[0].get('summary_text', '')
            else:
                raise ValueError("無効なモデル出力形式です")
                
            summary = summary.strip()
            if not summary:
                raise ValueError("空の要約が生成されました")
                
            return summary
            
        except Exception as e:
            logger.error(f"Model inference error: {str(e)}")
            raise ValueError(f"モデル推論エラー: {str(e)}")
            
    except Exception as e:
        logger.error(f"チャンク {index + 1} の要約に失敗: {str(e)}")
        raise

def summarize_text(text):
    """Summarize text with improved Japanese handling"""
    try:
        if not text:
            raise ValueError("入力テキストが空です")
        
        update_progress(0, 100, "テキストを準備中...")
        chunks = create_chunks(text, max_chunk_size=128, max_chunks=2)
        total_chunks = len(chunks)
        
        update_progress(0, total_chunks, "要約処理を開始します...")
        summaries = []
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            summary = summarize_chunk(chunk, i, total_chunks)
            if summary:
                summaries.append(summary)
            update_progress(i + 1, total_chunks)
        
        if not summaries:
            raise ValueError("要約を生成できませんでした")
            
        return ' '.join(summaries)
        
    except Exception as e:
        logger.error(f"要約処理に失敗: {str(e)}")
        raise

def answer_question(question, context):
    try:
        model = get_qa_model()
        answer = model(
            question=question,
            context=context
        )["answer"]
        
        return answer.strip()
        
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        raise

# Initialize models loading on import
load_models_async()
