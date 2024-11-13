import re
import threading
import unicodedata
import logging
import time
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and progress tracking
summarizer = None
qa_model = None
summarization_progress = {"current": 0, "total": 0, "status": "idle"}
processing_lock = threading.Lock()

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

@lru_cache(maxsize=1)
def load_models():
    """Load models with improved performance"""
    global summarizer, qa_model
    try:
        logger.info("Loading summarization model...")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        summarizer = pipeline("summarization", 
                            model=model, 
                            tokenizer=tokenizer,
                            device="cpu")  # Force CPU to avoid CUDA issues
        
        logger.info("Loading QA model...")
        qa_model = pipeline("question-answering", 
                          model="distilbert-base-cased-distilled-squad",
                          device="cpu")  # Force CPU to avoid CUDA issues
        
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def get_summarizer():
    """Get or initialize summarizer"""
    global summarizer
    if summarizer is None:
        load_models()
    return summarizer

def get_qa_model():
    """Get or initialize QA model"""
    global qa_model
    if qa_model is None:
        load_models()
    return qa_model

def get_text_length(text):
    """Get appropriate text length considering Japanese characters"""
    if not text:
        return 0
    length = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ['F', 'W']:
            length += 2
        else:
            length += 1
    return length

def clean_text(text):
    """Clean and validate text for summarization"""
    if not isinstance(text, str):
        raise ValueError("入力テキストは文字列である必要があります")
    
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[｜│\s\u3000]+', ' ', text)
    text = re.sub(r'([。．.!?！？])\s*', r'\1\n', text)
    return text.strip()

def create_chunks(text, max_chunk_size=1024, min_chunk_size=256, max_chunks=5):
    """Create properly sized chunks"""
    if not text:
        raise ValueError("入力テキストが空です")
    
    text = clean_text(text)
    chunks = text.split('\n')
    result_chunks = []
    current_chunk = []
    current_length = 0
    
    for chunk in chunks:
        chunk_length = get_text_length(chunk)
        if current_length + chunk_length <= max_chunk_size:
            current_chunk.append(chunk)
            current_length += chunk_length
        else:
            if current_chunk:
                result_chunks.append(' '.join(current_chunk))
            current_chunk = [chunk]
            current_length = chunk_length
    
    if current_chunk:
        result_chunks.append(' '.join(current_chunk))
    
    # Limit total chunks
    if len(result_chunks) > max_chunks:
        logger.warning(f"チャンク数を {len(result_chunks)} から {max_chunks} に削減します")
        result_chunks = result_chunks[:max_chunks]
    
    return result_chunks

def summarize_chunk(chunk, index, total_chunks):
    """Summarize a single chunk"""
    if not chunk.strip():
        return ""
        
    try:
        model = get_summarizer()
        inputs = {"text": chunk}
        
        summary = model(
            inputs=chunk,
            max_length=150,
            min_length=50,
            do_sample=False,
            num_beams=2,
            length_penalty=2.0,
            early_stopping=True
        )
        
        if summary and isinstance(summary, list) and len(summary) > 0:
            return summary[0].get('summary_text', '').strip()
        return ""
        
    except Exception as e:
        logger.error(f"チャンク {index + 1} の要約に失敗: {str(e)}")
        return ""

def summarize_text(text):
    """Summarize text with improved performance"""
    try:
        if not text:
            raise ValueError("入力テキストが空です")
        
        update_progress(0, 100, "テキストを準備中...")
        chunks = create_chunks(text)
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            raise ValueError("テキストを分割できませんでした")
        
        update_progress(0, total_chunks, "要約処理を開始します...")
        summaries = []
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            if time.time() - start_time > 20:
                raise TimeoutError("要約処理の制限時間を超えました")
            
            summary = summarize_chunk(chunk, i, total_chunks)
            if summary:
                summaries.append(summary)
            update_progress(i + 1, total_chunks, f"チャンク {i + 1}/{total_chunks} を処理中...")
        
        if not summaries:
            raise ValueError("要約を生成できませんでした")
        
        final_summary = ' '.join(summaries)
        update_progress(total_chunks, total_chunks, "要約が完了しました")
        return final_summary
        
    except Exception as e:
        update_progress(0, 0, f"エラー: {str(e)}")
        logger.error(f"要約処理に失敗: {str(e)}")
        raise

def answer_question(question, context):
    """Answer a question based on the context"""
    if not question or not context:
        raise ValueError("質問とコンテキストは空であってはいけません")
    
    try:
        model = get_qa_model()
        if not model:
            raise ValueError("QAモデルの初期化に失敗しました")
        
        inputs = {
            "question": question,
            "context": context
        }
        
        result = model(
            question=question,
            context=context,
            max_answer_length=100,
            handle_impossible_answer=True
        )
        
        if not result or not isinstance(result, dict) or 'answer' not in result:
            raise ValueError("無効な応答結果です")
        
        return result['answer']
        
    except Exception as e:
        logger.error(f"質問応答に失敗: {str(e)}")
        raise Exception(f"質問応答に失敗しました: {str(e)}")

# Initialize models on import
load_models()