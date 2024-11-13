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
        
        summarizer = {
            "model": model,
            "tokenizer": tokenizer
        }
        
        logger.info("Loading QA model...")
        qa_model = pipeline("question-answering", 
                          model="distilbert-base-cased-distilled-squad",
                          device="cpu")
        
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

def is_japanese_text(text):
    """Check if text contains Japanese characters"""
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
    return bool(japanese_pattern.search(text))

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

def create_chunks(text, max_chunk_size=1024, min_chunk_size=256, max_chunks=3):
    """Create properly sized chunks with improved Japanese text handling"""
    if not text:
        raise ValueError("入力テキストが空です")
    
    text = clean_text(text)
    sentences = text.split('\n')
    
    if not sentences:
        raise ValueError("テキストを文章に分割できませんでした")
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence_length = get_text_length(sentence)
        
        if sentence_length > max_chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long sentence at logical Japanese boundaries
            particles = r'(では|には|から|まで|として|による|について|という|との|への)'
            sub_parts = re.split(particles, sentence)
            current_sub_part = []
            current_sub_length = 0
            
            for part in sub_parts:
                part_length = get_text_length(part)
                if current_sub_length + part_length <= max_chunk_size:
                    current_sub_part.append(part)
                    current_sub_length += part_length
                else:
                    if current_sub_part:
                        chunks.append(' '.join(current_sub_part))
                    current_sub_part = [part]
                    current_sub_length = part_length
            
            if current_sub_part:
                chunks.append(' '.join(current_sub_part))
            continue
        
        if current_length + sentence_length > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Validate chunk content and limit total chunks
    valid_chunks = [chunk for chunk in chunks if chunk.strip()]
    if len(valid_chunks) > max_chunks:
        logger.warning(f"チャンク数を {len(valid_chunks)} から {max_chunks} に削減します")
        valid_chunks = valid_chunks[:max_chunks]
    
    if not valid_chunks:
        raise ValueError("有効なチャンクを生成できませんでした")
    
    return valid_chunks

def summarize_chunk(chunk, index, total_chunks, start_time):
    """Summarize chunk with improved error handling and timeout"""
    if not chunk.strip():
        return ""
        
    try:
        # Check chunk timeout (8 seconds)
        if time.time() - start_time > 8:
            raise TimeoutError(f"チャンク {index + 1} の処理がタイムアウトしました")
            
        model_data = get_summarizer()
        if not model_data or not isinstance(model_data, dict):
            raise ValueError("サマライザーの初期化に失敗しました")
            
        # Add proper tokenization for Japanese text
        inputs = model_data["tokenizer"](chunk, truncation=True, max_length=1024, return_tensors="pt")
        
        # Generate summary with faster parameters
        summary_ids = model_data["model"].generate(
            inputs["input_ids"],
            max_length=100,
            min_length=30,
            length_penalty=1.5,
            num_beams=2,
            early_stopping=True
        )
        
        summary = model_data["tokenizer"].decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()
        
    except Exception as e:
        logger.error(f"チャンク {index + 1} の要約に失敗: {str(e)}")
        raise

def summarize_text(text):
    """Summarize text with improved error handling and retry mechanism"""
    try:
        if not text:
            raise ValueError("入力テキストが空です")
        
        update_progress(0, 100, "テキストを準備中...")
        
        # Validate input text
        if len(text.strip()) < 10:
            raise ValueError("テキストが短すぎます")
        
        start_time = time.time()
        max_retries = 2
        current_retry = 0
        
        while current_retry <= max_retries:
            try:
                # Adjust chunk size based on retry attempt
                max_chunk_size = 1024 >> current_retry  # Reduce size on each retry
                chunks = create_chunks(text, max_chunk_size=max_chunk_size)
                total_chunks = len(chunks)
                
                if total_chunks == 0:
                    raise ValueError("テキストを分割できませんでした")
                
                update_progress(0, total_chunks, "要約処理を開始します...")
                summaries = []
                chunk_start_time = time.time()
                
                for i, chunk in enumerate(chunks):
                    # Check total timeout (30 seconds)
                    if time.time() - start_time > 30:
                        raise TimeoutError("要約処理の制限時間を超えました")
                    
                    if not chunk.strip():
                        logger.warning(f"チャンク {i + 1} が空です。スキップします。")
                        continue
                    
                    summary = summarize_chunk(chunk, i, total_chunks, chunk_start_time)
                    if summary:
                        summaries.append(summary)
                    update_progress(i + 1, total_chunks, f"チャンク {i + 1}/{total_chunks} を処理中...")
                    chunk_start_time = time.time()  # Reset timer for next chunk
                
                if summaries:
                    final_summary = ' '.join(summaries)
                    update_progress(total_chunks, total_chunks, "要約が完了しました")
                    return final_summary
                else:
                    raise ValueError("要約を生成できませんでした")
                    
            except TimeoutError:
                if current_retry == max_retries:
                    raise
                current_retry += 1
                logger.warning(f"タイムアウトのため、より小さいチャンクサイズで再試行します ({current_retry}/{max_retries})")
                continue
            except Exception as e:
                raise
                
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
