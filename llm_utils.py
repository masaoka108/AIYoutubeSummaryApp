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

def split_into_sentences(text):
    """Split text into sentences with improved Japanese boundary detection"""
    if not text:
        return []

    # Enhanced pattern for Japanese text boundaries
    pattern = r'([。．.!?！？]|\n|(?<=」)(?![\s。．.!?！？])|(?<=）)(?![\s。．.!?！？])|(?<=、)\s+(?=[^、。．.!?！？]*[。．.!?！？]))'
    
    try:
        parts = re.split(pattern, text)
        sentences = []
        
        i = 0
        while i < len(parts):
            current = parts[i].strip()
            boundary = parts[i + 1] if i + 1 < len(parts) else ""
            
            if current or boundary:
                sentence = (current + boundary).strip()
                if sentence:
                    # Handle nested quotes and parentheses
                    if ('「' in sentence and '」' not in sentence) or \
                       ('（' in sentence and '）' not in sentence):
                        continue_idx = i + 2
                        while continue_idx < len(parts):
                            if '」' in parts[continue_idx] or '）' in parts[continue_idx]:
                                sentence += ''.join(parts[i+2:continue_idx+1])
                                i = continue_idx
                                break
                            continue_idx += 1
                    sentences.append(sentence)
            i += 2
        
        return sentences if sentences else [text.strip()]
            
    except Exception as e:
        logger.warning(f"Failed to split text into sentences: {str(e)}")
        return [text.strip()]

def create_chunks(text, max_chunk_size=512, min_chunk_size=100, max_chunks=5):
    """Create properly sized chunks with improved Japanese text handling"""
    if not text:
        raise ValueError("入力テキストが空です")
    
    text = clean_text(text)
    sentences = split_into_sentences(text)
    
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
            # If current chunk exists, add it to chunks
            if current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long sentence at logical Japanese boundaries
            particles = r'(では|には|から|まで|として|による|について|という|との|への|もの|こと|など|まま|ため)'
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
                        chunks.append(''.join(current_sub_part))
                    current_sub_part = [part]
                    current_sub_length = part_length
            
            if current_sub_part:
                chunks.append(''.join(current_sub_part))
            continue
        
        if current_length + sentence_length > max_chunk_size:
            chunks.append(''.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    # Validate chunk content and limit total chunks
    valid_chunks = [chunk for chunk in chunks if chunk.strip()]
    if len(valid_chunks) > max_chunks:
        logger.warning(f"チャンク数を {len(valid_chunks)} から {max_chunks} に削減します")
        valid_chunks = valid_chunks[:max_chunks]
    
    if not valid_chunks:
        raise ValueError("有効なチャンクを生成できませんでした")
    
    return valid_chunks

def summarize_chunk(chunk, index, total_chunks):
    """Summarize chunk with improved error handling"""
    if not chunk.strip():
        return ""
        
    try:
        model_data = get_summarizer()
        if not model_data or not isinstance(model_data, dict):
            raise ValueError("サマライザーの初期化に失敗しました")
            
        # Add proper tokenization for Japanese text
        inputs = model_data["tokenizer"](chunk, truncation=True, max_length=1024, return_tensors="pt")
        
        # Generate summary with proper error handling
        summary_ids = model_data["model"].generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        summary = model_data["tokenizer"].decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()
        
    except Exception as e:
        logger.error(f"チャンク {index + 1} の要約に失敗: {str(e)}")
        raise Exception(f"チャンク {index + 1} の要約に失敗しました: {str(e)}")

def summarize_text(text):
    """Summarize text with improved error handling"""
    try:
        if not text:
            raise ValueError("入力テキストが空です")
        
        update_progress(0, 100, "テキストを準備中...")
        
        # Validate input text
        if len(text.strip()) < 10:
            raise ValueError("テキストが短すぎます")
        
        chunks = create_chunks(text)
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            raise ValueError("テキストを分割できませんでした")
        
        update_progress(0, total_chunks, "要約処理を開始します...")
        summaries = []
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            try:
                if time.time() - start_time > 20:
                    raise TimeoutError("要約処理の制限時間を超えました")
                
                if not chunk.strip():
                    logger.warning(f"チャンク {i + 1} が空です。スキップします。")
                    continue
                
                summary = summarize_chunk(chunk, i, total_chunks)
                if summary:
                    summaries.append(summary)
                update_progress(i + 1, total_chunks, f"チャンク {i + 1}/{total_chunks} を処理中...")
                
            except Exception as e:
                logger.error(f"チャンク {i + 1} の処理中にエラー: {str(e)}")
                update_progress(0, 0, f"エラー: チャンク {i + 1} の処理に失敗しました")
                raise
        
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
