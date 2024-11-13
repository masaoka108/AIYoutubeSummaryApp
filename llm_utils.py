import re
import threading
import unicodedata
import logging
import time
import torch
from transformers import pipeline, MBartTokenizer, MBartForConditionalGeneration
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

@lru_cache(maxsize=1)
def load_models():
    """Load models with Japanese support"""
    global summarizer, qa_model
    try:
        logger.info("Loading summarization model...")
        summarizer = pipeline(
            "summarization",
            model="facebook/mbart-large-cc25",
            tokenizer="facebook/mbart-large-cc25",
            device="cpu"
        )
        
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
    """Split text into sentences with Japanese-specific handling"""
    if not text:
        return []

    # Japanese-specific sentence boundary pattern
    pattern = r'([。．.!?！？\n]|[。．.!?！？]\s)'
    
    try:
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if re.search(pattern, char):
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
            
        return sentences
    except Exception as e:
        logger.warning(f"文章の分割に失敗: {str(e)}")
        return [text.strip()]

def create_chunks(text, max_chunk_size=1024, min_chunk_size=256, max_chunks=3):
    """Create properly sized chunks with Japanese text handling"""
    if not text:
        raise ValueError("入力テキストが空です")
    
    # Validate Japanese content
    if not is_japanese_text(text):
        raise ValueError("日本語のテキストが含まれていません")
    
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
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long sentences at Japanese-specific boundaries
            particles = r'(では|には|から|まで|として|による|について|という|との|への|もの|こと|など|まま|ため)'
            sub_parts = re.split(particles, sentence)
            current_sub_part = []
            current_sub_length = 0
            
            for part in sub_parts:
                if not part.strip():
                    continue
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

def summarize_chunk(chunk, index, total_chunks, start_time):
    """Summarize chunk with Japanese language support"""
    if not chunk.strip():
        return ""
        
    try:
        # Check chunk timeout (8 seconds)
        if time.time() - start_time > 8:
            raise TimeoutError(f"チャンク {index + 1} の処理がタイムアウトしました")
        
        model = get_summarizer()
        if not model:
            raise ValueError("サマライザーの初期化に失敗しました")
        
        # Set Japanese as target language
        inputs = {
            "text": chunk,
            "max_length": 100,
            "min_length": 30,
            "length_penalty": 1.5,
            "num_beams": 2,
            "early_stopping": True,
            "forced_bos_token_id": model.tokenizer.lang_code_to_id["ja_XX"]
        }
        
        summary = model(**inputs)[0]["summary_text"]
        
        # Validate Japanese output
        summary = validate_japanese_output(summary)
        return summary.strip()
        
    except Exception as e:
        logger.error(f"チャンク {index + 1} の要約に失敗: {str(e)}")
        raise

def summarize_text(text):
    """Summarize text with improved Japanese handling"""
    try:
        if not text:
            raise ValueError("入力テキストが空です")
        
        update_progress(0, 100, "テキストを準備中...")
        
        if len(text.strip()) < 10:
            raise ValueError("テキストが短すぎます")
        
        start_time = time.time()
        max_retries = 2
        current_retry = 0
        
        while current_retry <= max_retries:
            try:
                max_chunk_size = 1024 >> current_retry
                chunks = create_chunks(text, max_chunk_size=max_chunk_size)
                total_chunks = len(chunks)
                
                if total_chunks == 0:
                    raise ValueError("テキストを分割できませんでした")
                
                update_progress(0, total_chunks, "要約処理を開始します...")
                summaries = []
                chunk_start_time = time.time()
                
                for i, chunk in enumerate(chunks):
                    if time.time() - start_time > 30:
                        raise TimeoutError("要約処理の制限時間を超えました")
                    
                    if not chunk.strip():
                        logger.warning(f"チャンク {i + 1} が空です。スキップします。")
                        continue
                    
                    summary = summarize_chunk(chunk, i, total_chunks, chunk_start_time)
                    if summary:
                        summaries.append(summary)
                    update_progress(i + 1, total_chunks, f"チャンク {i + 1}/{total_chunks} を処理中...")
                    chunk_start_time = time.time()
                
                if summaries:
                    final_summary = ' '.join(summaries)
                    validate_japanese_output(final_summary)
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
