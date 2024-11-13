import re
import threading
import unicodedata
import logging
import time
from transformers import MBartForConditionalGeneration, MBartTokenizer, pipeline

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

def load_models():
    """Load models with Japanese summarization support"""
    global summarizer, qa_model
    try:
        logger.info("Loading summarization model...")
        summarizer = pipeline("summarization", model="facebook/mbart-large-cc25")
        
        logger.info("Loading QA model...")
        qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        
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
            length += 2  # Count Japanese characters as 2 units
        else:
            length += 1
    return length

def clean_text(text):
    """Clean and validate text for summarization"""
    if not isinstance(text, str):
        raise ValueError("入力テキストは文字列である必要があります")
    
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\s\u3000]+', ' ', text).strip()
    return text

def split_into_sentences(text):
    """Split text into sentences with improved Japanese boundary detection"""
    if not text:
        return []

    pattern = r'([。．.!?！？\n]|\s{2,}|(?<=」)(?![\s。．.!?！？])|(?<=）)(?![\s。．.!?！？]))'
    
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
                    sentences.append(sentence)
            i += 2
        
        return sentences if sentences else [text.strip()] if text.strip() else []
            
    except Exception as e:
        logger.warning(f"Failed to split text into sentences: {str(e)}")
        return [text.strip()] if text.strip() else []

def merge_short_chunks(chunks, max_chunk_size=512):
    """Merge short chunks to reduce total number of chunks"""
    if not chunks:
        return chunks
        
    merged = []
    current = chunks[0]
    current_length = get_text_length(current)
    
    for chunk in chunks[1:]:
        chunk_length = get_text_length(chunk)
        if current_length + chunk_length <= max_chunk_size:
            current += ' ' + chunk
            current_length += chunk_length
        else:
            merged.append(current)
            current = chunk
            current_length = chunk_length
    
    merged.append(current)
    return merged

def create_chunks(text, max_chunk_size=512, min_chunk_size=100, max_chunks=20):
    """Create properly sized chunks with improved Japanese text handling"""
    if not text:
        raise ValueError("入力テキストが空です")
    
    start_time = time.time()
    text = clean_text(text)
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Check chunking timeout
        if time.time() - start_time > 5:
            logger.warning("Chunking timeout reached")
            break
            
        sentence_length = get_text_length(sentence)
        
        if sentence_length > max_chunk_size:
            if current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long sentence at logical points
            sub_parts = re.split(r'(で|に|を|は|が|の|、|。)', sentence)
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
    
    # Merge short chunks and limit total chunks
    chunks = merge_short_chunks(chunks, max_chunk_size)
    if len(chunks) > max_chunks:
        logger.warning(f"Reducing chunks from {len(chunks)} to {max_chunks}")
        chunks = chunks[:max_chunks]
    
    return chunks if chunks else [text[:max_chunk_size]]

def summarize_chunk_with_retry(chunk, index, total_chunks, max_length=150, min_length=50):
    """Summarize chunk with improved Japanese text handling and timeout"""
    if not chunk or not chunk.strip():
        raise ValueError(f"チャンク {index + 1} が空です")
    
    chunk_sizes = [512, 256]  # Reduced retry attempts to 2
    model = get_summarizer()
    start_time = time.time()
    
    for chunk_size in chunk_sizes:
        try:
            # Check chunk timeout
            if time.time() - start_time > 8:
                raise TimeoutError(f"チャンク {index + 1} の処理がタイムアウトしました")
                
            current_length = get_text_length(chunk)
            logger.info(f"Processing chunk {index + 1}/{total_chunks} (length: {current_length})")
            
            if current_length > chunk_size:
                chunk = clean_text(chunk[:chunk_size])
                current_length = get_text_length(chunk)
                logger.info(f"Truncated chunk {index + 1} to length: {current_length}")
            
            try:
                summary = model(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    num_beams=2  # Faster beam search
                )
                
                if summary and isinstance(summary, list) and len(summary) > 0:
                    summary_text = summary[0].get('summary_text', '').strip()
                    if summary_text:
                        logger.info(f"Successfully summarized chunk {index + 1}/{total_chunks}")
                        return summary_text
                        
            except Exception as e:
                logger.warning(f"Summarization error in chunk {index + 1}: {str(e)}")
                if chunk_size == chunk_sizes[-1]:
                    raise
                continue
            
        except Exception as e:
            if isinstance(e, TimeoutError):
                raise
            logger.warning(f"Failed to process chunk {index + 1} with size {chunk_size}: {str(e)}")
            if chunk_size == chunk_sizes[-1]:
                raise ValueError(f"チャンク {index + 1} の要約に失敗しました: {str(e)}")
            continue
    
    raise ValueError(f"チャンク {index + 1} の要約に失敗しました")

def summarize_text(text):
    """Summarize text with improved Japanese support and progress tracking"""
    try:
        if not text:
            raise ValueError("入力テキストが空です")
        
        update_progress(0, 100, "準備中...")
        text = clean_text(text)
        chunks = create_chunks(text)  # Using default max_chunks=20
        total_chunks = len(chunks)
        logger.info(f"Created {total_chunks} chunks for summarization")
        
        update_progress(0, total_chunks, "要約処理中...")
        summaries = []
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            try:
                # Check total timeout
                if time.time() - start_time > 30:
                    raise TimeoutError("要約処理がタイムアウトしました")
                
                summary = summarize_chunk_with_retry(chunk, i, total_chunks)
                if summary:
                    summaries.append(summary)
                update_progress(i + 1, total_chunks, "要約処理中...")
                
            except Exception as e:
                logger.error(f"Failed to summarize chunk {i + 1}: {str(e)}")
                # Cleanup on error
                update_progress(0, 0, f"エラー: {str(e)}")
                raise
        
        if not summaries:
            raise ValueError("要約を生成できませんでした")
        
        final_summary = ' '.join(summaries)
        update_progress(total_chunks, total_chunks, "完了")
        return final_summary
        
    except Exception as e:
        # Ensure proper cleanup
        update_progress(0, 0, f"エラー: {str(e)}")
        logger.error(f"Summarization failed: {str(e)}")
        raise Exception(f"テキストの要約に失敗しました: {str(e)}")

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
            max_answer_length=100
        )
        
        if not result or not isinstance(result, dict) or 'answer' not in result:
            raise ValueError("無効な応答結果です")
            
        return result['answer']
        
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        raise Exception(f"質問応答に失敗しました: {str(e)}")

# Initialize models
load_models()
