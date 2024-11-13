import re
import threading
import unicodedata
import logging
import time

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
        import ollama
        
        # Initialize Ollama models
        summarizer = ollama.Client()
        qa_model = summarizer  # Use same client for both
        
        # Test model availability
        response = summarizer.chat(
            model="llama2-japanese",
            messages=[{"role": "user", "content": "テスト"}]
        )
        
        if response:
            models_loaded.set()
            logger.info("Ollama models loaded successfully")
            
    except Exception as e:
        logger.error(f"Error loading Ollama models: {str(e)}")

def get_model():
    """Get or initialize model"""
    global summarizer
    if summarizer is None:
        load_models_async()
        models_loaded.wait(timeout=30)  # Wait up to 30 seconds for models to load
    return summarizer

def clean_text(text):
    """Clean and validate text for summarization"""
    if not isinstance(text, str):
        raise ValueError("入力テキストは文字列である必要があります")
    
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[｜│\s\u3000]+', ' ', text)
    text = re.sub(r'([。．.!?！？])\s*', r'\1\n', text)
    return text.strip()

def create_chunks(text, max_chunk_size=512, max_chunks=3):
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
    """Summarize chunk with Japanese language support"""
    try:
        model = get_model()
        response = model.chat(
            model="llama2-japanese",
            messages=[
                {"role": "system", "content": "あなたは要約を生成する日本語のAIアシスタントです。"},
                {"role": "user", "content": f"以下の文章を要約してください：{chunk}"}
            ]
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Chunk {index + 1} summarization failed: {str(e)}")
        raise

def summarize_text(text):
    """Summarize text with improved Japanese handling"""
    try:
        if not text:
            raise ValueError("入力テキストが空です")
        
        update_progress(0, 100, "テキストを準備中...")
        chunks = create_chunks(text, max_chunk_size=512, max_chunks=3)
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
    """Answer a question based on the context"""
    if not question or not context:
        raise ValueError("質問とコンテキストは空であってはいけません")
    
    try:
        model = get_model()
        response = model.chat(
            model="llama2-japanese",
            messages=[
                {"role": "system", "content": "あなたは質問に回答する日本語のAIアシスタントです。"},
                {"role": "user", "content": f"以下のコンテキストに基づいて質問に答えてください:\nコンテキスト: {context}\n質問: {question}"}
            ]
        )
        
        if not response or 'message' not in response or 'content' not in response['message']:
            raise ValueError("無効な応答結果です")
        
        return response['message']['content']
        
    except Exception as e:
        logger.error(f"質問応答に失敗: {str(e)}")
        raise Exception(f"質問応答に失敗しました: {str(e)}")

# Initialize models loading on import
load_models_async()
