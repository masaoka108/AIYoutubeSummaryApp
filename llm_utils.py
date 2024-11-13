import re
import threading
import unicodedata
import logging
import time
import google.generativeai as genai
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and progress tracking
client = None
models_loaded = threading.Event()
processing_lock = threading.Lock()
summarization_progress = {"current": 0, "total": 0, "status": "idle"}

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

def is_japanese_text(text):
    """Check if text contains Japanese characters"""
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
    return bool(japanese_pattern.search(text))

def validate_japanese_output(text):
    """Validate if the text contains Japanese characters when required"""
    if not any(unicodedata.name(char).startswith('CJK UNIFIED') or 
               unicodedata.name(char).startswith('HIRAGANA') or 
               unicodedata.name(char).startswith('KATAKANA') 
               for char in text):
        raise ValueError("Generated text is not in Japanese")
    return text

def load_models_async():
    """Initialize Gemini model"""
    global client
    try:
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        client = genai.GenerativeModel('gemini-pro')
        models_loaded.set()
        logger.info("Gemini model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Gemini model: {str(e)}")
        raise

def get_model():
    """Get or initialize Gemini model"""
    global client
    if client is None:
        if not models_loaded.is_set():
            load_models_async()
            models_loaded.wait(timeout=30)
    return client

def get_qa_model():
    """Get or initialize QA model (uses same model)"""
    return get_model()

def clean_text(text):
    """Clean and validate text for summarization"""
    if not isinstance(text, str):
        raise ValueError("入力テキストは文字列である必要があります")
    
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[｜│\s\u3000]+', ' ', text)
    text = re.sub(r'([。．.!?！？])\s*', r'\1\n', text)
    return text.strip()

def create_chunks(text, max_chunk_size=1500, max_chunks=3):
    """Create properly sized chunks with language detection"""
    if not text:
        raise ValueError("入力テキストが空です")
    
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
        model = get_model()
        if not chunk.strip():
            raise ValueError("空のチャンクは処理できません")
        
        # For English content, explicitly request Japanese output
        is_jp = is_japanese_text(chunk)
        if not is_jp:
            prompt = f'''以下の英語の文章を日本語で要約してください。重要なポイントを3-5つにまとめてください：

{chunk}

以下のフォーマットで出力してください：

# 要約

- 重要ポイント：[具体的な内容を日本語で記載]'''
        else:
            prompt = f'''以下の文章を要約してください。重要なポイントを3-5つにまとめてください：

{chunk}

以下のフォーマットで出力してください：

# 要約

- 重要ポイント：[具体的な内容]'''

        response = model.generate_content(prompt)
        summary = response.text.strip()
        
        # Validate summary content
        if not summary or "文章の内容が明確に示されていない" in summary or "文章がない" in summary:
            raise ValueError("有効な要約を生成できませんでした")
            
        if not is_jp:
            validate_japanese_output(summary)
            
        return summary
        
    except Exception as e:
        logger.error(f"Chunk {index + 1} summarization failed: {str(e)}")
        raise

def summarize_text(text):
    """Summarize text with improved language handling"""
    start_time = time.time()
    max_processing_time = 25  # seconds
    
    try:
        if not text:
            raise ValueError("入力テキストが空です")
            
        update_progress(0, 100, "テキストを準備中...")
        chunks = create_chunks(text)
        total_chunks = len(chunks)
        
        update_progress(0, total_chunks, "要約処理を開始します...")
        summaries = []
        
        for i, chunk in enumerate(chunks):
            # Check total processing time
            if time.time() - start_time > max_processing_time:
                raise TimeoutError("要約処理の制限時間を超えました")
                
            if not chunk.strip():
                continue
                
            summary = summarize_chunk(chunk, i, total_chunks)
            if summary:
                summaries.append(summary)
            update_progress(i + 1, total_chunks)
            
        if not summaries:
            raise ValueError("要約を生成できませんでした")
            
        # Combine summaries into one
        combined_summaries = []
        for i, summary in enumerate(summaries):
            # Remove extra "# 要約" headers except for the first one
            if i == 0:
                combined_summaries.append(summary)
            else:
                # Remove header and keep only bullet points
                summary_lines = summary.split('\n')
                bullet_points = [line for line in summary_lines if line.strip() and not line.startswith('#')]
                combined_summaries.extend(bullet_points)
        
        return '\n'.join(combined_summaries)
        
    except TimeoutError as e:
        logger.error(f"Timeout error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"要約処理に失敗: {str(e)}")
        raise

def answer_question(question, context):
    try:
        if not question or not context:
            raise ValueError("質問とコンテキストは必須です")
            
        model = get_qa_model()
        if model is None:
            raise ValueError("QAモデルの初期化に失敗しました")
        
        # Always respond in Japanese
        prompt = f'''以下の文章に基づいて、日本語で質問に答えてください。

文章：
{context}

質問：
{question}'''
        
        response = model.generate_content(prompt)
        answer = response.text.strip()
        
        if not answer:
            raise ValueError("有効な回答を生成できませんでした")
            
        # Validate Japanese output
        validate_japanese_output(answer)
        return answer
        
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        raise

# Initialize models loading on import
load_models_async()
