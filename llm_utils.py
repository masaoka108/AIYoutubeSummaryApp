import re
import threading
import unicodedata
import logging
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for lazy loading
summarizer = None
qa_model = None
model_lock = threading.Lock()

def load_models():
    """Load models in a separate thread"""
    global summarizer, qa_model
    with model_lock:
        if summarizer is None:
            logger.info("Loading summarization model...")
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        if qa_model is None:
            logger.info("Loading QA model...")
            qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

def get_summarizer():
    """Get or initialize summarizer"""
    if summarizer is None:
        load_models()
    return summarizer

def get_qa_model():
    """Get or initialize QA model"""
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
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_sentences(text):
    """Split text into sentences with proper Japanese boundary detection"""
    if not text:
        return []

    # Pattern for Japanese and English sentence boundaries including multiple spaces
    pattern = r'([。．.!?！？\n]|\s{2,})'
    
    try:
        # Split text into sentences while preserving boundaries
        parts = re.split(pattern, text)
        sentences = []
        
        i = 0
        while i < len(parts):
            current = parts[i].strip()
            boundary = parts[i + 1] if i + 1 < len(parts) else ""
            
            if current or boundary:
                sentence = (current + boundary).strip()
                if sentence:  # Only add non-empty sentences
                    sentences.append(sentence)
            i += 2
        
        if not sentences:
            # If no valid sentences found, treat the whole text as one sentence
            return [text.strip()] if text.strip() else []
            
        return sentences
        
    except Exception as e:
        logger.warning(f"Failed to split text into sentences: {str(e)}")
        # Return original text as single sentence if splitting fails
        return [text.strip()] if text.strip() else []

def truncate_to_size(text, target_size):
    """Truncate text to target size with improved boundary detection"""
    if not text:
        return text
        
    current_length = get_text_length(text)
    if current_length <= target_size:
        return text
        
    try:
        sentences = split_into_sentences(text)
        result = []
        current_length = 0
        
        # Try to include complete sentences
        for sentence in sentences:
            sentence_length = get_text_length(sentence)
            if current_length + sentence_length <= target_size:
                result.append(sentence)
                current_length += sentence_length
            else:
                break
        
        if result:
            return ''.join(result)
            
        # If no complete sentence fits, take the first sentence and truncate it
        first_sentence = sentences[0] if sentences else text
        truncated = ''
        current_length = 0
        
        for char in first_sentence:
            char_length = 2 if unicodedata.east_asian_width(char) in ['F', 'W'] else 1
            if current_length + char_length > target_size:
                break
            truncated += char
            current_length += char_length
            
        return truncated if truncated else text[:target_size]
        
    except Exception as e:
        logger.warning(f"Failed to truncate text properly: {str(e)}")
        # Fallback to simple truncation
        return text[:target_size]

def create_chunks(text, max_chunk_size=512, min_chunk_size=50):
    """Create properly sized chunks with strict size limits"""
    if not text:
        raise ValueError("入力テキストが空です")
    
    text = clean_text(text)
    text_length = get_text_length(text)
    
    if text_length <= min_chunk_size:
        return [text]
    
    chunks = []
    sentences = split_into_sentences(text)
    
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = get_text_length(sentence)
        
        # If adding this sentence would exceed max size
        if current_length + sentence_length > max_chunk_size:
            if current_chunk:
                # Join and validate current chunk
                chunk_text = ''.join(current_chunk)
                chunk_length = get_text_length(chunk_text)
                
                if chunk_length > max_chunk_size:
                    # Truncate if chunk is still too large
                    chunk_text = truncate_to_size(chunk_text, max_chunk_size)
                
                if chunk_text:
                    chunks.append(chunk_text)
                
                # Start new chunk with current sentence
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                # If single sentence is too long, truncate it
                truncated = truncate_to_size(sentence, max_chunk_size)
                if truncated:
                    chunks.append(truncated)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Handle the last chunk
    if current_chunk:
        chunk_text = ''.join(current_chunk)
        chunk_length = get_text_length(chunk_text)
        
        if chunk_length > max_chunk_size:
            chunk_text = truncate_to_size(chunk_text, max_chunk_size)
            
        if chunk_text:
            chunks.append(chunk_text)
    
    # Ensure we have at least one chunk
    if not chunks:
        # If no valid chunks created, create one from truncated original text
        return [truncate_to_size(text, max_chunk_size)]
    
    return chunks

def summarize_chunk_with_retry(chunk, index, total_chunks, max_length, min_length):
    """Summarize chunk with improved error handling and retries"""
    if not chunk or not chunk.strip():
        raise ValueError(f"チャンク {index + 1} が空です")
    
    chunk_sizes = [512, 256, 128]  # Gradually reduce chunk size on failure
    model = get_summarizer()
    
    for chunk_size in chunk_sizes:
        try:
            # Validate and truncate chunk
            current_length = get_text_length(chunk)
            logger.info(f"Processing chunk {index + 1}/{total_chunks} (length: {current_length})")
            
            if current_length > chunk_size:
                chunk = truncate_to_size(chunk, chunk_size)
                if not chunk:
                    logger.warning(f"Failed to truncate chunk {index + 1}")
                    continue
                
                current_length = get_text_length(chunk)
                logger.info(f"Truncated chunk {index + 1} to length: {current_length}")
            
            # Additional validation
            if current_length < min_length:
                logger.warning(f"Chunk {index + 1} is too short after truncation")
                continue
            
            # Attempt summarization with error handling
            try:
                summary = model(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                
                if summary and isinstance(summary, list) and len(summary) > 0:
                    summary_text = summary[0].get('summary_text', '').strip()
                    if summary_text:
                        logger.info(f"Successfully summarized chunk {index + 1}/{total_chunks}")
                        return summary_text
            except IndexError as e:
                logger.warning(f"Index error in chunk {index + 1}: {str(e)}")
                continue
            except Exception as e:
                logger.warning(f"Summarization error in chunk {index + 1}: {str(e)}")
                continue
            
            logger.warning(f"Invalid summary result for chunk {index + 1}")
            
        except Exception as e:
            logger.warning(f"Failed to process chunk {index + 1} with size {chunk_size}: {str(e)}")
            if chunk_size == chunk_sizes[-1]:
                raise ValueError(f"チャンク {index + 1} の要約に失敗しました: {str(e)}")
            continue
    
    raise ValueError(f"チャンク {index + 1} の要約に失敗しました")

def summarize_text(text, max_length=130, min_length=30):
    """Summarize text with improved Japanese support and error handling"""
    try:
        if not text:
            raise ValueError("入力テキストが空です")
        
        text = clean_text(text)
        chunks = create_chunks(text)
        logger.info(f"Created {len(chunks)} chunks for summarization")
        
        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                summary = summarize_chunk_with_retry(chunk, i, len(chunks), max_length, min_length)
                if summary:
                    summaries.append(summary)
            except Exception as e:
                logger.error(f"Failed to summarize chunk {i + 1}: {str(e)}")
                raise
        
        if not summaries:
            raise ValueError("要約を生成できませんでした")
        
        return ' '.join(summaries)
        
    except Exception as e:
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
            inputs={
                'question': question,
                'context': context
            },
            max_answer_length=100
        )
        
        if not result or not isinstance(result, dict) or 'answer' not in result:
            raise ValueError("無効な応答結果です")
            
        return result['answer']
        
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        raise Exception(f"質問応答に失敗しました: {str(e)}")

# Start loading models in background
threading.Thread(target=load_models, daemon=True).start()
