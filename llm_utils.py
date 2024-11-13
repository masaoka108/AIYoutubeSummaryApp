from transformers import pipeline
import threading
import time
import logging
import re
import unicodedata

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
    
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    return text

def split_into_sentences(text):
    """Split text into sentences with proper boundary detection"""
    # Pattern for sentence boundaries including multiple spaces
    pattern = r'([。．.!?！？\n]|\s{2,})'
    sentences = []
    
    # Split and combine sentences with their punctuation
    parts = re.split(pattern, text)
    
    i = 0
    while i < len(parts):
        current = parts[i].strip()
        # Get punctuation/boundary if available
        boundary = parts[i + 1] if i + 1 < len(parts) else ""
        
        if current or boundary:  # Only add non-empty sentences
            sentences.append(current + boundary)
        
        i += 2  # Skip the boundary in next iteration
    
    return [s for s in sentences if s.strip()]  # Filter out empty sentences

def create_chunks(text, initial_chunk_size=512, min_chunk_size=50):
    """Create properly sized chunks of text with improved Japanese support"""
    if not text:
        raise ValueError("入力テキストが空です")
    
    # Clean and validate text
    text = clean_text(text)
    is_japanese = is_japanese_text(text)
    logger.info(f"Text contains Japanese characters: {is_japanese}")
    
    # Get text length considering Japanese characters
    text_length = get_text_length(text)
    logger.info(f"Total text length (weighted): {text_length}")
    
    if text_length <= min_chunk_size:
        logger.info(f"Text length ({text_length}) is smaller than minimum chunk size, returning as single chunk")
        return [text]
    
    # Split text into sentences
    sentences = split_into_sentences(text)
    logger.info(f"Split text into {len(sentences)} sentences")
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = get_text_length(sentence)
        logger.debug(f"Processing sentence with length {sentence_length}: {sentence[:50]}...")
        
        # If adding this sentence would exceed chunk size and we have enough content
        if current_length + sentence_length > initial_chunk_size and current_length >= min_chunk_size:
            chunk_text = ''.join(current_chunk)
            chunk_length = get_text_length(chunk_text)
            logger.info(f"Created chunk with length {chunk_length}")
            if chunk_length >= min_chunk_size:
                chunks.append(chunk_text)
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add the last chunk if it meets minimum size
    if current_chunk:
        chunk_text = ''.join(current_chunk)
        chunk_length = get_text_length(chunk_text)
        if chunk_length >= min_chunk_size:
            logger.info(f"Created final chunk with length {chunk_length}")
            chunks.append(chunk_text)
    
    # Validate chunks
    if not chunks:
        raise ValueError("テキストを適切なチャンクに分割できませんでした")
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def validate_chunk(chunk, index, total_chunks):
    """Validate chunk before processing"""
    if not chunk or not chunk.strip():
        raise ValueError(f"チャンク {index + 1}/{total_chunks} が空です")
    
    chunk_length = get_text_length(chunk)
    logger.info(f"Validating chunk {index + 1}/{total_chunks} (length: {chunk_length})")
    
    if chunk_length < 10:  # Minimum viable chunk size
        raise ValueError(f"チャンク {index + 1}/{total_chunks} が短すぎます")
    
    # Log first few characters for debugging
    preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
    logger.debug(f"Chunk {index + 1} preview: {preview}")
    
    return True

def truncate_to_size(text, target_size):
    """Truncate text to target size while respecting sentence boundaries"""
    if not text:
        return text
        
    sentences = split_into_sentences(text)
    result = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = get_text_length(sentence)
        if current_length + sentence_length > target_size:
            break
        result.append(sentence)
        current_length += sentence_length
    
    truncated = ''.join(result)
    if not truncated:
        raise ValueError("テキストを適切なサイズに調整できませんでした")
    
    return truncated

def summarize_chunk_with_retry(chunk, index, total_chunks, max_length, min_length):
    """Summarize a single chunk with improved error handling"""
    chunk_sizes = [512, 256, 128]  # Gradually reduce chunk size on failure
    model = get_summarizer()
    
    if not chunk or not chunk.strip():
        raise ValueError(f"チャンク {index + 1} が空です")
    
    for chunk_size in chunk_sizes:
        try:
            # Validate and truncate chunk if needed
            current_length = get_text_length(chunk)
            logger.info(f"Processing chunk {index + 1}/{total_chunks} (current length: {current_length})")
            
            if current_length > chunk_size:
                logger.info(f"Truncating chunk {index + 1} to size {chunk_size}")
                try:
                    chunk = truncate_to_size(chunk, chunk_size)
                    current_length = get_text_length(chunk)
                    logger.info(f"Truncated chunk length: {current_length}")
                except ValueError as e:
                    logger.warning(f"Failed to truncate chunk: {str(e)}")
                    continue
            
            # Validate chunk after truncation
            validate_chunk(chunk, index, total_chunks)
            
            # Attempt summarization
            logger.info(f"Attempting summarization with size {chunk_size}")
            summary = model(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            
            if not summary or not isinstance(summary, list) or len(summary) == 0:
                raise ValueError(f"チャンク {index + 1} の要約結果が無効です")
            
            logger.info(f"Successfully summarized chunk {index + 1}/{total_chunks}")
            return summary[0]['summary_text']
            
        except Exception as e:
            logger.warning(f"Failed to summarize chunk {index + 1} with size {chunk_size}: {str(e)}")
            if chunk_size == chunk_sizes[-1]:  # If this was the smallest chunk size
                raise ValueError(f"チャンク {index + 1} の要約に失敗しました: {str(e)}")
            continue

def summarize_text(text, max_length=130, min_length=30):
    """Summarize the given text with improved Japanese support"""
    try:
        if not text:
            raise ValueError("入力テキストが空です")
        
        text_length = get_text_length(text)
        logger.info(f"Starting summarization of text (length: {text_length})")
        
        # Create properly sized chunks
        chunks = create_chunks(text)
        logger.info(f"Created {len(chunks)} chunks for summarization")
        
        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                summary = summarize_chunk_with_retry(chunk, i, len(chunks), max_length, min_length)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Failed to summarize chunk {i + 1}: {str(e)}")
                raise
        
        final_summary = ' '.join(summaries)
        logger.info("Summarization completed successfully")
        return final_summary
        
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        raise Exception(f"テキストの要約に失敗しました: {str(e)}")

def answer_question(question, context):
    """Answer a question based on the context"""
    if not question or not context:
        raise ValueError("質問とコンテキストは空であってはいけません")
    
    try:
        model = get_qa_model()
        result = model(
            question=question,
            context=context,
            max_answer_length=100
        )
        return result['answer']
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        raise Exception(f"質問応答に失敗しました: {str(e)}")

# Start loading models in background
threading.Thread(target=load_models, daemon=True).start()
