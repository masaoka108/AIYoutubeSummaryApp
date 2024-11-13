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

def create_chunks(text, initial_chunk_size=512, min_chunk_size=50):
    """Create properly sized chunks of text with Japanese support"""
    if not text:
        raise ValueError("入力テキストが空です")
    
    # Clean the text first
    text = clean_text(text)
    is_japanese = is_japanese_text(text)
    logger.info(f"Text contains Japanese characters: {is_japanese}")
    
    # If text is shorter than minimum size, return as is
    text_length = get_text_length(text)
    if text_length <= min_chunk_size:
        logger.info(f"Text length ({text_length}) is smaller than minimum chunk size, returning as single chunk")
        return [text]
    
    # Japanese sentence boundary pattern
    sentence_pattern = r'([。．.!?！？\n])'
    
    chunks = []
    sentences = re.split(sentence_pattern, text)
    current_chunk = []
    current_length = 0
    
    i = 0
    while i < len(sentences):
        # Get current sentence and its punctuation if available
        current_sentence = sentences[i]
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
        full_sentence = current_sentence + punctuation
        
        sentence_length = get_text_length(full_sentence)
        
        # If adding this sentence would exceed chunk size and we have enough content
        if current_length + sentence_length > initial_chunk_size and current_length >= min_chunk_size:
            chunk_text = ''.join(current_chunk)
            logger.info(f"Created chunk with length: {get_text_length(chunk_text)}")
            chunks.append(chunk_text)
            current_chunk = [full_sentence]
            current_length = sentence_length
        else:
            current_chunk.append(full_sentence)
            current_length += sentence_length
        
        i += 2  # Skip the punctuation in the next iteration
    
    # Add the last chunk if it has content
    if current_chunk:
        chunk_text = ''.join(current_chunk)
        if get_text_length(chunk_text) >= min_chunk_size:
            logger.info(f"Created final chunk with length: {get_text_length(chunk_text)}")
            chunks.append(chunk_text)
    
    return chunks

def validate_chunk(chunk, index, total_chunks):
    """Validate chunk before processing"""
    if not chunk:
        raise ValueError(f"チャンク {index + 1}/{total_chunks} が空です")
    
    chunk_length = get_text_length(chunk)
    logger.info(f"Validating chunk {index + 1}/{total_chunks} (length: {chunk_length})")
    
    if chunk_length < 10:  # Minimum viable chunk size
        raise ValueError(f"チャンク {index + 1}/{total_chunks} が短すぎます")
    
    return True

def summarize_chunk_with_retry(chunk, index, total_chunks, max_length, min_length):
    """Summarize a single chunk with gradual size reduction on failure"""
    chunk_sizes = [512, 256, 128]  # Gradually reduce chunk size on failure
    model = get_summarizer()
    
    for chunk_size in chunk_sizes:
        try:
            # Truncate chunk if needed
            if get_text_length(chunk) > chunk_size:
                logger.info(f"Reducing chunk {index + 1} size to {chunk_size}")
                # Find the last sentence boundary before chunk_size
                sentences = re.split(r'([。．.!?！？\n])', chunk[:chunk_size * 2])
                truncated_chunk = ''
                current_length = 0
                
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')
                    sentence_length = get_text_length(sentence)
                    if current_length + sentence_length > chunk_size:
                        break
                    truncated_chunk += sentence
                    current_length += sentence_length
                
                chunk = truncated_chunk
            
            validate_chunk(chunk, index, total_chunks)
            logger.info(f"Attempting summarization of chunk {index + 1}/{total_chunks} with size {chunk_size}")
            
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
            max_answer_length=100  # Add reasonable limit
        )
        return result['answer']
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        raise Exception(f"質問応答に失敗しました: {str(e)}")

# Start loading models in background
threading.Thread(target=load_models, daemon=True).start()
