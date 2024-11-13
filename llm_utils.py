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

def create_chunks(text, chunk_size=1024, min_chunk_size=100):
    """Create properly sized chunks of text with Japanese support"""
    if not text:
        raise ValueError("入力テキストが空です")
    
    # Clean the text first
    text = clean_text(text)
    is_japanese = is_japanese_text(text)
    logger.info(f"Text contains Japanese characters: {is_japanese}")
    
    # Adjust chunk sizes for Japanese text
    if is_japanese:
        chunk_size = chunk_size // 2  # Smaller chunks for Japanese
        min_chunk_size = min_chunk_size // 2
    
    # If text is shorter than minimum size, return as is
    text_length = get_text_length(text)
    if text_length <= min_chunk_size:
        logger.info(f"Text length ({text_length}) is smaller than minimum chunk size, returning as single chunk")
        return [text]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Split into sentences first
    sentences = re.split(r'([。．.!?！？\n])', text)
    current_sentence = ""
    
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            current_sentence = sentences[i]
            if i + 1 < len(sentences):
                current_sentence += sentences[i + 1]  # Add the separator back
                
            sentence_length = get_text_length(current_sentence)
            
            # If adding this sentence would exceed chunk size and we have enough content
            if current_length + sentence_length > chunk_size and current_length >= min_chunk_size:
                chunk_text = ''.join(current_chunk)
                logger.info(f"Created chunk with length: {get_text_length(chunk_text)}")
                chunks.append(chunk_text)
                current_chunk = [current_sentence]
                current_length = sentence_length
            else:
                current_chunk.append(current_sentence)
                current_length += sentence_length
    
    # Add the last chunk if it meets minimum size
    if current_chunk and current_length >= min_chunk_size:
        chunk_text = ''.join(current_chunk)
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

def summarize_text(text, max_length=130, min_length=30, max_retries=3):
    """Summarize the given text with improved Japanese support"""
    try:
        if not text:
            raise ValueError("入力テキストが空です")
        
        text_length = get_text_length(text)
        logger.info(f"Starting summarization of text (length: {text_length})")
        model = get_summarizer()
        
        # Create properly sized chunks
        chunks = create_chunks(text)
        logger.info(f"Created {len(chunks)} chunks for summarization")
        
        summaries = []
        for i, chunk in enumerate(chunks):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Validate chunk before processing
                    validate_chunk(chunk, i, len(chunks))
                    logger.info(f"Processing chunk {i+1}/{len(chunks)} (length: {get_text_length(chunk)})")
                    
                    # Ensure the chunk is within model's maximum length
                    if get_text_length(chunk) > 1024:
                        logger.warning(f"Chunk {i+1} exceeds maximum length, truncating...")
                        chunk = chunk[:1024]
                    
                    summary = model(chunk, 
                                  max_length=max_length,
                                  min_length=min_length,
                                  do_sample=False)
                    
                    if not summary or not isinstance(summary, list) or len(summary) == 0:
                        raise ValueError(f"チャンク {i+1} の要約結果が無効です")
                    
                    summaries.append(summary[0]['summary_text'])
                    logger.info(f"Successfully summarized chunk {i+1}/{len(chunks)}")
                    break
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Attempt {retry_count} failed for chunk {i+1}: {str(e)}")
                    if retry_count == max_retries:
                        logger.error(f"Failed to summarize chunk {i+1} after {max_retries} attempts")
                        raise ValueError(f"チャンク {i+1} の要約に失敗しました: {str(e)}")
                    time.sleep(1)  # Wait before retry
        
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
