from transformers import pipeline
import threading
import time
import logging
import math

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

def clean_text(text):
    """Clean and validate text for summarization"""
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    return text

def create_chunks(text, chunk_size=1024, min_chunk_size=100):
    """Create properly sized chunks of text for summarization"""
    if not text:
        raise ValueError("Input text is empty")
    
    # Clean the text first
    text = clean_text(text)
    
    # If text is shorter than minimum size, return as is
    if len(text) <= min_chunk_size:
        return [text]
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > chunk_size and current_length >= min_chunk_size:
            # Join the current chunk and add to chunks
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    # Add the last chunk if it meets minimum size
    if current_chunk and current_length >= min_chunk_size:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_text(text, max_length=130, min_length=30, max_retries=3):
    """Summarize the given text with improved error handling"""
    try:
        if not text:
            raise ValueError("Input text is empty")
        
        logger.info(f"Starting summarization of text (length: {len(text)})")
        model = get_summarizer()
        
        # Create properly sized chunks
        chunks = create_chunks(text)
        logger.info(f"Created {len(chunks)} chunks for summarization")
        
        summaries = []
        for i, chunk in enumerate(chunks):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    summary = model(chunk, 
                                 max_length=max_length,
                                 min_length=min_length,
                                 do_sample=False)
                    
                    if not summary or not isinstance(summary, list) or len(summary) == 0:
                        raise ValueError(f"Invalid summary output for chunk {i+1}")
                        
                    summaries.append(summary[0]['summary_text'])
                    break
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Attempt {retry_count} failed for chunk {i+1}: {str(e)}")
                    if retry_count == max_retries:
                        logger.error(f"Failed to summarize chunk {i+1} after {max_retries} attempts")
                        raise
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
        raise ValueError("Question and context must not be empty")
    
    try:
        model = get_qa_model()
        result = model(question=question, context=context)
        return result['answer']
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        raise Exception(f"質問応答に失敗しました: {str(e)}")

# Start loading models in background
threading.Thread(target=load_models, daemon=True).start()
