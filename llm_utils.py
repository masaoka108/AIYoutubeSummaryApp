import re
import threading
import unicodedata
import logging
from transformers import MBartForConditionalGeneration, MBartTokenizer, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
summarizer = None
qa_model = None

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
    
    # Normalize Japanese text
    text = unicodedata.normalize('NFKC', text)
    # Remove extra whitespace while preserving Japanese text
    text = re.sub(r'[\s\u3000]+', ' ', text).strip()
    return text

def split_into_sentences(text):
    """Split text into sentences with improved Japanese boundary detection"""
    if not text:
        return []

    # Enhanced pattern for Japanese sentence boundaries
    pattern = r'([。．.!?！？\n]|\s{2,}|(?<=」)(?![\s。．.!?！？])|(?<=）)(?![\s。．.!?！？]))'
    
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
                if sentence:
                    # Handle Japanese parentheses and quotes
                    if ('「' in sentence and '」' not in sentence) or ('（' in sentence and '）' not in sentence):
                        continue_idx = i + 2
                        while continue_idx < len(parts):
                            sentence += parts[continue_idx]
                            if '」' in parts[continue_idx] or '）' in parts[continue_idx]:
                                break
                            continue_idx += 1
                        i = continue_idx
                    sentences.append(sentence)
            i += 2
        
        if not sentences:
            return [text.strip()] if text.strip() else []
            
        return sentences
        
    except Exception as e:
        logger.warning(f"Failed to split text into sentences: {str(e)}")
        return [text.strip()] if text.strip() else []

def create_chunks(text, max_chunk_size=256, min_chunk_size=50):
    """Create properly sized chunks with improved Japanese text handling"""
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
        
        # Handle long sentences
        if sentence_length > max_chunk_size:
            if current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long sentence at Japanese particles or punctuation
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
        
        # Normal sentence handling
        if current_length + sentence_length > max_chunk_size:
            if current_chunk:
                chunks.append(''.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add the last chunk
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    # Ensure we have at least one chunk
    if not chunks:
        return [text[:max_chunk_size]]
    
    return chunks

def summarize_chunk_with_retry(chunk, index, total_chunks, max_length=130, min_length=30):
    """Summarize chunk with improved Japanese text handling"""
    if not chunk or not chunk.strip():
        raise ValueError(f"チャンク {index + 1} が空です")
    
    chunk_sizes = [256, 128, 64]
    model = get_summarizer()
    
    for chunk_size in chunk_sizes:
        try:
            current_length = get_text_length(chunk)
            logger.info(f"Processing chunk {index + 1}/{total_chunks} (length: {current_length})")
            
            if current_length > chunk_size:
                chunk = clean_text(chunk[:chunk_size])
                current_length = get_text_length(chunk)
                logger.info(f"Truncated chunk {index + 1} to length: {current_length}")
            
            # Attempt summarization with Japanese-specific settings
            try:
                summary = model(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                
                if summary and isinstance(summary, list) and len(summary) > 0:
                    summary_text = summary[0].get('summary_text', '').strip()
                    if summary_text:
                        logger.info(f"Successfully summarized chunk {index + 1}/{total_chunks}")
                        return summary_text
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
    """Summarize text with improved Japanese support"""
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