from transformers import pipeline
import threading
import time

# Global variables for lazy loading
summarizer = None
qa_model = None
model_lock = threading.Lock()

def load_models():
    """Load models in a separate thread"""
    global summarizer, qa_model
    with model_lock:
        if summarizer is None:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        if qa_model is None:
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

def summarize_text(text, max_length=130):
    """Summarize the given text"""
    model = get_summarizer()
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summaries = []
    
    for chunk in chunks:
        summary = model(chunk, max_length=max_length, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    return ' '.join(summaries)

def answer_question(question, context):
    """Answer a question based on the context"""
    model = get_qa_model()
    result = model(question=question, context=context)
    return result['answer']

# Start loading models in background
threading.Thread(target=load_models, daemon=True).start()
