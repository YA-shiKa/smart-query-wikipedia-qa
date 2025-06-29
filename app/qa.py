from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from app.config import QA_MODEL

qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL)
qa_pipeline = pipeline("text2text-generation", model=qa_model, tokenizer=qa_tokenizer)

def answer_question(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}"
    answer = qa_pipeline(prompt, max_length=100, do_sample=False)[0]['generated_text']
    return answer.strip()
