import wikipedia
import wikipediaapi
from app.config import WIKI_CHUNK_SIZE, WIKI_CHUNK_OVERLAP

wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='RAGPipeline/1.0 (https://github.com/YA-shiKa)'
)

def get_wikipedia_content(topic):
    try:
        print(f"\n🔍 Searching for topic: {topic}")
        search_results = wikipedia.search(topic)
        print(f"🔎 Wikipedia search results: {search_results}")

        if not search_results:
            return {"status": "not_found", "content": None}

        best_match = search_results[0]
        print(f"✅ Best match chosen: {best_match}")

        page = wiki_wiki.page(best_match)
        if not page.exists():
            print("🚫 Page does not exist according to wikipediaapi.")
            return {"status": "not_found", "content": None}

        return {"status": "ok", "content": page.text, "title": best_match}

    except wikipedia.exceptions.DisambiguationError as e:
        print(f"⚠️ Ambiguous topic. Options: {e.options}")
        return {"status": "ambiguous", "options": e.options}
    except Exception as e:
        print(f"💥 General error: {e}")
        return {"status": "error", "message": str(e)}

def split_text(text, tokenizer=None, chunk_size=512, chunk_overlap=50):
    from transformers import AutoTokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.convert_tokens_to_string(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - chunk_overlap
    return chunks

