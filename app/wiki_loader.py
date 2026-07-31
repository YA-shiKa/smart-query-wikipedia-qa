import logging

import requests
import wikipediaapi
from transformers import AutoTokenizer

from app.config import EMBEDDING_MODEL, WIKI_CHUNK_SIZE, WIKI_CHUNK_OVERLAP

logger = logging.getLogger("wikirag.wiki_loader")

# A descriptive User-Agent is required by Wikimedia's API etiquette policy.
# Requests without one are increasingly rejected outright (an HTML error
# page instead of JSON), which is exactly what caused the unmaintained
# `wikipedia` PyPI package (which sends no User-Agent at all) to start
# raising JSONDecodeError. We use plain `requests` for search with this
# header, and reuse the same header for wikipediaapi's page fetch.
USER_AGENT = "SelfHealingRAG/1.0 (https://github.com/YA-shiKa; contact: example@example.com)"
SEARCH_API_URL = "https://en.wikipedia.org/w/api.php"

wiki_wiki = wikipediaapi.Wikipedia(language="en", user_agent=USER_AGENT)

# Loaded once on first use instead of at import time (or on every
# split_text() call, as the original code did) -- avoids a network call
# just from importing this module, and avoids re-downloading/re-initializing
# the tokenizer on every single request.
_chunk_tokenizer = None


def _get_chunk_tokenizer():
    global _chunk_tokenizer
    if _chunk_tokenizer is None:
        _chunk_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return _chunk_tokenizer


def _search_wikipedia_titles(query, limit=5):
    """Search Wikipedia via the MediaWiki API directly instead of the
    unmaintained `wikipedia` PyPI package. Returns a list of page titles,
    or [] on any network/parsing failure (treated as "not found" by the
    caller rather than crashing the pipeline)."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
    }
    try:
        resp = requests.get(SEARCH_API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.error("Wikipedia search request failed for '%s': %s", query, e)
        return []
    return [r["title"] for r in data.get("query", {}).get("search", [])]


def get_wikipedia_content(topic):
    try:
        logger.info("Searching Wikipedia for topic: %s", topic)
        search_results = _search_wikipedia_titles(topic)
        logger.debug("Wikipedia search results: %s", search_results)

        if not search_results:
            return {"status": "not_found", "content": None}

        best_match = search_results[0]
        logger.info("Best match chosen: %s", best_match)

        page = wiki_wiki.page(best_match)
        if not page.exists():
            logger.warning("Page does not exist according to wikipediaapi: %s", best_match)
            return {"status": "not_found", "content": None}

        # The original code caught wikipedia.exceptions.DisambiguationError
        # here, but that exception is only ever raised by wikipedia.page(),
        # which was never called -- so disambiguation was silently never
        # detected. Detect it properly via the page's own categories instead.
        if any("disambiguation" in c.lower() for c in page.categories):
            options = search_results
            logger.info("Ambiguous topic '%s'. Options: %s", topic, options)
            return {"status": "ambiguous", "options": options}

        return {"status": "ok", "content": page.text, "title": best_match}

    except Exception as e:  # noqa: BLE001 - surface any fetch error to the caller
        logger.exception("Error fetching Wikipedia content for topic '%s'", topic)
        return {"status": "error", "message": str(e)}


def split_text(text, tokenizer=None, chunk_size=WIKI_CHUNK_SIZE, chunk_overlap=WIKI_CHUNK_OVERLAP):
    tokenizer = tokenizer or _get_chunk_tokenizer()
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
