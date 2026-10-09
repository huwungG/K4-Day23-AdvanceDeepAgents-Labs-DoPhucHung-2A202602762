"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_ARXIV_NS = "{http://www.w3.org/2005/Atom}"
_arxiv_lock = threading.Lock()
_ARXIV_LAST_CALL = [0.0]


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt >= attempts - 1:
                raise
            if exc.retry_after is not None:
                try:
                    delay = float(exc.retry_after)
                except (TypeError, ValueError):
                    delay = base * (2.0**attempt)
                    delay = min(delay, cap) + random.uniform(0, 1.0)
                    delay = min(delay, cap)
                else:
                    if delay < 0:
                        delay = 0.0
                    delay = min(delay, cap)
            else:
                delay = min(base * (2.0**attempt), cap)
                delay = delay + random.uniform(0, min(1.0, cap - delay) if cap - delay > 0 else 0.0)
                delay = min(delay, cap)
            time.sleep(delay)


def _parse_retry_after(value):
    if value is None:
        return None
    try:
        return max(0.0, float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _http_get(url, params=None, timeout=30.0):
    """GET that converts retryable conditions into RetryableError (for use inside with_retry)."""
    try:
        resp = httpx.get(url, params=params, timeout=timeout, follow_redirects=True,
                         headers={"User-Agent": "deep-research-lab/1.0 (mailto:student@example.org)"})
    except httpx.TransportError as exc:
        raise RetryableError(f"transport error: {exc}") from exc
    if resp.status_code in RETRYABLE_STATUS:
        raise RetryableError(f"HTTP {resp.status_code} from {url}",
                             retry_after=_parse_retry_after(resp.headers.get("Retry-After")))
    return resp


def _collapse(text, limit=600):
    text = " ".join(str(text or "").split())
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


def _redact(text, secrets):
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


# ---- Exa MCP helpers ----
_RATE_PHRASES = re.compile(r"rate.?limit|too many requests|quota.{0,30}exceed|status.?429|error.?429|rate_limited|rate-limited", re.I)


def _exa_is_rate_limited(text, meta):
    # 1. Explicit flags inside result._meta.
    if isinstance(meta, dict):
        for key, value in meta.items():
            kl = str(key).lower()
            if any(w in kl for w in ("rate", "limit", "quota", "throttl", "retry", "429")):
                if value is True or (isinstance(value, (int, float)) and value != 0):
                    return True
                if isinstance(value, str) and value.strip().lower() not in ("", "false", "0", "no", "none"):
                    return True
        # Nested scan: stringify small metas and look for signals.
        try:
            blob = json.dumps(meta).lower()
        except Exception:
            blob = ""
        if "ratelimit" in blob or "rate_limit" in blob or "rate limited" in blob or "too many requests" in blob:
            return True
    # 2. HTTP-200-with-message body: a rate notice has no search results ("URL:").
    if isinstance(text, str) and _RATE_PHRASES.search(text):
        if "URL:" not in text:
            return True
        # Even with URL:-like content, Exa's own notice names the key / dashboard.
        low = text.lower()
        if "exaapikey" in low or "dashboard.exa.ai" in low or "free tier" in low or "upgrade" in low:
            return True
    return False


def _exa_endpoint():
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if key:
        return f"{EXA_URL}?exaApiKey={key}", key
    return EXA_URL, ""


def _exa_call(tool_name, arguments, timeout=60.0):
    """Call one Exa MCP tool over plain HTTP (JSON-RPC) and return (text, meta).

    Raises RetryableError on rate limits / retryable HTTP, other Exceptions otherwise.
    """
    endpoint, key = _exa_endpoint()
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": tool_name, "arguments": arguments}}

    def _do():
        try:
            resp = httpx.post(endpoint, json=payload, timeout=timeout,
                              headers={"Content-Type": "application/json",
                                       "Accept": "application/json, text/event-stream"})
        except httpx.TransportError as exc:
            raise RetryableError(f"Exa transport error: {type(exc).__name__}") from exc
        if resp.status_code in RETRYABLE_STATUS:
            raise RetryableError(f"Exa HTTP {resp.status_code}",
                                 retry_after=_parse_retry_after(resp.headers.get("Retry-After")))
        if resp.status_code != 200:
            raise RuntimeError(f"Exa HTTP {resp.status_code}")
        # SSE: find data: lines.
        data_obj = None
        for line in resp.text.splitlines():
            stripped = line.strip()
            if stripped.startswith("data:"):
                candidate = stripped[len("data:"):].strip()
                if candidate == "[DONE]":
                    continue
                try:
                    data_obj = json.loads(candidate)
                except ValueError:
                    continue
                break
        if data_obj is None:
            # Fallback: whole body might be plain JSON.
            try:
                data_obj = json.loads(resp.text)
            except ValueError as exc:
                raise RuntimeError("Exa returned an unreadable response") from exc
        if isinstance(data_obj, dict) and "error" in data_obj and data_obj["error"]:
            err = data_obj["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            if _RATE_PHRASES.search(str(msg)):
                raise RetryableError(f"Exa rate limited: {msg}")
            raise RuntimeError(f"Exa JSON-RPC error: {msg}")
        result = (data_obj.get("result", {}) if isinstance(data_obj, dict) else {})
        meta = result.get("_meta") if isinstance(result, dict) else None
        parts = []
        content = result.get("content", []) if isinstance(result, dict) else []
        for item in content if isinstance(content, list) else []:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
        text = "\n\n".join(parts).strip()
        if _exa_is_rate_limited(text, meta):
            raise RetryableError("Exa rate limited (free tier): retry later or set EXA_API_KEY")
        return text, meta

    # Longer cap: shared-IP rate limits can last minutes (GUIDE 1.4).
    return with_retry(_do, attempts=6, base=2.0, cap=60.0)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    try:
        terms = re.findall(r"[\w\-]+", str(query or ""), flags=re.UNICODE)
        # Drop boolean operators / lone punctuation so LLM phrasing cannot break the query.
        terms = [t for t in terms if t.strip("-_") and t.upper() not in ("AND", "OR", "NOT")]
        if not terms:
            return "NO RESULTS"
        try:
            limit = int(max_results)
        except (TypeError, ValueError):
            limit = 10
        limit = max(1, min(30, limit))
        search_query = " AND ".join(f"all:{t}" for t in terms)

        with _arxiv_lock:
            wait = 3.0 - (time.monotonic() - _ARXIV_LAST_CALL[0])
            if wait > 0:
                time.sleep(wait)
            try:
                def _do():
                    return _http_get(ARXIV_URL, params={"search_query": search_query,
                                                       "sortBy": "submittedDate",
                                                       "sortOrder": "descending",
                                                       "start": 0,
                                                       "max_results": limit}, timeout=30.0)
                # Longer cap + extra attempts: whole class may share one IP (GUIDE 1.2).
                resp = with_retry(_do, attempts=6, base=2.0, cap=60.0)
            finally:
                _ARXIV_LAST_CALL[0] = time.monotonic()

        root = xml.etree.ElementTree.fromstring(resp.text)
        records = []
        for entry in root.findall(f"{_ARXIV_NS}entry"):
            id_el = entry.find(f"{_ARXIV_NS}id")
            if id_el is None or not id_el.text:
                continue
            raw = id_el.text.strip()
            m = re.search(r"/abs/([^/\s?#]+)", raw)
            raw_id = m.group(1) if m else raw.rsplit("/", 1)[-1]
            clean_id = re.sub(r"v\d+$", "", raw_id)
            if not clean_id:
                continue
            pub_el = entry.find(f"{_ARXIV_NS}published")
            title_el = entry.find(f"{_ARXIV_NS}title")
            sum_el = entry.find(f"{_ARXIV_NS}summary")
            records.append({
                "id": clean_id,
                "url": f"https://arxiv.org/abs/{clean_id}",
                "published": (pub_el.text.strip()[:10] if pub_el is not None and pub_el.text else ""),
                "title": " ".join((title_el.text or "").split()) if title_el is not None else "",
                "summary": _collapse(sum_el.text if sum_el is not None and sum_el.text else ""),
            })
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:  # never raise: the agent tries another source
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
def _hf_record(item):
    paper = item.get("paper", {}) if isinstance(item, dict) else {}
    pid = paper.get("id") if isinstance(paper, dict) else None
    if not pid:
        return None
    if isinstance(paper, dict):
        summary = paper.get("ai_summary") or paper.get("summary") or item.get("summary") or ""
        title = paper.get("title") or item.get("title") or ""
        published = paper.get("publishedAt") or item.get("publishedAt") or ""
        upvotes = paper.get("upvotes", 0) or 0
        github = paper.get("githubRepo") or ""
        stars = paper.get("githubStars", 0) or 0
    else:
        summary, title, published, upvotes, github, stars = "", "", "", 0, "", 0
    try:
        upvotes = int(upvotes)
    except (TypeError, ValueError):
        upvotes = 0
    try:
        stars = int(stars)
    except (TypeError, ValueError):
        stars = 0
    return {"id": str(pid),
            "url": f"https://huggingface.co/papers/{pid}",
            "published": str(published)[:10],
            "title": " ".join(str(title).split()),
            "summary": _collapse(summary),
            "upvotes": upvotes,
            "github": str(github or ""),
            "stars": stars}


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 30
        limit = max(1, min(100, limit))
        params = {"limit": limit}
        if str(date or "").strip():
            params["date"] = str(date).strip()

        def _do():
            return _http_get(HF_DAILY_URL, params=params, timeout=30.0)

        resp = with_retry(_do, attempts=5, base=1.0, cap=30.0)
        try:
            items = resp.json()
        except ValueError as exc:
            return f"ERROR: bad JSON from Hugging Face daily papers: {exc}"
        if not isinstance(items, list):
            return f"ERROR: unexpected Hugging Face daily papers response"
        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            rec = _hf_record(item)
            if rec:
                records.append(rec)
        kw = str(keyword or "").strip().lower()
        if kw:
            records = [r for r in records if kw in (r["title"] + " " + r["summary"]).lower()]
        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        if not str(query or "").strip():
            return "NO RESULTS"
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 10
        limit = max(1, min(50, limit))

        def _do():
            return _http_get(HF_SEARCH_URL, params={"q": str(query), "limit": limit}, timeout=30.0)

        resp = with_retry(_do, attempts=5, base=1.0, cap=30.0)
        try:
            items = resp.json()
        except ValueError as exc:
            return f"ERROR: bad JSON from Hugging Face search: {exc}"
        if not isinstance(items, list):
            return f"ERROR: unexpected Hugging Face search response"
        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            rec = _hf_record(item)
            if rec:
                records.append(rec)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    _key = (os.getenv("EXA_API_KEY") or "").strip()
    try:
        if not str(query or "").strip():
            return "NO RESULTS"
        try:
            num_results = int(num_results)
        except (TypeError, ValueError):
            num_results = 5
        num_results = max(1, min(10, num_results))
        obj = str(objective or "").strip()
        if not obj:
            obj = f"Find authoritative, detailed sources about: {str(query).strip()}"
        text, _meta = _exa_call("web_search_exa", {"query": str(query), "objective": obj, "numResults": num_results})
        if not text.strip():
            return "NO RESULTS"
        return text
    except Exception as exc:
        return f"ERROR: {_redact(f'{type(exc).__name__}: {exc}', [_key])}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    _key = (os.getenv("EXA_API_KEY") or "").strip()
    try:
        target = str(url or "").strip()
        if not target or not (target.startswith("http://") or target.startswith("https://")):
            return "NO RESULTS"
        text, _meta = _exa_call("web_fetch_exa", {"urls": [target]})
        if not text.strip():
            return "NO RESULTS"
        if len(text) > 12000:
            text = text[:12000].rstrip() + "…"
        return text
    except Exception as exc:
        return f"ERROR: {_redact(f'{type(exc).__name__}: {exc}', [_key])}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
