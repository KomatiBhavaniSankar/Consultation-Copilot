"""Language models: Gemini only. Models in GEMINI_MODELS are tried in order.

Free-tier quota is per model, so extra models in the list = extra daily requests.
"""
import json
import os
import re
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def _env_list(name: str, default: str) -> list:
    return [m.strip() for m in os.getenv(name, default).split(",") if m.strip()]


GEMINI_MODELS = _env_list("GEMINI_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.8-flash")


def build_llms():
    if not os.getenv("GOOGLE_API_KEY"):
        return []
    return [(f"gemini:{model}",
             ChatGoogleGenerativeAI(model=model, temperature=0.2, max_retries=1))
            for model in GEMINI_MODELS]


LLMS = build_llms()


def _to_text(content) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    else:
        text = str(content)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model output")
    return json.loads(text[start:end + 1])


COOLDOWN: dict = {}   # model name -> time.time() until which it is skipped

RETRY_DELAY_RE = re.compile(r"""retryDelay['"]?\s*:\s*['"]?(\d+)s""")


def _cooldown_seconds(error: Exception) -> int:
    """Seconds to skip a model after a quota / missing / overloaded error (0 = a normal failure)."""
    text = str(error)
    low = text.lower()
    if "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in low or "quota" in low:
        m = RETRY_DELAY_RE.search(text)
        return min(int(m.group(1)) + 5, 6 * 3600) if m else 900
    if "NOT_FOUND" in text or "404" in text or "does not exist" in low:   # model name wrong / not available to this key
        return 24 * 3600
    if "401" in text or "403" in text or "api key" in low or "permission" in low:
        return 6 * 3600
    if "UNAVAILABLE" in text or "503" in text or "overloaded" in low:
        return 60
    return 0


def ask(prompt: str, as_json: bool = False):
    """Try each Gemini model in order, skipping ones in cooldown."""
    if not LLMS:
        raise RuntimeError("No language model is configured. Set GOOGLE_API_KEY.")
    last_error = None
    for name, model in LLMS:
        if time.time() < COOLDOWN.get(name, 0):
            continue
        try:
            text = _to_text(model.invoke(prompt).content)
            result = _parse_json(text) if as_json else text
            print(f"[llm] answered by {name}")
            return result
        except Exception as error:
            wait = _cooldown_seconds(error)
            if wait:
                COOLDOWN[name] = time.time() + wait
                print(f"[llm] {name} unavailable, skipping it for {max(wait // 60, 1)} min. "
                      f"Reason: {type(error).__name__}: {str(error)[:300]}")
            else:
                print(f"[llm] {name} failed: {type(error).__name__}: {str(error)[:200]}")
            last_error = error
    raise RuntimeError(f"All Gemini models failed: {last_error}")