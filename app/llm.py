"""Language models: Gemini -> Grok (xAI) -> local Ollama, tried in order.

LLM_PREFERENCE (in .env) is handy while testing:
  auto  : cloud APIs first, local Ollama as the last resort   (default)
  api   : cloud APIs only
  local : local Ollama only (no data leaves the machine)
"""
import json
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()


def _env_list(name: str, default: str) -> list:
    return [m.strip() for m in os.getenv(name, default).split(",") if m.strip()]


# Comma-separated lists. Free-tier quota is per model, so extra models = extra daily requests.
GEMINI_MODELS = _env_list("GEMINI_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite")
XAI_MODELS = _env_list("XAI_MODELS", "grok-4")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))   # Ollama defaults to 4096, which truncates long consultations
LLM_PREFERENCE = os.getenv("LLM_PREFERENCE", "auto").strip().lower()



def build_llms():
    llms = []

    use_api = LLM_PREFERENCE in ("auto", "api")
    use_local = LLM_PREFERENCE in ("auto", "local")

    # 1) Gemini (one entry per model, each has its own free quota)
    if use_api and os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        for model in GEMINI_MODELS:
            llms.append((f"gemini:{model}",
                         ChatGoogleGenerativeAI(model=model, temperature=0.2, max_retries=1)))

    # 2) Grok via xAI's OpenAI-compatible API
    if use_api and os.getenv("XAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        for model in XAI_MODELS:
            llms.append((f"grok:{model}",
                         ChatOpenAI(model=model, temperature=0.2, max_retries=1,
                                    api_key=os.getenv("XAI_API_KEY"), base_url=XAI_BASE_URL)))

    # 3) Local Ollama: last resort, never runs out
    if not use_local:
        return llms
    from langchain_ollama import ChatOllama
    extra = {}
    think = os.getenv("OLLAMA_THINKING", "").lower()   # "off" = faster, "on" = slower but may be more careful
    if think in ("on", "off"):
        extra["reasoning"] = think == "on"
    llms.append(("ollama", ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.2,
                                      num_ctx=OLLAMA_NUM_CTX, format="json", **extra)))
    return llms


LLMS = build_llms()


def _to_text(content) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    else:
        text = str(content)
    # Reasoning models (qwen3, deepseek-r1...) print their thinking first; drop it.
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
    if "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in low or "quota" in low \
            or "insufficient" in low or "credits" in low:
        m = RETRY_DELAY_RE.search(text)
        return min(int(m.group(1)) + 5, 6 * 3600) if m else 900
    if "NOT_FOUND" in text or "404" in text or "does not exist" in low:   # model retired / not available to this key
        return 24 * 3600
    if "401" in text or "403" in text or "api key" in low or "permission" in low:   # bad or unauthorised key
        return 6 * 3600
    if "UNAVAILABLE" in text or "503" in text or "overloaded" in low:   # temporary overload
        return 60
    return 0


def ask(prompt: str, as_json: bool = False):
    """Try each model in order; skip ones in cooldown; fall through to Grok, then Ollama."""
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
    if not LLMS:
        raise RuntimeError("No language model is configured. Set GOOGLE_API_KEY / XAI_API_KEY, or LLM_PREFERENCE=local with Ollama.")
    raise RuntimeError(f"All language models failed: {last_error}")
