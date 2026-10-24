"""Offline test harness for the diabetes copilot pipeline.

What is REAL: every function in app/main.py (screening, tests, trends, history, plan, retrieve, grade,
advise, finalize) and the SQLite record book.
What is FAKED: the language model (scripted per test), the embedding database (word-overlap search) and,
only if they are not installed, fastapi / langgraph / langchain, so the suite runs with no GPU, no
internet and no API key:   python -m pytest tests
"""
import importlib
import os
import re
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp = tempfile.mkdtemp(prefix="copilot_test_")
os.environ.update({"SQLITE_PATH": os.path.join(_tmp, "records.db"), "STATIC_DIR": os.path.join(_tmp, "static"),
                   "AUTO_INGEST": "false", "LLM_PREFERENCE": "local", "GOOGLE_API_KEY": "", "XAI_API_KEY": ""})


# ---------------------------------------------------------------- stand-ins for missing libraries
class _Anything:
    """Accepts any call; works as a decorator."""
    def __call__(self, *a, **k):
        return a[0] if len(a) == 1 and callable(a[0]) and not k else _Anything()

    def __getattr__(self, name):
        return _Anything()


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    m.__getattr__ = lambda attr: _Anything()
    sys.modules[name] = m
    return m


def _install_missing_libraries():
    def missing(name):
        try:
            importlib.import_module(name)
            return False
        except Exception:
            return True

    if missing("fastapi"):
        class HTTPException(Exception):
            def __init__(self, status_code=500, detail=""):
                super().__init__(detail)
                self.status_code, self.detail = status_code, detail
        _module("fastapi", FastAPI=lambda *a, **k: _Anything(), File=_Anything(), Form=_Anything(),
                HTTPException=HTTPException, Request=type("Request", (), {}), UploadFile=type("UploadFile", (), {}))
        _module("fastapi.concurrency", run_in_threadpool=_Anything())
        _module("fastapi.responses", HTMLResponse=type("HTMLResponse", (), {}), Response=type("Response", (), {}))
        _module("fastapi.staticfiles", StaticFiles=_Anything())
    if missing("langchain_core"):
        class Document:
            def __init__(self, page_content="", metadata=None):
                self.page_content, self.metadata = page_content, metadata or {}
        _module("langchain_core")
        _module("langchain_core.documents", Document=Document)
    if missing("langchain_text_splitters"):
        class Splitter:
            def __init__(self, chunk_size=1000, chunk_overlap=0, **k):
                self.n = chunk_size

            def split_text(self, text):
                return [text[i:i + self.n] for i in range(0, len(text), self.n)] or [""]
        _module("langchain_text_splitters", RecursiveCharacterTextSplitter=Splitter)
    if missing("langgraph"):
        class StateGraph:   # runs the nodes in the order the edges were added (the app's graph is a straight line)
            def __init__(self, _state):
                self.nodes, self.order = {}, []

            def add_node(self, name, fn):
                self.nodes[name] = fn

            def add_edge(self, a, b):
                if a not in ("__start__",) and b != "__end__":
                    pass
                self.order.append((a, b))

            def compile(self):
                graph = self

                class App:
                    async def ainvoke(self, start):
                        state, cur = dict(start), "__start__"
                        nxt = dict(graph.order)
                        while nxt.get(cur) and nxt[cur] != "__end__":
                            cur = nxt[cur]
                            state.update(graph.nodes[cur](state) or {})
                        return state
                return App()
        _module("langgraph")
        _module("langgraph.graph", StateGraph=StateGraph, START="__start__", END="__end__")


_install_missing_libraries()


# ---------------------------------------------------------------- fake vector stores and guideline search
def _words(text):
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


class FakeStore:
    """Word-overlap stand-in for a Chroma collection."""
    def __init__(self):
        self.docs = []
        self._collection = types.SimpleNamespace(count=lambda: len(self.docs))

    def add_documents(self, docs):
        self.docs.extend(docs)

    def delete(self, where=None):
        key, val = next(iter((where or {}).items()), (None, None))
        self.docs = [d for d in self.docs if d.metadata.get(key) != val]

    def similarity_search(self, query, k=4):
        q = _words(query)
        ranked = sorted(self.docs, key=lambda d: -len(q & _words(d.page_content)))
        return ranked[:k]


def _guideline_passages(mode):
    from app.modes import MODES
    out = []
    for domain in MODES[mode]["domains"]:
        for f in sorted((ROOT / "knowledge" / domain).glob("*.md")):
            for para in re.split(r"\n\s*\n", f.read_text(encoding="utf-8")):
                if len(para.strip()) > 40:
                    out.append({"source": f.name, "page_label": "-", "text": para.strip()})
    return out


def _fake_search(mode, query, k):
    q = _words(query)
    hits = sorted(_guideline_passages(mode), key=lambda p: -len(q & _words(p["text"])))
    return [h for h in hits[:k] if q & _words(h["text"])] or hits[:1]


_kb = _module("app.kb", RETRIEVAL_K=4, case_stores={m: FakeStore() for m in ("emergency", "critical", "lifestyle", "maternity")},
              ingest_all=lambda *a, **k: {}, kb_status=lambda: {}, mode_chunks=lambda *a, **k: [], search=_fake_search)


# ---------------------------------------------------------------- scripted language model
class ScriptedLLM:
    """Decides what the 'model' answers from the wording of the prompt. mode: 'good' | 'empty' (valid but blank) | 'garbage' (wrong JSON shape) | 'down' (raises)."""
    def __init__(self):
        self.mode, self.script = "good", {}

    def __call__(self, prompt, as_json=False, **kw):
        if self.mode == "down":
            raise RuntimeError("model unavailable (test)")
        if self.mode == "empty":
            return {}
        if self.mode == "garbage":
            return ["not", "a", "json", "object"]
        s = self.script
        if "information extraction assistant" in prompt:
            return s.get("extract", {})
        if "diabetes screening assistant" in prompt:
            return s.get("predict", {})
        if "take-home care sheet" in prompt:
            return s.get("sheet", {})
        if "plan evidence retrieval" in prompt:
            return s.get("plan", {"questions": [{"question": "First-line treatment and lifestyle advice for diabetes", "tool": "guidelines"},
                                                {"question": "How similar patients were treated", "tool": "clinic_records"}]})
        if "grade search results" in prompt:
            return {"relevant_ids": sorted(set(re.findall(r"\[(E\d+)\]", prompt)))}
        if "Rewrite each one" in prompt:
            return {"queries": {}}
        if "qualified doctor" in prompt:
            return s.get("advice", {})
        return {}


LLM = ScriptedLLM()
_module("app.llm", LLMS=[], ask=LLM)


@pytest.fixture(scope="session")
def main():
    return importlib.import_module("app.main")


@pytest.fixture()
def llm():
    LLM.mode, LLM.script = "good", {}
    return LLM


@pytest.fixture()
def clean_records(main):
    """Empty record book and empty case index before each test."""
    with main.db() as conn:
        conn.execute("DELETE FROM consultations")
    _kb.case_stores["lifestyle"].docs.clear()
    yield
