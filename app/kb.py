"""Vector databases. Every folder in knowledge/ becomes its own Chroma collection.

Drop PDFs / .md / .txt / .html / .json into a folder and run `python -m app.ingest`
(or restart the server, or POST /kb/reindex). Only new or changed files are embedded.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.modes import DOMAINS, MODES, case_collection

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", BASE_DIR / "knowledge"))
DB_DIR = os.getenv("DB_DIR", str(BASE_DIR / "chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
MANIFEST = Path(DB_DIR) / "ingest_manifest.json"
TEXT_EXT = {".txt", ".md", ".json"}
HTML_EXT = {".html", ".htm"}

print("Loading local embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

kb_stores = {d: Chroma(collection_name=c["collection"], embedding_function=embeddings, persist_directory=DB_DIR)
             for d, c in DOMAINS.items()}
case_stores = {m: Chroma(collection_name=case_collection(m), embedding_function=embeddings, persist_directory=DB_DIR)
               for m in MODES}
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


# ---------- reading files ----------

def _simple(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


def load_file(path: Path) -> list:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(str(path)).load()
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if ext in HTML_EXT:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "noscript", "iframe", "form"]):
            tag.decompose()
        raw = soup.get_text("\n", strip=True)
    return [Document(page_content=raw, metadata={"source": str(path)})] if raw.strip() else []


def _skip(p: Path) -> bool:
    n = p.name.lower()
    return n.startswith(("_", ".", "readme")) or p.suffix.lower() not in TEXT_EXT | HTML_EXT | {".pdf"}


def _sha(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _delete_all(store: Chroma):
    ids = store.get()["ids"]
    for i in range(0, len(ids), 500):
        store.delete(ids=ids[i:i + 500])


# ---------- ingestion ----------

def ingest_domain(domain: str, rebuild: bool = False) -> dict:
    """Embed new/changed files of one folder; forget files that were removed."""
    store, folder = kb_stores[domain], KNOWLEDGE_DIR / domain
    folder.mkdir(parents=True, exist_ok=True)
    manifest = _manifest()
    seen = {} if rebuild else manifest.get(domain, {})
    if rebuild:
        _delete_all(store)

    files = sorted(p for p in folder.rglob("*") if p.is_file() and not _skip(p))
    current = {str(p.relative_to(folder)): p for p in files}
    report = {"domain": domain, "added": [], "updated": [], "removed": [], "unchanged": 0, "empty": [], "chunks": 0}

    for rel in set(seen) - set(current):            # file deleted from the folder
        store.delete(where={"source_path": rel})
        report["removed"].append(rel)
        seen.pop(rel)

    for rel, path in current.items():
        digest = _sha(path)
        if seen.get(rel) == digest:
            report["unchanged"] += 1
            continue
        if rel in seen:                              # file changed: drop its old chunks first
            store.delete(where={"source_path": rel})
        try:
            docs = load_file(path)
        except Exception as error:
            print(f"  ! could not read {rel}: {error}")
            continue
        chunks = splitter.split_documents(docs)
        if not chunks:
            report["empty"].append(rel)              # e.g. a scanned PDF with no text layer
            continue
        for i, c in enumerate(chunks):
            c.metadata = {**_simple(c.metadata), "domain": domain, "source_path": rel, "chunk": i}
        ids = [f"{domain}:{digest[:12]}:{i}" for i in range(len(chunks))]
        for i in range(0, len(chunks), 64):
            store.add_documents(chunks[i:i + 64], ids=ids[i:i + 64])
        report["updated" if rel in seen else "added"].append(rel)
        report["chunks"] += len(chunks)
        seen[rel] = digest

    manifest[domain] = seen
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    return report


def ingest_all(rebuild: bool = False, only: Optional[str] = None) -> list:
    reports = []
    for domain in DOMAINS:
        if only and domain != only:
            continue
        r = ingest_domain(domain, rebuild)
        print(f"[kb] {domain}: +{len(r['added'])} new, ~{len(r['updated'])} changed, -{len(r['removed'])} removed, "
              f"{r['unchanged']} unchanged -> {kb_stores[domain]._collection.count()} chunks"
              + (f"  (no text found in: {', '.join(r['empty'])})" if r["empty"] else ""))
        reports.append(r)
    return reports


def kb_status() -> list:
    out = []
    for d, c in DOMAINS.items():
        folder = KNOWLEDGE_DIR / d
        files = [p for p in folder.rglob("*") if p.is_file() and not _skip(p)] if folder.is_dir() else []
        out.append({"domain": d, "label": c["label"], "folder": f"knowledge/{d}",
                    "files": len(files), "chunks": kb_stores[d]._collection.count()})
    return out


# ---------- searching ----------

def search(mode: str, query: str, k: int = RETRIEVAL_K) -> Optional[list]:
    """Best passages across every folder the mode uses. None = the mode has no indexed data at all."""
    hits, total = [], 0
    for d in MODES[mode]["domains"]:
        store = kb_stores[d]
        n = store._collection.count()
        total += n
        if n:
            for doc, score in store.similarity_search_with_score(query, k=min(k, n)):
                hits.append((score, d, doc))
    if not total:
        return None
    hits.sort(key=lambda h: h[0])
    out = []
    for _, d, doc in hits[:k]:
        page = doc.metadata.get("page")
        out.append({"source": os.path.basename(str(doc.metadata.get("source", "guideline"))),
                    "page_label": f"p.{page + 1}" if isinstance(page, int) else "",
                    "text": doc.page_content, "domain": DOMAINS[d]["label"]})
    return out


def mode_chunks(mode: str) -> int:
    return sum(kb_stores[d]._collection.count() for d in MODES[mode]["domains"])
