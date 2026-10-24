---
title: Consultation Copilot
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Consultation Copilot (v5)

A doctor-patient copilot that **listens to a conversation by voice or reads it as text**, searches your own documents (stored in vector databases), and answers. It has four use cases, a Tailwind interface that works on phones and desktops, and a Docker setup for deployment.

| Mode | For | What happens | Knowledge folders used |
|---|---|---|---|
| **Emergency** | Parent, bystander, patient | Stroke, child or baby choking, chest pain. The built-in steps appear **instantly** (no AI wait). One AI call then adds details from your documents. | `heart_stroke`, `child_choking`, `critical_conditions` |
| **Critical care** | Doctor, paramedic | Heart attack and stroke. An instant **clinician checklist** (ECG, door-to-balloon, thrombolysis window...), then AI details. | `critical_conditions`, `heart_stroke` |
| **Lifestyle disease** | Doctor | Your original diabetes copilot: screening, tests, medicines, care sheet PDF. | `lifestyle_disease` |
| **Childbirth** | Obstetrician, midwife | Labor and delivery, with attention to older (35+) mothers. Agent plans searches, grades the passages, rewrites weak queries, answers only from what it found. | `childbirth` |

Every mode accepts **typed or pasted conversation text** and **voice**. In the two emergency modes, voice is *live*: the browser transcribes while people speak and the steps appear mid-sentence. In the other modes, the recording is transcribed on the server with Whisper (more accurate for medical words).

> **Safety.** This is decision support, not a diagnosis. The built-in emergency steps follow widely used first-aid guidance (AHA / Red Cross / NHS style) but must be **reviewed by your clinical lead** before real-world use. They are in `app/triage.py`. The call buttons use `EMERGENCY_NUMBER` / `AMBULANCE_NUMBER` from `.env` (default 112 / 108, India).

---

## 1. Quick start with Docker (recommended)

You need Docker Desktop (Windows/Mac) or Docker Engine + Compose (Linux). About 8 GB free disk.

```bash
cd agentic-rag
cp .env.example .env          # Windows PowerShell: copy .env.example .env
# edit .env: add GOOGLE_API_KEY and/or XAI_API_KEY, OR set LLM_PREFERENCE=local (see step 3)
docker compose up -d --build
```

The first build takes a while (it installs the libraries and downloads the speech and embedding models). Then open **http://localhost:8001**.
The first start also embeds the starter documents. Watch it with `docker compose logs -f app` until you see `Uvicorn running`.

### Which AI answers? (three ways, use whichever you have)

1. **Cloud API**: put `GOOGLE_API_KEY` (and/or `XAI_API_KEY`) in `.env`. Done. With `LLM_PREFERENCE=auto` the app tries Gemini, then Grok, then local Ollama.
2. **Local model in Docker** (nothing leaves your machine; good for testing):
   ```bash
   # in .env:  LLM_PREFERENCE=local   and   OLLAMA_MODEL=gemma3:4b
   docker compose --profile local-llm up -d --build
   docker compose exec ollama ollama pull gemma3:4b      # one-time download
   ```
   Roughly: 4B models need about 8 GB RAM, 12B models about 16 GB. CPU-only is slow (minutes per consultation). The emergency modes still answer instantly because the built-in steps do not need the model.
3. **Ollama already installed on your PC**: in `.env` set `OLLAMA_URL=http://host.docker.internal:11434` and `LLM_PREFERENCE=local`.

For testing, `LLM_PREFERENCE=api` uses only cloud models and `local` uses only Ollama.

### Everyday commands
```bash
docker compose up -d --build        # start / apply code changes
docker compose logs -f app          # watch the log
docker compose restart app
docker compose down                 # stop (data is kept)
```

---

## 2. Add your data and build the vector databases

Each problem area is a folder under `knowledge/`, and each folder becomes **its own vector database collection**:

```
knowledge/
  heart_stroke/          stroke information, PDFs, emergency steps
  child_choking/         child and infant choking, first-aid PDFs
  lifestyle_disease/     diabetes and other lifestyle conditions
  childbirth/            parturition, labor and delivery
  critical_conditions/   heart attack, cardiac arrest, stroke pathways
```

1. Copy your PDFs, `.md`, `.txt`, `.html` or `.json` files into the right folder. PDFs need real text (scanned images need OCR first).
2. Build the vector databases:
   ```bash
   docker compose exec app python -m app.ingest             # new or changed files only
   docker compose exec app python -m app.ingest --rebuild   # wipe and re-embed everything
   docker compose exec app python -m app.ingest --only childbirth
   ```
   Restarting the app also ingests automatically (`AUTO_INGEST=true`). Without Docker, run the same commands as `python -m app.ingest`.
3. Check what is indexed: `http://localhost:8000/kb/status` (files and chunks per folder).

The `knowledge/` folder is mounted into the container, so you do **not** rebuild the image to add documents. The `.md` files already in each folder are short **starter notes** so the system works on day one. Replace or extend them with your own reviewed documents. To fill `lifestyle_disease/` from the web: `python scripts/scrape_diabetes.py`.

**Add a brand-new problem area:** create `knowledge/<name>/`, add one line to `DOMAINS` in `app/modes.py`, and list it in the `domains` of any mode.

---

## 3. Using it

* **Mode tabs** at the top choose the use case. `http://your-site/#emergency` opens straight into Emergency mode: save that link to a phone home screen.
* **Emergency / Critical care**: tap a tile (stroke, heart attack, choking...) for the steps with no typing, or press the microphone and speak, or type. The red card is the instant built-in protocol; "Specific to this situation" is the AI result with sources. Cardiac arrest has a CPR beat button (110 per minute).
* **Lifestyle / Childbirth**: press the microphone (a recording is transcribed on the server) or open "paste or type the conversation". Typing `Doctor:` and `Patient:` at the start of lines splits the conversation by speaker.
* **Save to records** stores the consultation and adds it to that mode's search index. **Patient records** lists them, filterable by use case.
* The collapsible panes (Conversation, Sources, How the agent worked) open and close as before.

---

## 4. Use it on phones: HTTPS is required for the microphone

Browsers only allow the microphone on `https://` or `localhost`. Opening `http://192.168.x.x:8000` from a phone will show the page but the microphone will be blocked (typing and tiles still work). To fix this, deploy with HTTPS (next section).

---

## 5. Deploy on a server (VPS, cloud VM or on-premise)

1. Get a Linux server (Ubuntu 22.04+, 4 GB RAM minimum; more if you run a local model) and a **domain name**. Create a DNS `A` record pointing the domain to the server's IP. Open ports **80 and 443** in the firewall.
2. Install Docker: `curl -fsSL https://get.docker.com | sh`
3. Upload this folder (`scp -r agentic-rag user@server:~/` or `git clone`).
4. Configure:
   ```bash
   cd agentic-rag && cp .env.example .env && nano .env
   ```
   Set `DOMAIN=copilot.example.com`, `APP_BIND=127.0.0.1`, **`APP_PASSWORD=<a strong password>`**, and your API keys.
5. Start with automatic HTTPS:
   ```bash
   docker compose --profile https up -d --build
   ```
   Add `--profile local-llm` as well if you want Ollama on the same server. Visit `https://copilot.example.com` (browser asks for `APP_USER` / `APP_PASSWORD`).

**Update later:** upload the new code and run `docker compose --profile https up -d --build`. Records and vector databases live in the `copilot_data` volume and survive rebuilds.

**Back up** (records + vector databases):
```bash
docker run --rm -v consultation-copilot_copilot_data:/d -v "$PWD":/b alpine tar czf /b/backup.tgz -C /d .
```

**Bring over your old records** (from the earlier version's `records.db`):
```bash
docker compose up -d
docker compose cp /path/to/old/records.db app:/var/lib/copilot/records.db
docker compose exec -u root app chown app:app /var/lib/copilot/records.db
docker compose restart app
curl -X POST http://localhost:8001/records/reindex      # rebuilds the search index from the records (add -u admin:PASSWORD if set)
```
Old records appear under *Lifestyle disease*.

---

## 6. Run without Docker (development)

Needs Python 3.10+ and (optional) Node 18+.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # then edit it
cd frontend && npm install && npm run build:css && cd ..     # optional, see below
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000. If you skip the `npm` step the page loads Tailwind from its CDN instead (needs internet; fine for development, but build the CSS for production). For Ollama: install it, run `ollama pull gemma3:4b`, set `OLLAMA_URL=http://localhost:11434`.

**Tests** (no AI or models needed): `pip install pytest && python -m pytest tests`

`tests/test_diabetes_features.py` checks that, for every diabetes case (type 2, type 1, prediabetes, gestational, LADA, MODY, secondary, no diagnosis stated, odd spellings of the diagnosis) and for a model that answers well, answers blank, answers in the wrong shape or is down, the page still gets: possible types with confirming tests, medicines (general and what your clinic gave), similar patients of the same type, and the diagnosis plan (follow-up and considerations). A conversation that is not about diabetes must instead return questions to ask. When the model gives nothing for a section, that section shows the **built-in reference** (`STANDARD_CARE` in `app/main.py`) marked "unverified". Have a clinician review that table, as with `app/triage.py`.

---

## 7. Settings reference (`.env`)

| Setting | Meaning |
|---|---|
| `LLM_PREFERENCE` | `auto`, `api` or `local` |
| `GOOGLE_API_KEY`, `XAI_API_KEY` | cloud model keys (leave empty to skip) |
| `OLLAMA_MODEL`, `OLLAMA_URL`, `OLLAMA_NUM_CTX` | local model |
| `WHISPER_MODEL`, `WHISPER_LANG` | voice recording transcription (`auto` detects language) |
| `EMERGENCY_NUMBER`, `AMBULANCE_NUMBER` | numbers on the red call buttons |
| `EMERGENCY_LLM_TIMEOUT` | seconds the emergency modes wait for the AI before relying on the built-in steps |
| `AUTO_INGEST` | embed new/changed files at every start |
| `APP_USER`, `APP_PASSWORD` | login for the whole app (empty password = no login) |
| `APP_PORT`, `APP_BIND`, `DOMAIN` | deployment |

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| Microphone says blocked / does nothing on a phone | Use HTTPS (section 5). Allow the microphone in the browser. |
| Live listening says "needs an internet connection" | Chrome and Safari send speech to their own servers. Type, or use a non-emergency mode, where the server transcribes. |
| "No language model is configured" | Add an API key, or `LLM_PREFERENCE=local` and run Ollama. |
| Emergency result shows only the red card plus a warning | The AI was slow or unavailable. The steps are still valid. Check `docker compose logs app`. |
| "No documents are loaded" warning | Add files to `knowledge/<folder>` and run the ingest. |
| A PDF is not searchable | It has no text layer (scanned). The ingest log lists it as "no text found". OCR it first. |
| App is slow to start | It loads models and indexes documents. The health check allows 4 minutes. |
| Build fails while installing Python packages | Re-run. Packages are not version-pinned. After one good build run `docker compose run --rm app pip freeze > requirements.lock` and keep it. |

## 9. Before real use

* Have a clinician review every step in `app/triage.py` and the documents in `knowledge/`.
* Set `APP_PASSWORD`, use HTTPS, and keep `.env` and any key files out of git and out of images (the `.dockerignore` already excludes them).
* The app stores patient conversations. Decide where the server lives, who can reach it, and your backup and retention rules to meet your local health-data law.
* Typed/spoken text is sent to the cloud model only if you use a cloud key. For fully private operation use `LLM_PREFERENCE=local`.
* Built-in keyword triage understands **English**. Other languages still get the AI answer, and the tiles work in any language.

## Project layout

```
app/main.py        FastAPI server, pipelines, records
app/modes.py       the four modes and the knowledge folders
app/triage.py      instant emergency protocols and keyword triage
app/guidance.py    one-call emergency guidance, agentic childbirth advice
app/kb.py          vector databases and ingestion
app/llm.py         Gemini -> Grok -> Ollama chain
app/ingest.py      python -m app.ingest
frontend/          index.html + Tailwind (src/input.css, tailwind.config.js)
knowledge/         one folder per problem area
scripts/           scrape_diabetes.py
Dockerfile  docker-compose.yml  Caddyfile  .env.example
```
