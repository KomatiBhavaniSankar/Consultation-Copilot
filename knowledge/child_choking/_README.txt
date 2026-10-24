Put the child and infant choking (steps, first-aid PDFs, prevention) documents for this problem area in this folder.

Accepted: PDF (must contain real text, not only scanned images), .md, .txt, .html, .json
Subfolders are fine.

After adding or changing files, turn them into the vector database:
  Docker : docker compose exec app python -m app.ingest
  Local  : python -m app.ingest
  or restart the app (it ingests automatically), or POST /kb/reindex.
Only new or changed files are embedded. Deleted files are removed from the database.
Use `python -m app.ingest --rebuild --only child_choking` to rebuild this folder from scratch.

Files whose names start with "_" or "README" are ignored, like this one.

The .md notes already here are short STARTER notes so the system works on first run.
Replace or extend them with your own clinically reviewed documents.
Used by modes: Emergency
