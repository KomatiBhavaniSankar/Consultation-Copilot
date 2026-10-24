"""Command line: turn the knowledge/ folders into vector databases.

  python -m app.ingest                      # new/changed files only
  python -m app.ingest --rebuild            # wipe and re-embed everything
  python -m app.ingest --only childbirth    # one folder
"""
import argparse

from app.modes import DOMAINS

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="delete and rebuild the collection(s)")
    ap.add_argument("--only", choices=list(DOMAINS), help="only this knowledge folder")
    args = ap.parse_args()
    from app.kb import ingest_all   # imported late: loads the embedding model
    ingest_all(rebuild=args.rebuild, only=args.only)
