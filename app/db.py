"""Patient-record database.

  DATABASE_URL set   -> Supabase (Postgres). Use this on free hosts: their disks are wiped on restart.
  DATABASE_URL empty -> the local SQLite file (SQLITE_PATH). Used for local development and the tests.

main.py only ever calls `with db() as conn: conn.execute("... ? ...", args)`, so the Postgres side
below translates that small SQLite-flavoured API (? placeholders, LIKE, lastrowid, row[0] / row["x"]).
"""
import os
import re
import sqlite3
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
USE_POSTGRES = DATABASE_URL.startswith(("postgres://", "postgresql://"))
SQLITE_PATH = os.getenv("SQLITE_PATH", str(Path(__file__).resolve().parent.parent / "records.db"))

# ---------------------------------------------------------------- schemas

SQLITE_SCHEMA = """CREATE TABLE IF NOT EXISTS consultations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    patient_name TEXT NOT NULL,
    diagnosis TEXT,
    medicines TEXT,          -- JSON list
    vitals TEXT,
    doctor_approach TEXT,
    transcript TEXT,
    dialogue TEXT            -- JSON list of {speaker, text}
)"""

# Same columns as supabase/schema.sql. JSON is kept as TEXT on purpose: the app json.loads() it itself.
POSTGRES_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS public.consultations (
        id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        created_at      TEXT NOT NULL,
        patient_name    TEXT NOT NULL,
        diagnosis       TEXT,
        medicines       TEXT,
        vitals          TEXT,
        doctor_approach TEXT,
        transcript      TEXT,
        dialogue        TEXT,
        sheet           TEXT,
        prediction      TEXT,
        mode            TEXT DEFAULT 'lifestyle'
    )""",
    "CREATE INDEX IF NOT EXISTS consultations_mode_idx ON public.consultations (mode)",
]


# ---------------------------------------------------------------- Postgres adapter

class _Row(dict):
    """A dict that also answers row[0], like sqlite3.Row does."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


def _row_factory(cursor):
    names = [c.name for c in (cursor.description or [])]
    return lambda values: _Row(zip(names, values))


def to_postgres(sql: str) -> str:
    """SQLite-style SQL -> Postgres: '?' -> '%s', literal '%' escaped, LIKE -> ILIKE (SQLite's LIKE ignores case)."""
    sql = sql.replace("%", "%%").replace("?", "%s")
    return re.sub(r"\bLIKE\b", "ILIKE", sql)


class _Cursor:
    def __init__(self, cur, inserted: bool):
        self._cur = cur
        self.lastrowid = None
        if inserted:
            row = cur.fetchone()
            self.lastrowid = row["id"] if row else None

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __iter__(self):
        return iter(self._cur)


class _PgConn:
    """`with db() as conn:` -> borrows a pooled connection, commits on success, rolls back on error."""

    def __enter__(self):
        self._ctx = _get_pool().connection()
        self._conn = self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)

    def execute(self, sql: str, args=()):
        sql = to_postgres(sql)
        inserted = sql.lstrip().upper().startswith("INSERT")
        if inserted and "RETURNING" not in sql.upper():
            sql += " RETURNING id"
        return _Cursor(self._conn.execute(sql, tuple(args)), inserted)


_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        from psycopg_pool import ConnectionPool

        url = DATABASE_URL
        if "sslmode=" not in url:
            url += ("&" if "?" in url else "?") + "sslmode=require"
        _pool = ConnectionPool(
            url,
            min_size=1,
            max_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_idle=240,                       # drop idle connections before Supabase's pooler does
            timeout=30,
            check=ConnectionPool.check_connection,
            # prepare_threshold=None: no server-side prepared statements, so it also works through
            # Supabase's transaction pooler (port 6543), not just the session pooler (5432).
            kwargs={"row_factory": _row_factory, "prepare_threshold": None},
            open=True,
        )
    return _pool


# ---------------------------------------------------------------- public API

def db():
    if USE_POSTGRES:
        return _PgConn()
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema():
    """Create the table if it is missing (and, for old SQLite files, add the columns later versions need)."""
    if USE_POSTGRES:
        with db() as conn:
            for statement in POSTGRES_SCHEMA:
                conn.execute(statement)
        print("[db] using Supabase / Postgres")
        return
    Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.execute(SQLITE_SCHEMA)
    with db() as conn:
        cols = [c[1] for c in conn.execute("PRAGMA table_info(consultations)")]
        for col in ("sheet", "prediction"):          # columns added in later versions
            if col not in cols:
                conn.execute(f"ALTER TABLE consultations ADD COLUMN {col} TEXT")
        if "mode" not in cols:                        # old rows are lifestyle (diabetes) records
            conn.execute("ALTER TABLE consultations ADD COLUMN mode TEXT DEFAULT 'lifestyle'")
    print(f"[db] using local SQLite: {SQLITE_PATH}")
