#!/usr/bin/env python3
"""Traceable offline text retrieval; retrieval is NOT factual verification.

Data written to SQLite stays local. Requires Python SQLite with FTS5 enabled.
Only returns literal excerpts from recorded documents, never invented citations.
"""
from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path
from corpus_builder import read_documents


def index_corpus(data: Path, database: Path):
    docs = read_documents(data)
    database.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database)
    try:
        conn.execute('CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, path TEXT UNIQUE, title TEXT, domain TEXT, license TEXT, source_url TEXT, sha256 TEXT, content TEXT)')
        conn.execute('CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(title, content, content="documents", content_rowid="id")')
        with conn:
            conn.execute('DELETE FROM docs_fts')
            conn.execute('DELETE FROM documents')
            for doc in docs:
                cur = conn.execute('INSERT INTO documents (path,title,domain,license,source_url,sha256,content) VALUES (?,?,?,?,?,?,?)',
                                   (doc.path, doc.title, doc.domain, doc.license, doc.url, doc.sha256, doc.text))
                conn.execute('INSERT INTO docs_fts(rowid,title,content) VALUES (?,?,?)', (cur.lastrowid, doc.title, doc.text))
    finally:
        conn.close()
    return len(docs)


def search(database: Path, query: str, limit: int = 5):
    if not query.strip() or not 1 <= limit <= 50:
        raise ValueError('query must be nonempty, and 1 <= limit <= 50')
    # FTS5 phrase quoting avoids user-provided syntax/operator injection.
    terms = [t.replace('"', '""') for t in query.split() if t.strip()]
    if not terms:
        return []
    match = ' OR '.join('"' + t + '"' for t in terms[:16])
    conn = sqlite3.connect(database)
    try:
        rows = conn.execute('''SELECT d.path,d.title,d.domain,d.license,d.source_url,d.sha256,
                                     snippet(docs_fts, 1, '[', ']', ' … ', 28),
                                     bm25(docs_fts) AS score
                              FROM docs_fts JOIN documents d ON docs_fts.rowid=d.id
                              WHERE docs_fts MATCH ? ORDER BY score LIMIT ?''', (match, limit)).fetchall()
    finally:
        conn.close()
    return [{'status': 'SOURCE_RETRIEVED_NOT_FACT_CHECKED', 'path': r[0], 'title': r[1],
             'domain': r[2], 'license': r[3], 'source_url': r[4], 'sha256': r[5],
             'excerpt': r[6], 'rank_score': r[7]} for r in rows]


def main():
    p = argparse.ArgumentParser(description='Local source retrieval without unsupported truth claims')
    sub = p.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('index')
    i.add_argument('--data', required=True)
    i.add_argument('--db', default='./evidence.sqlite3')
    s = sub.add_parser('search')
    s.add_argument('--db', default='./evidence.sqlite3')
    s.add_argument('--query', required=True)
    s.add_argument('--limit', type=int, default=5)
    args = p.parse_args()
    if args.cmd == 'index':
        print(json.dumps({'indexed_documents': index_corpus(Path(args.data), Path(args.db))}))
    else:
        print(json.dumps({'query': args.query, 'results': search(Path(args.db), args.query, args.limit),
                          'notice': 'A retrieved source is not proof that a statement is true.'},
                         ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
