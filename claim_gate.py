#!/usr/bin/env python3
"""Minimal epistemic claim gate: rejects unsupported statements by default.

A source quotation can be matched against archived bytes, but this does NOT prove
that the source's factual content is correct. Do not convert this distinction to
"verified fact" without independent expert / cross-source validation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from corpus_builder import read_documents


def inspect_claims(corpus: Path, claims: list[dict]):
    docs = {d.path: d for d in read_documents(corpus)}
    results = []
    for i, claim in enumerate(claims):
        kind = claim.get('kind', 'assertion')
        statement = str(claim.get('statement', '')).strip()
        path = claim.get('source_path', '')
        quote = claim.get('source_quote', '')
        if not statement:
            state, reason = 'REJECTED', 'Missing statement'
        elif kind == 'unknown':
            state, reason = 'ABSTENTION', 'Explicit unknown; no unsupported factual claim'
        elif kind == 'quotation':
            d = docs.get(path)
            if not d:
                state, reason = 'REJECTED', 'Source is missing from archived corpus'
            elif not quote or quote != statement or quote not in d.text:
                state, reason = 'REJECTED', 'Quoted statement is not an exact passage from the source'
            else:
                state, reason = 'SOURCE_QUOTATION_MATCHED', 'Exact source passage confirmed, factual truth NOT independently verified'
        elif kind == 'inference':
            d = docs.get(path)
            if d and quote and quote in d.text:
                state, reason = 'INFERENCE_UNVERIFIED', 'Source text located; logical validity and factual truth require additional checks'
            else:
                state, reason = 'REJECTED', 'Inference lacks a located source passage'
        else:
            state, reason = 'REJECTED', 'Unsupported assertion: no external truth verification'
        results.append({'index': i, 'statement': statement, 'kind': kind,
                        'state': state, 'reason': reason, 'source_path': path})
    return results


def main():
    p=argparse.ArgumentParser(description='Reject unsupported factual claims; source matching != truth')
    p.add_argument('--corpus', required=True)
    p.add_argument('--claims', required=True, help='JSON list of {kind,statement,source_path,source_quote}')
    a=p.parse_args()
    claims=json.loads(Path(a.claims).read_text(encoding='utf-8'))
    if not isinstance(claims, list):
        raise ValueError('--claims must contain a JSON list')
    print(json.dumps({'results': inspect_claims(Path(a.corpus), claims),
                      'note': 'Only exact source quotations are matched; source truth remains unverified'},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
