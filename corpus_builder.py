#!/usr/bin/env python3
"""Explicit opt-in Wikipedia sample acquisition and auditable local corpus loading.

Not a legal approval tool or factual verifier. Network only for `wiki` command.
No background downloads, external dependencies, or silent data collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import tempfile
import os
from collections import Counter
from email.utils import parsedate_to_datetime
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LICENSE = 'CC BY-SA 4.0 (verify article-specific terms)'
LICENSE_URL = 'https://creativecommons.org/licenses/by-sa/4.0/'
API = 'https://en.wikipedia.org/w/api.php'
TOPICS = {
    'mathematics': ['Probability theory', 'Linear algebra', 'Bayesian inference', 'Mathematical proof'],
    'computing': ['Computer science', 'Algorithm', 'Computer security', 'Machine learning'],
    'physical_science': ['Thermodynamics', 'Scientific method', 'Quantum mechanics', 'Classical mechanics'],
    'life_science': ['Evolution', 'Cell biology', 'Genetics', 'Ecology'],
    'social_science': ['Economics', 'Sociology', 'Psychology', 'Political science'],
    'history': ['History of science', 'History of mathematics', 'Industrial Revolution', 'Historiography'],
    'engineering': ['Control theory', 'Systems engineering', 'Electrical engineering', 'Reliability engineering'],
    'philosophy': ['Epistemology', 'Logic', 'Ethics', 'Philosophy of science'],
    'medicine': ['Epidemiology', 'Evidence-based medicine', 'Public health', 'Clinical trial'],
    'arts': ['Music theory', 'Architecture', 'Literary criticism', 'Art history'],
}


def balanced_topics(domains: list[str], max_pages: int) -> list[tuple[str, str]]:
    """Deterministic round-robin coverage instead of first-domain bias.

    Does not manufacture extra pages or claim equal quality between domains.
    """
    if max_pages < 1:
        raise ValueError('max_pages must be positive')
    unknown = [d for d in domains if d not in TOPICS]
    if unknown:
        raise ValueError(f'Unknown domains: {unknown}')
    if not domains:
        raise ValueError('Select at least one domain')
    order = list(dict.fromkeys(domains))
    seen_titles = set()
    selected = []
    for i in range(max(len(TOPICS[d]) for d in order)):
        for domain in order:
            choices = TOPICS[domain]
            if i >= len(choices):
                continue
            title = choices[i]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            selected.append((domain, title))
            if len(selected) >= max_pages:
                return selected
    return selected


def text_quality(text: str) -> dict:
    """Cheap diagnostics. Does not establish accuracy or editorial quality."""
    words = re.findall(r"\b[\w'-]+\b", text.casefold())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    counts = Counter(lines)
    largest_line_share = max(counts.values()) / len(lines) if lines else 1.0
    return {'characters': len(text), 'words': len(words),
            'unique_word_fraction': len(set(words)) / len(words) if words else 0.0,
            'repeated_line_fraction': largest_line_share if len(lines) > 1 else 0.0}


def digest(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def canonical(content: str) -> str:
    return re.sub(r'\s+', ' ', content).strip().casefold()


@dataclass(frozen=True)
class Document:
    path: str
    title: str
    text: str
    domain: str
    license: str
    url: str
    sha256: str
    provenance: str


def read_documents(root: Path, default_license: str = 'UNVERIFIED', max_bytes: int = 16_000_000) -> list[Document]:
    if not root.exists():
        raise FileNotFoundError(f'Corpus path does not exist: {root}')
    if root.is_symlink():
        raise ValueError('Corpus root symlinks are not supported')
    manifest_path = root / 'manifest.jsonl' if root.is_dir() else None
    entries = {}
    if manifest_path and manifest_path.exists():
        for row in manifest_path.read_text(encoding='utf-8').splitlines():
            if not row.strip():
                continue
            item = json.loads(row)
            name = str(item['path'])
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Manifest contains an unsafe path')
            if name in entries:
                raise ValueError(f'Duplicate manifest path {name}')
            entries[name] = item
    paths = ([root] if root.is_file() else sorted(p for p in root.rglob('*')
              if p.is_file() and p.suffix.lower() in ('.txt', '.md')
              and not any(part.startswith('.') for part in p.relative_to(root).parts)))
    result = []
    seen = set()
    for p in paths:
        if p.is_symlink() or (root.is_dir() and not p.resolve().is_relative_to(root.resolve())):
            raise ValueError(f'Symlink or path outside corpus root: {p}')
        if p.stat().st_size > max_bytes:
            raise ValueError(f'Document exceeds --max-document-bytes: {p}')
        name = p.name if root.is_file() else p.relative_to(root).as_posix()
        content = p.read_text(encoding='utf-8')
        if not content.strip():
            continue
        signature = digest(content)
        meta = entries.get(name, {})
        if meta.get('sha256') and meta['sha256'] != signature:
            raise ValueError(f'Manifest checksum mismatch: {name}')
        norm_hash = digest(canonical(content))
        if norm_hash in seen:
            continue
        seen.add(norm_hash)
        result.append(Document(name, meta.get('title', p.stem), content,
                               meta.get('domain', 'unclassified'),
                               meta.get('license', default_license),
                               meta.get('url', ''), signature,
                               'manifest' if meta else 'local_unverified'))
    if not result:
        raise ValueError(f'No unique nonempty .txt/.md documents in {root}')
    # Fail closed for dangling manifest entries; silently ignoring missing source files is unsafe.
    missing = set(entries) - {x.path for x in result}
    if missing:
        raise ValueError(f'Manifest entries absent or deduplicated: {sorted(missing)[:5]}')
    return result


def split_documents(docs: list[Document], seed: int, val_fraction: float = .15):
    if len(docs) < 2:
        raise ValueError('Require at least two distinct documents for document-level holdout. '
                         'Use smoke for a synthetic demonstration.')
    if not 0 < val_fraction < .5:
        raise ValueError('val_fraction must be between 0 and 0.5')
    target = max(1, min(len(docs) - 1, round(len(docs) * val_fraction)))
    rng = random.Random(seed)
    groups: dict[str, list[Document]] = {}
    for d in docs:
        groups.setdefault(d.domain, []).append(d)
    for group in groups.values():
        rng.shuffle(group)
    # Maximize held-out domain breadth while retaining a training example for
    # each domain when possible. Respect the requested holdout fraction.
    counts = {domain: 0 for domain in groups}
    domain_order = sorted(groups)
    rng.shuffle(domain_order)
    chosen = []
    for _ in range(target):
        available = [domain for domain in domain_order
                     if len(groups[domain]) - counts[domain] > 1]
        if not available:
            available = [domain for domain in domain_order
                         if len(groups[domain]) - counts[domain] > 0]
        if not available:
            raise RuntimeError('Unable to produce requested holdout')
        # Choose less represented domains first, then larger groups (with a
        # seeded shuffle as the stable tie-breaker).
        domain = min(available, key=lambda d: (counts[d] / len(groups[d]), -len(groups[d]), domain_order.index(d)))
        chosen.append(groups[domain][counts[domain]])
        counts[domain] += 1
    val_paths = {d.path for d in chosen}
    train = [d for d in docs if d.path not in val_paths]
    val = [d for d in docs if d.path in val_paths]
    assert {d.sha256 for d in train}.isdisjoint({d.sha256 for d in val})
    return train, val


def corpus_fingerprint(train: list[Document], val: list[Document]) -> str:
    return digest(json.dumps({'train': [(d.path, d.sha256, d.license, d.url) for d in train],
                              'val': [(d.path, d.sha256, d.license, d.url) for d in val]},
                             sort_keys=True, ensure_ascii=False))


def wiki_fetch(title: str, user_agent: str, timeout: float, max_bytes: int) -> dict:
    if not user_agent or not ('@' in user_agent or 'https://' in user_agent):
        raise ValueError('Provide a descriptive --user-agent with contact email or HTTPS URL')
    args = {'action': 'query', 'format': 'json', 'formatversion': '2', 'redirects': '1',
            'prop': 'extracts|info|revisions', 'inprop': 'url', 'rvprop': 'ids|timestamp',
            'titles': title, 'explaintext': '1', 'maxlag': '5'}
    url = API + '?' + urllib.parse.urlencode(args)
    request = urllib.request.Request(url, headers={'User-Agent': user_agent,
                                                   'Accept': 'application/json'})
    # Wikimedia can respond with 429/503 + Retry-After; honour throttling.
    # Network remains opt-in and sequential. No infinite retries.
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise ValueError(f'Wikipedia response exceeds {max_bytes} bytes: {title}')
            data = json.loads(body)
            if data.get('error', {}).get('code') == 'maxlag':
                if attempt == 2:
                    raise RuntimeError(f'Wikipedia maxlag retries exhausted: {title}')
                time.sleep(5 * (attempt + 1))
                continue
            if 'error' in data:
                raise ValueError(f"Wikipedia API error: {data['error'].get('code', 'unknown')}")
            pages = data['query']['pages']
            if len(pages) != 1 or pages[0].get('missing'):
                raise ValueError(f'Page missing: {title}')
            return pages[0]
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == 2:
                raise
            retry_after = exc.headers.get('Retry-After') if exc.headers else None
            delay = 5 * (attempt + 1)
            if retry_after:
                try:
                    delay = max(delay, int(retry_after))
                except ValueError:
                    try:
                        from datetime import datetime, timezone
                        delay = max(delay, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                    except (TypeError, ValueError, OverflowError):
                        pass
            time.sleep(min(120.0, delay))
    raise RuntimeError(f'Wikipedia fetch exhausted retries: {title}')


def collect_wiki(out: Path, user_agent: str, domains: list[str], max_pages: int,
                 timeout: float = 20, delay: float = .5, fetcher=wiki_fetch):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite an existing nonempty corpus: {out}')
    if out.exists() and not out.is_dir():
        raise FileExistsError(f'Output exists and is not a directory: {out}')
    if max_pages < 2:
        raise ValueError('max-pages must be at least 2 for document-level validation')
    if delay < 0 or timeout <= 0:
        raise ValueError('delay must be nonnegative and timeout positive')
    requested = balanced_topics(domains, max_pages)
    if len(requested) < 2:
        raise ValueError('Fewer than two distinct topics available')
    data = []
    seen = set()
    for i, (domain, title) in enumerate(requested):
        try:
            page = fetcher(title, user_agent, timeout, 3_000_000)
            content = str(page.get('extract', '')).strip()
            quality = text_quality(content)
            if len(content) < 400 or quality['words'] < 40:
                print(f'SKIP (short extract): {title}', flush=True)
                continue
            h = digest(canonical(content))
            if h in seen:
                print(f'SKIP (duplicate): {title}', flush=True)
                continue
            seen.add(h)
            revs = page.get('revisions') or []
            rev = revs[0] if revs else {}
            stable_url = page.get('fullurl') or ('https://en.wikipedia.org/wiki/' + urllib.parse.quote(title.replace(' ', '_')))
            file_name = f'{i:03d}_{re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")}.txt'
            data.append((file_name, content, {'path': file_name, 'title': page.get('title', title),
                'domain': domain, 'source': 'English Wikipedia', 'url': stable_url,
                'history_url': stable_url + '?action=history', 'revision_id': rev.get('revid'),
                'revision_timestamp': rev.get('timestamp'), 'license': LICENSE, 'license_url': LICENSE_URL,
                'license_status': 'SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED',
                'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'sha256': digest(content), 'processing': 'MediaWiki plaintext extract',
                'quality_diagnostics': quality}))
            print(f'Fetched {title}: {len(content):,} characters', flush=True)
        except (urllib.error.URLError, ValueError, KeyError, OSError, json.JSONDecodeError, RuntimeError) as exc:
            print(f'WARNING: skipping {title}: {exc}', flush=True)
        if i < len(requested) - 1:
            time.sleep(delay)
    if len(data) < 2:
        raise RuntimeError('Acquisition returned fewer than two usable documents; no corpus was written.')

    # Stage a complete corpus and publish it only after all files + manifest exist.
    # No network activity occurs after staging starts.
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{out.name}.staging-', dir=out.parent))
    try:
        for path, content, _ in data:
            (stage / path).write_text(content + '\n', encoding='utf-8')
        rows = []
        for path, _, meta in data:
            meta['sha256'] = digest((stage / path).read_text(encoding='utf-8'))
            rows.append(meta)
        (stage / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        # Validate what was actually written before publication.
        read_documents(stage)
        if out.exists():
            if any(out.iterdir()):
                raise FileExistsError(f'Corpus created concurrently at {out}')
            out.rmdir()
        os.replace(stage, out)
    finally:
        if stage.exists():
            import shutil
            shutil.rmtree(stage)
    by_domain = dict(sorted(Counter(row['domain'] for row in rows).items()))
    print(json.dumps({'documents': len(rows), 'requested': len(requested),
                      'domains': by_domain,
                      'warnings': ['Source license statement is not independent rights clearance',
                                   'Downloaded source text is not fact verified']}, indent=2))
    print(f'Corpus ready: {out}. Review licenses, attribution, quality and source claims before reuse.')


def main():
    p = argparse.ArgumentParser(description='Build or validate a provenance-aware text corpus.')
    sub = p.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('wiki', help='OPT-IN network acquisition of Wikipedia text extracts')
    w.add_argument('--out', default='./corpus_wiki')
    w.add_argument('--user-agent', required=True, help='Descriptive name and contact email/URL')
    w.add_argument('--domains', nargs='+', choices=sorted(TOPICS), default=list(TOPICS))
    w.add_argument('--max-pages', type=int, default=32)
    w.add_argument('--timeout', type=float, default=20)
    w.add_argument('--delay', type=float, default=.5)
    a = sub.add_parser('audit', help='Check document hash, duplicates, and license metadata')
    a.add_argument('--data', required=True)
    a.add_argument('--require-license', action='store_true')
    args = p.parse_args()
    if args.cmd == 'wiki':
        collect_wiki(Path(args.out), args.user_agent, args.domains, args.max_pages, args.timeout, args.delay)
    else:
        docs = read_documents(Path(args.data))
        if args.require_license and any(d.license == 'UNVERIFIED' for d in docs):
            raise ValueError('Unknown licenses; add manifest metadata before production use')
        qualities = [text_quality(d.text) for d in docs]
        domain_counts = dict(sorted(Counter(d.domain for d in docs).items()))
        print(json.dumps({'documents': len(docs), 'domains': domain_counts,
                          'total_utf8_bytes': sum(len(d.text.encode('utf-8')) for d in docs),
                          'total_words': sum(q['words'] for q in qualities),
                          'short_documents_under_100_words': sum(q['words'] < 100 for q in qualities),
                          'potentially_repetitive_documents': sum(q['repeated_line_fraction'] > .5 and q['words'] > 100 for q in qualities),
                          'unknown_licenses': sum(d.license == 'UNVERIFIED' for d in docs),
                          'license_claims_independently_verified': False,
                          'factual_claims_verified': False}, indent=2))


if __name__ == '__main__':
    main()
