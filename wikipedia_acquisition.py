#!/usr/bin/env python3
"""Explicit, bounded Wikipedia category acquisition for dataset experiments.

Network access occurs only through the `collect` command. Text and provenance
are staged locally and published only after a complete, validated collection.
This tool does not establish legal clearance or independently verify articles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import email.utils
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

API = 'https://en.wikipedia.org/w/api.php'
TERMS = 'https://foundation.wikimedia.org/wiki/Terms_of_Use'
CC_BY_SA = 'https://creativecommons.org/licenses/by-sa/4.0/'
GFDL = 'https://www.gnu.org/licenses/fdl-1.3.html'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_request(params: dict, user_agent: str, timeout: float, max_bytes: int,
                 opener=urllib.request.urlopen, sleeper=time.sleep) -> dict:
    if not user_agent or not ('@' in user_agent or 'https://' in user_agent):
        raise ValueError('Provide a descriptive --user-agent with contact email or HTTPS URL')
    query = dict(params)
    query.update({'action': 'query', 'format': 'json', 'formatversion': '2', 'maxlag': '5'})
    request = urllib.request.Request(API + '?' + urllib.parse.urlencode(query),
                                     headers={'User-Agent': user_agent, 'Accept': 'application/json'})
    for attempt in range(4):
        try:
            with opener(request, timeout=timeout) as response:
                body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise ValueError(f'Wikipedia response exceeds {max_bytes} bytes')
            data = json.loads(body)
            error = data.get('error', {})
            if error.get('code') in {'maxlag', 'ratelimited'} and attempt < 3:
                sleeper(5 * (attempt + 1))
                continue
            if error:
                raise ValueError(f"Wikipedia API error: {error.get('code', 'unknown')}")
            return data
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt >= 3:
                raise
            retry_after = exc.headers.get('Retry-After') if exc.headers else None
            delay = 5 * (attempt + 1)
            if retry_after:
                try:
                    delay = max(5, min(120, int(retry_after)))
                except (ValueError, TypeError):
                    try:
                        retry_time = email.utils.parsedate_to_datetime(retry_after)
                        delay = max(5, min(120, int((retry_time - datetime.now(timezone.utc)).total_seconds())))
                    except (TypeError, ValueError, OverflowError):
                        pass
            sleeper(delay)
    raise RuntimeError('Wikipedia API retry limit exhausted')


def read_categories(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path}:{line_no}: invalid JSON: {exc}') from exc
        if not isinstance(row, dict) or not row.get('domain') or not row.get('category'):
            raise ValueError(f'{path}:{line_no}: each row requires domain and category')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', row['domain']):
            raise ValueError(f'{path}:{line_no}: invalid domain label')
        depth = row.get('max_depth', 0)
        if not isinstance(depth, int) or not 0 <= depth <= 3:
            raise ValueError(f'{path}:{line_no}: max_depth must be an integer from 0 to 3')
        rows.append({'domain': row['domain'], 'category': row['category'], 'max_depth': depth})
    if not rows:
        raise ValueError('Category list contains no rows')
    return rows


def enumerate_pages(category: dict, user_agent: str, timeout: float, delay: float,
                    max_pages: int, requester=json_request, sleeper=time.sleep,
                    max_categories: int = 100,
                    excluded_titles: set[str] | None = None) -> list[dict]:
    root = str(category['category']).removeprefix('Category:')
    queue = deque([(root, 0)])
    visited_categories = set()
    pages = {}
    while queue and len(pages) < max_pages and len(visited_categories) < max_categories:
        name, depth = queue.popleft()
        canonical = name.casefold()
        if canonical in visited_categories:
            continue
        visited_categories.add(canonical)
        continuation = {}
        while len(pages) < max_pages:
            params = {'list': 'categorymembers', 'cmtitle': 'Category:' + name,
                      'cmtype': 'page|subcat', 'cmlimit': '500', **continuation}
            result = requester(params, user_agent, timeout, 2_000_000)
            members = result.get('query', {}).get('categorymembers', [])
            for member in members:
                title = str(member.get('title', '')).strip()
                if member.get('ns') == 14 or title.startswith('Category:'):
                    if depth < category['max_depth']:
                        queue.append((title.removeprefix('Category:'), depth + 1))
                elif member.get('ns') == 0 and title:
                    if title.casefold() not in (excluded_titles or set()):
                        pages.setdefault(title, {'title': title, 'domain': category['domain'],
                                                 'category': 'Category:' + name})
                    if len(pages) >= max_pages:
                        break
            continuation = result.get('continue', {})
            if not continuation:
                break
            sleeper(delay)
        if queue:
            sleeper(delay)
    return list(pages.values())


def fetch_article(title: str, user_agent: str, timeout: float, requester=json_request) -> dict:
    result = requester({'prop': 'extracts|info|revisions', 'inprop': 'url',
                        'rvprop': 'ids|timestamp', 'titles': title, 'explaintext': '1'},
                       user_agent, timeout, 6_000_000)
    pages = result.get('query', {}).get('pages', [])
    if len(pages) != 1 or pages[0].get('missing'):
        raise ValueError(f'Wikipedia page missing: {title}')
    return pages[0]


def collect(out: Path, categories_file: Path, user_agent: str, max_pages: int,
            timeout: float = 30, delay: float = 1.0, min_words: int = 80,
            requester=json_request, sleeper=time.sleep, max_categories: int = 100,
            exclude_manifests: list[Path] | None = None) -> dict:
    if max_pages < 1 or timeout <= 0 or delay < 0 or min_words < 1 or max_categories < 1:
        raise ValueError('max-pages/min-words/max-categories must be positive; timeout positive; delay nonnegative')
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {out}')
    categories = read_categories(categories_file)
    selected, seen_titles = [], set()
    excluded_titles = set()
    for manifest_path in exclude_manifests or []:
        for row in (json.loads(line) for line in manifest_path.read_text(encoding='utf-8').splitlines() if line.strip()):
            title = row.get('title')
            if isinstance(title, str) and title.strip():
                excluded_titles.add(title.strip().casefold())
    # Give each selected domain an equal initial budget; the first broad
    # category cannot consume the whole request before later domains run.
    by_domain: dict[str, list[dict]] = {}
    for category in categories:
        by_domain.setdefault(category['domain'], []).append(category)
    domain_order = list(by_domain)
    base_budget, remainder = divmod(max_pages, len(domain_order))
    for domain_index, domain in enumerate(domain_order):
        domain_budget = base_budget + (1 if domain_index < remainder else 0)
        cat_list = by_domain[domain]
        cat_budget, cat_remainder = divmod(domain_budget, len(cat_list))
        domain_selected = 0
        for cat_index, category in enumerate(cat_list):
            budget = cat_budget + (1 if cat_index < cat_remainder else 0)
            budget = min(budget, domain_budget - domain_selected)
            if budget <= 0:
                continue
            candidates = enumerate_pages(category, user_agent, timeout, delay,
                                         budget, requester, sleeper, max_categories,
                                         excluded_titles)
            for page in candidates:
                key = page['title'].casefold()
                if key not in seen_titles and key not in excluded_titles:
                    seen_titles.add(key)
                    selected.append(page)
                    domain_selected += 1
                    if domain_selected >= domain_budget:
                        break
            sleeper(delay)
    if not selected:
        raise RuntimeError('Selected categories yielded no article pages')
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{out.name}.staging-', dir=out.parent))
    rows, hashes = [], set()
    try:
        for index, selection in enumerate(selected):
            try:
                page = fetch_article(selection['title'], user_agent, timeout, requester)
                text = str(page.get('extract', '')).strip()
                words = re.findall(r"\b[\w'-]+\b", text)
                if len(words) < min_words:
                    print(f"SKIP short extract: {selection['title']}", flush=True)
                    continue
                normalized = re.sub(r'\s+', ' ', text).strip().casefold()
                normalized_hash = digest(normalized.encode('utf-8'))
                if normalized_hash in hashes:
                    print(f"SKIP duplicate extract: {selection['title']}", flush=True)
                    continue
                hashes.add(normalized_hash)
                revision = (page.get('revisions') or [{}])[0]
                pageid = page.get('pageid')
                rev_id = revision.get('revid')
                if pageid is None or rev_id is None:
                    raise ValueError(f"Missing page or revision identifier: {selection['title']}")
                filename = f'{pageid}_{re.sub(r"[^a-z0-9]+", "_", selection["title"].lower()).strip("_")[:80]}.txt'
                content = text + '\n'
                (stage / filename).write_text(content, encoding='utf-8')
                url = page.get('fullurl') or ('https://en.wikipedia.org/wiki/' + urllib.parse.quote(selection['title'].replace(' ', '_')))
                lowered_domain = selection['domain'].lower().replace('-', '_')
                bucket = ('mathematics' if lowered_domain in {'mathematics', 'math'} else
                          'engineering' if lowered_domain == 'engineering' else
                          'encyclopedic' if lowered_domain == 'encyclopedic' else
                          'scientific' if lowered_domain in {'scientific', 'physics', 'physical_science',
                                                            'biology', 'life_science', 'computing', 'medicine'} else
                          'other_domains')
                row = {
                    'source_id': f'wikipedia-{pageid}-{rev_id}', 'path': filename,
                    'title': page.get('title', selection['title']), 'publisher': 'English Wikipedia',
                    'source': 'English Wikipedia', 'source_url': url, 'url': url,
                    'history_url': url + '?action=history', 'revision_id': rev_id,
                    'revision_timestamp': revision.get('timestamp'),
                    'publication_date': revision.get('timestamp') or datetime.now(timezone.utc).date().isoformat(),
                    'license_id': 'CC BY-SA 4.0 and GFDL; verify article-specific notices and imported content',
                    'license_url': CC_BY_SA, 'license_terms_url': TERMS,
                    'license_scope': 'Standard article text; revision-specific and third-party notices require review.',
                    'additional_license_urls': [GFDL],
                    'license_status': 'SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED',
                    'domain': selection['domain'], 'source_type': 'wikipedia_article',
                    'raw_category': bucket,
                    'source_category': selection['category'], 'source_group': f'wikipedia-page-{pageid}',
                    'retrieved_at': datetime.now(timezone.utc).isoformat(),
                    'content_hash': digest(content.encode('utf-8')),
                    'processing_history': [{'operation': 'MediaWiki plaintext extract', 'tool': 'wikipedia_acquisition.py'}],
                    'quality_diagnostics': {'words': len(words), 'characters': len(text)},
                    'source_quality_status': 'UNREVIEWED',
                    'attribution_note': 'Retain the article URL/history and license notice; plaintext extraction is a modification.',
                }
                rows.append(row)
                print(f"Fetched [{selection['domain']}] {selection['title']}: {len(words):,} words", flush=True)
            except (urllib.error.URLError, ValueError, KeyError, OSError, json.JSONDecodeError, RuntimeError) as exc:
                print(f"WARNING: skipping {selection['title']}: {exc}", flush=True)
            if index < len(selected) - 1:
                sleeper(delay)
        if not rows:
            raise RuntimeError('No usable articles; staged collection was not published')
        (stage / 'manifest.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        # Recheck exact hashes of the staged bytes before exposing the directory.
        for row in rows:
            if digest((stage / row['path']).read_bytes()) != row['content_hash']:
                raise ValueError(f"Staged article checksum mismatch: {row['path']}")
        if out.exists():
            raise FileExistsError(f'Output created concurrently: {out}')
        os.replace(stage, out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    counts = dict(sorted(Counter(row['domain'] for row in rows).items()))
    return {'documents': len(rows), 'requested_page_limit': max_pages, 'domains': counts,
            'license_clearance_verified': False, 'article_quality_independently_reviewed': False,
            'attribution_required': True, 'output': str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description='Opt-in, bounded Wikipedia category acquisition')
    sub = parser.add_subparsers(dest='cmd', required=True)
    command = sub.add_parser('collect')
    command.add_argument('--out', required=True)
    command.add_argument('--categories', required=True, help='JSONL rows containing domain, category, optional max_depth')
    command.add_argument('--exclude-manifest', action='append', default=[],
                         help='Existing JSONL manifest; skip titles already staged (repeatable)')
    command.add_argument('--user-agent', required=True, help='Descriptive name and contact email/HTTPS URL')
    command.add_argument('--max-pages', type=int, default=1000)
    command.add_argument('--min-words', type=int, default=80, help='Skip shorter extracts')
    command.add_argument('--max-categories', type=int, default=100, help='Maximum categories traversed per seed')
    command.add_argument('--timeout', type=float, default=30)
    command.add_argument('--delay', type=float, default=1.0)
    args = parser.parse_args()
    result = collect(Path(args.out), Path(args.categories), args.user_agent, args.max_pages,
                     args.timeout, args.delay, args.min_words, max_categories=args.max_categories,
                     exclude_manifests=[Path(p) for p in args.exclude_manifest])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
