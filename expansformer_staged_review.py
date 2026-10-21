#!/usr/bin/env python3
"""Read-only quality/provenance triage of Expansformer staged source batches.

Produces a review queue. NEVER changes an admission, license, or quality status.
Never imports material into the canonical dataset. No network calls.

Examples:
 python expansformer_staged_review.py --base . --out ./review_queue_330 --expected-count 330
 python expansformer_staged_review.py --batch ./topic_gap_batch_240_revised \
   --batch ./topic_gap_human_expansion --out ./review_queue_330
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

HEX64 = re.compile(r'^(?:sha256:)?[0-9a-fA-F]{64}$')
PMC = re.compile(r'PMC\d+', re.I)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def norm_hash(text: str) -> str:
    return hashlib.sha256(' '.join(text.split()).casefold().encode('utf-8')).hexdigest()


def safe_resolve(base: Path, candidate: str) -> Path:
    p = Path(candidate)
    if not candidate or p.is_absolute() or '..' in p.parts:
        raise ValueError('unsafe or empty manifest path')
    base_real = base.resolve()
    resolved = (base / p).resolve()
    if not resolved.is_relative_to(base_real):
        raise ValueError('manifest path escapes manifest directory or traverses symlink')
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError('missing or non-regular manifest file')
    return resolved


def find_manifests(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f'Staged batch not found: {root}')
    if root.is_symlink():
        raise ValueError(f'Symlink batch root rejected: {root}')
    # A batch with a top-level manifest is treated as a single dataset: do not
    # ingest unrelated nested manifests/snapshots that may exist below it.
    top = root / 'manifest.jsonl'
    if top.is_file():
        return [top]
    result = []
    for m in sorted(root.rglob('manifest.jsonl')):
        if m.is_symlink() or not m.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Unsafe manifest: {m}')
        # Nested raw/archive/abstract/old/temp manifests are not candidates.
        rel = m.relative_to(root)
        if len(rel.parts) == 2 and m.parent.is_dir() and not m.parent.is_symlink():
            result.append(m)
    if not result:
        raise ValueError(f'No root or immediate-category manifest.jsonl found: {root}')
    return result


def canonical_source(row: dict) -> str:
    pmcid = str(row.get('pmcid', '')).upper()
    if not pmcid:
        url = str(row.get('source_url') or row.get('url') or '')
        m = PMC.search(url)
        if m:
            pmcid = m.group().upper()
    if pmcid:
        return 'pmc:' + pmcid
    group = str(row.get('source_group') or '').strip()
    if group:
        return 'group:' + group.casefold()
    url = str(row.get('source_url') or row.get('url') or '').strip().rstrip('/')
    if url:
        return 'url:' + url.casefold()
    return 'id:' + str(row.get('source_id', '')).casefold()


def load_records(roots: list[Path]) -> tuple[list[dict], list[dict], list[dict]]:
    records, errors, extras = [], [], []
    for root in roots:
        for manifest in find_manifests(root):
            folder = manifest.parent
            manifest_paths = set()
            for line_no, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(), 1):
                if not line.strip():
                    continue
                identifier = f'{manifest}:{line_no}'
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('manifest row is not an object')
                    relative = str(row.get('path') or '')
                    if relative in manifest_paths:
                        raise ValueError('repeated path in manifest')
                    manifest_paths.add(relative)
                    p = safe_resolve(folder, relative)
                    if p.suffix.lower() not in {'.txt', '.md'}:
                        raise ValueError('training text must end in .txt or .md')
                    actual = hash_file(p)
                    recorded = str(row.get('content_hash') or row.get('sha256') or '')
                    if not HEX64.fullmatch(recorded):
                        raise ValueError('missing or malformed content SHA256')
                    if actual != recorded.lower().removeprefix('sha256:'):
                        raise ValueError('content SHA256 mismatch')
                    raw = str(row.get('raw_source_path') or '')
                    if raw:
                        raw_p = safe_resolve(folder, raw)
                        raw_expected = str(row.get('raw_content_hash') or '')
                        if not HEX64.fullmatch(raw_expected) or hash_file(raw_p) != raw_expected.lower().removeprefix('sha256:'):
                            raise ValueError('raw-source SHA256 mismatch')
                    content = p.read_text(encoding='utf-8')
                    if not content.strip():
                        raise ValueError('empty text')
                    words = len(content.split())
                    records.append({
                        'source_id': str(row.get('source_id') or ''),
                        'canonical_source': canonical_source(row),
                        'batch': str(root.resolve()),
                        'manifest': str(manifest.resolve()),
                        'path': str(p),
                        'title': str(row.get('title') or p.stem),
                        'domain': str(row.get('domain') or folder.name),
                        'publisher': str(row.get('publisher') or ''),
                        'source_url': str(row.get('source_url') or row.get('url') or ''),
                        'license_status': str(row.get('license_status') or 'UNKNOWN'),
                        'license': str(row.get('license_id') or row.get('license') or ''),
                        'license_review_evidence': str(row.get('license_review_evidence') or ''),
                        'quality_status': str(row.get('source_quality_status') or 'UNREVIEWED'),
                        'content_hash': actual,
                        'normalized_hash': norm_hash(content),
                        'words': words,
                        'review_priority': 'high' if words < 1500 else 'normal',
                    })
                except (UnicodeError, ValueError, OSError, TypeError, json.JSONDecodeError) as ex:
                    errors.append({'location': identifier, 'error': str(ex)})
            # Only main text selected by manifest counts. Stray .txt files in
            # that folder (but not raw_sources/ or abstracts/) are flagged.
            actual_text_files = {
                p.relative_to(folder).as_posix() for p in folder.glob('*')
                if p.is_file() and p.suffix.lower() in {'.txt', '.md'}
            }
            if manifest == root / 'manifest.jsonl':
                # Root manifest paths may include subfolders; report stray
                # candidates underneath categories only, not unrelated nested
                # staging archives.
                actual_text_files = {
                    p.relative_to(folder).as_posix() for p in folder.rglob('*')
                    if p.is_file() and p.suffix.lower() in {'.txt', '.md'}
                    and not any(x.lower() in {'abstracts', 'raw_sources', 'excluded', 'archive', 'archives', 'snapshots'}
                                for x in p.relative_to(folder).parts[:-1])
                }
            for orphan in sorted(actual_text_files - manifest_paths):
                extras.append({'manifest': str(manifest), 'unlisted_text': orphan})
    return records, errors, extras


def collect_duplicate_issues(records: list[dict]) -> list[dict]:
    issues = []
    keys = ('canonical_source', 'source_id', 'content_hash', 'normalized_hash')
    for key in keys:
        lookup = defaultdict(list)
        for record in records:
            k = record[key]
            if k:
                lookup[k].append(record)
        for value, rows in sorted(lookup.items()):
            if len(rows) > 1:
                issues.append({'type': key, 'key': value,
                               'paths': [r['path'] for r in rows],
                               'source_ids': [r['source_id'] for r in rows]})
    return issues


def write_report(out: Path, records: list[dict], errors: list[dict], extras: list[dict],
                 issues: list[dict], expected: int) -> dict:
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing report directory: {out}')
    if any(out.resolve().is_relative_to(Path(r['batch'])) for r in records):
        raise ValueError('Output must be outside candidate batch directories')
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'candidate_documents_valid_hashes': len(records),
        'expected_documents': expected,
        'count_matches_expectation': len(records) == expected,
        'unique_sources': len({r['canonical_source'] for r in records}),
        'unique_content_hashes': len({r['content_hash'] for r in records}),
        'domain_counts': dict(sorted(Counter(r['domain'] for r in records).items())),
        'license_statuses': dict(sorted(Counter(r['license_status'] for r in records).items())),
        'quality_statuses': dict(sorted(Counter(r['quality_status'] for r in records).items())),
        'integrity_errors': errors,
        'duplicate_issues': issues,
        'unlisted_text_files': extras,
        'canonical_dataset_modified': False,
        'verified_factual_claims': False,
        'independently_verified_license_claims': False,
        'reviewer_identities_authenticated': False,
        'ready_for_training': False,
        'next_action': 'Complete real, documented source-specific license and scientific-quality review; then use the project admission tool. Do not edit audit output to claim approval.',
    }
    with tempfile.TemporaryDirectory(prefix='.expansformer_review_', dir=out.parent) as temp:
        work = Path(temp)
        (work / 'audit.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        headers = ['source_id','canonical_source','domain','title','publisher','source_url',
                   'path','words','license','license_status','license_review_evidence',
                   'quality_status','review_priority','content_hash']
        with (work / 'review_queue.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(sorted(records, key=lambda r: (r['domain'], r['title'], r['source_id'])))
        with (work / 'duplicate_review.json').open('w', encoding='utf-8') as f:
            json.dump(issues, f, indent=2)
            f.write('\n')
        (work / 'README.txt').write_text(
            'THIS IS A CANDIDATE REVIEW QUEUE, NOT AN APPROVED DATASET.\n'
            'No human approval, license validity, or factual correctness is established.\n'
            'Never change status labels without documented, authentic reviews.\n'
            'The canonical training dataset was not changed.\n', encoding='utf-8')
        os.rename(work, out)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', type=Path, default=Path('.'))
    parser.add_argument('--batch', action='append', type=Path, help='One or more staged batch roots; autodetected if omitted')
    parser.add_argument('--out', type=Path, default=Path('./review_queue_330'))
    parser.add_argument('--expected-count', type=int, default=330)
    args = parser.parse_args(argv)
    try:
        if args.batch:
            roots = args.batch
        else:
            roots = [args.base / 'topic_gap_batch_240_revised', args.base / 'topic_gap_human_expansion']
        records, errors, extras = load_records(roots)
        issues = collect_duplicate_issues(records)
        result = write_report(args.out, records, errors, extras, issues, args.expected_count)
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({k: result[k] for k in ('candidate_documents_valid_hashes', 'expected_documents',
              'count_matches_expectation', 'unique_sources','domain_counts', 'license_statuses',
              'quality_statuses')}, indent=2))
    print(f'Integrity errors: {len(errors)} | duplicate flags: {len(issues)} | unlisted text: {len(extras)}')
    print('Canonical training data was not modified. No sources were approved.')
    print(f'Review files: {args.out}')
    return 1 if errors or issues or extras or len(records) != args.expected_count else 0

if __name__ == '__main__':
    raise SystemExit(main())
