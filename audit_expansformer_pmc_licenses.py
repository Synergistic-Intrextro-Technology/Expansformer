#!/usr/bin/env python3
"""Read-only audit of staged Europe PMC JATS license declarations.

This tool does not decide lawful use, authenticate evidence, change manifests,
or approve any training document. Source declarations are NOT rights clearance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

CC_LICENSE_URL = re.compile(
    r'https?://(?:www\.)?creativecommons\.org/licenses/([a-z]+(?:-[a-z]+)*)/(\d+(?:\.\d+)?)/?',
    re.I,
)
HREF_WORD = re.compile(r'creat(?:ive\s+commons|ivecommons)', re.I)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def local_name(tag: str) -> str:
    return tag.rsplit('}', 1)[-1].lower()


def extract_xml_license(raw_path: Path) -> dict:
    root = ET.parse(raw_path).getroot()
    # JATS licensing is article-level front matter; don't mine references/citations.
    article_meta = [node for node in root.iter() if local_name(node.tag) == 'article-meta']
    parents = article_meta[:1] if article_meta else [root]
    declarations = []
    for scope in parents:
        for element in scope.iter():
            if local_name(element.tag) != 'license':
                continue
            urls = []
            for descendant in element.iter():
                for attr, val in descendant.attrib.items():
                    if local_name(attr) in ('href', 'uri') and 'creativecommons.org' in val.lower():
                        urls.append(val.strip())
                for text in (descendant.text, descendant.tail):
                    if text:
                        urls.extend(m.group() for m in CC_LICENSE_URL.finditer(text))
            # Preserve actual node declaration text for conflicting license markers.
            words = ' '.join(' '.join(element.itertext()).split())
            declarations.append({'urls': list(dict.fromkeys(urls)), 'text': words[:3000]})
    return {'declarations': declarations, 'raw_license_count': len(declarations)}


def classify_licenses(data: dict) -> tuple[str, str]:
    declarations = data['declarations']
    if not declarations:
        return 'NO_LICENSE_NODE', 'No JATS license node found in article metadata'

    versions = set()
    for node in declarations:
        for url in node['urls']:
            match = CC_LICENSE_URL.search(url)
            if match:
                versions.add((match.group(1).lower(), match.group(2)))

    if not versions:
        return 'UNRESOLVED_DECLARATION', 'License node exists but no parseable CC license URL'
    families = {kind for kind, _ in versions}
    if families == {'by'}:
        if len(versions) > 1:
            return 'MULTIPLE_CC_BY_VERSIONS', 'Multiple CC BY versions declared; inspect manually'
        return 'SOURCE_DECLARED_CC_BY', 'JATS metadata declares plain CC BY; not independently cleared'
    if 'by' in families:
        return 'CONFLICTING_LICENSES', 'Plain CC BY and another CC license declared'
    return 'NON_CC_BY_VARIANT', ', '.join(f'CC {kind.upper()} {version}' for kind, version in sorted(versions))


def safe_relative_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Unsafe relative path: {relative!r}')
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes manifest folder: {relative!r}')
    return path


def audit_one(manifest: Path, record: dict) -> dict:
    result = {
        'pmcid': str(record.get('pmcid', '')),
        'source_id': str(record.get('source_id', '')),
        'domain': str(record.get('domain', '')),
        'title': str(record.get('title', '')),
        'source_url': str(record.get('source_url', '')),
        'manifest': str(manifest),
        'status': '',
        'detail': '',
        'xml_license_urls': '',
        'manifest_license_url': str(record.get('license_url', '')),
        'manifest_xml_license_mismatch': False,
        'integrity_error': '',
    }
    try:
        raw = safe_relative_path(manifest.parent, record['raw_source_path'])
        text = safe_relative_path(manifest.parent, record['path'])
        for path, key in ((raw, 'raw_content_hash'), (text, 'content_hash')):
            if not path.is_file():
                raise ValueError(f'Missing {path}')
            expected = record.get(key)
            if not expected or sha256_file(path) != expected:
                raise ValueError(f'Invalid or missing {key} for {path}')
        xml_data = extract_xml_license(raw)
        result['status'], result['detail'] = classify_licenses(xml_data)
        urls = sorted({url for node in xml_data['declarations'] for url in node['urls']})
        result['xml_license_urls'] = ' ; '.join(urls)
        # Manifest URL can be a generic hardcoded CC BY default. Flag contradictions.
        if urls and result['manifest_license_url']:
            raw_types = {(m.group(1).lower(), m.group(2)) for u in urls if (m := CC_LICENSE_URL.search(u))}
            manifest_match = CC_LICENSE_URL.search(result['manifest_license_url'])
            if manifest_match and raw_types and (manifest_match.group(1).lower(), manifest_match.group(2)) not in raw_types:
                result['manifest_xml_license_mismatch'] = True
    except (OSError, ValueError, KeyError, ET.ParseError) as exc:
        result['status'] = 'INTEGRITY_OR_XML_ERROR'
        result['detail'] = f'{type(exc).__name__}: {exc}'
        result['integrity_error'] = result['detail']
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'), help='Expansformer project directory')
    parser.add_argument('--out', type=Path, default=Path('pmc_license_review'), help='NEW output folder (never overwritten)')
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    manifests = sorted((root / 'topic_gap_human_expansion').glob('*/manifest.jsonl'))
    if not manifests:
        parser.error(f'No topic_gap_human_expansion category manifests found beneath {root}')
    out = args.out.expanduser().resolve()
    if out.exists():
        parser.error(f'Refusing to overwrite an existing report directory: {out}')
    if out == root or root in out.parents and ('topic_gap_human_expansion' in out.parts):
        parser.error('Refusing to create reports inside the raw staging directory')

    rows, issues = [], []
    for manifest in manifests:
        try:
            for line_no, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    rows.append(audit_one(manifest, obj))
                except (ValueError, TypeError) as exc:
                    issues.append(f'{manifest}:{line_no}: {exc}')
        except OSError as exc:
            issues.append(f'Cannot read {manifest}: {exc}')
    counts = Counter(row['status'] for row in rows)
    mismatches = [row['source_id'] for row in rows if row['manifest_xml_license_mismatch']]
    report = {
        'documents_scanned': len(rows),
        'category_manifests': [str(p) for p in manifests],
        'source_declaration_statuses': dict(sorted(counts.items())),
        'manifest_xml_license_mismatch_count': len(mismatches),
        'manifest_xml_mismatch_source_ids': mismatches,
        'manifest_errors': issues,
        'training_source_approval': 'NOT_EVALUATED',
        'rights_independently_cleared': False,
        'factual_quality_verified': False,
        'note': 'Only article metadata license declarations checked. Not legal clearance or human quality review.'
    }
    out.mkdir(parents=True)
    (out / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (out / 'documents.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ['pmcid','status'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('Detailed findings:', out / 'documents.csv')
    return 0 if not issues and not counts.get('INTEGRITY_OR_XML_ERROR', 0) else 2

if __name__ == '__main__':
    sys.exit(main())
