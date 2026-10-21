#!/usr/bin/env python3
"""Additive Expansformer v0.4 neural integration and disposable staged-corpus trainer.

Runs from the Expansformer_v03_Research project with --project ., without writing to
canonical expansformer_dataset/. Original five research modules are inspiration,
not executable dependencies. New trainable pathways are defined by v0.4 model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit, urlunsplit
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE / 'payload'
NEURAL_FILES = ('expansformer_v04_model.py', 'train_expansformer_v04.py',
                'evaluate_adaptive_v04.py', 'compare_runs_v04.py', 'tests/test_v04.py')
SCHEMA = 'expansformer-v04-experimental-corpus-v1'
CC_URI = re.compile(r'creativecommons\.org/licenses/(by(?:-nc(?:-nd|-sa)?|-nd|-sa)?)/(\d+(?:\.\d+)?)/?', re.I)
TEXT_CC_BY = re.compile(r'creative\s+commons\s+attribution(?:\s+(?:4\.0|3\.0)\s+international)?\s+license(?:\s*\(\s*cc\s*by\s*\))?', re.I)
TEXT_RESTRICTED = re.compile(r'\bCC\s*BY\s*[-–]?\s*(?:NC|ND|SA)\b|\bnon[-\s]?commercial\b|\bno[-\s]?derivatives\b', re.I)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_file(base: Path, rel: str) -> Path:
    r = Path(rel)
    if r.is_absolute() or not r.parts or '..' in r.parts or any(p in ('', '.') for p in r.parts):
        raise ValueError(f'Unsafe manifest path: {rel!r}')
    path = base / r
    if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f'Symlink or escaped path: {path}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def records(manifest: Path):
    if not manifest.is_file():
        return
    for i, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f'Invalid manifest object at {manifest}:{i}')
            yield item


def install(project: Path) -> dict:
    # Fail before changing anything if another implementation already exists.
    decisions = []
    for rel in NEURAL_FILES:
        source = safe_file(PAYLOAD, rel)
        dst = project / rel
        if dst.is_symlink():
            raise ValueError(f'Refusing symlink destination: {dst}')
        if dst.exists() and (not dst.is_file() or digest(dst.read_bytes()) != digest(source.read_bytes())):
            raise ValueError(f'Conflicting v0.4 file: {dst}. No files were changed.')
        decisions.append((source, dst, 'already_present' if dst.exists() else 'install'))
    for source, dst, action in decisions:
        if action != 'install':
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Preserve existing modules; exclusive creation avoids accidental overwrites.
        with dst.open('xb') as fp:
            fp.write(source.read_bytes())
    result = {'installed': sum(action == 'install' for _, _, action in decisions),
              'already_present': sum(action == 'already_present' for _, _, action in decisions),
              'files': [str(dst.relative_to(project)) for _, dst, _ in decisions]}
    print('Neural integration:', json.dumps(result, indent=2))
    return result


def normalize(url: str):
    p = urlsplit(url.strip())
    if p.scheme not in ('http', 'https') or not p.netloc:
        raise ValueError(f'Invalid article URL: {url!r}')
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip('/'), p.query, ''))


def license_from_jats(xml_bytes: bytes) -> tuple[bool, str]:
    # Parse local raw source and *all* license nodes; never infer a version from manifest claims.
    root = ET.fromstring(xml_bytes)
    licenses = [e for e in root.iter() if e.tag.split('}')[-1] == 'license']
    if not licenses:
        return False, 'XML_HAS_NO_LICENSE_NODE'
    all_text = []
    for lic in licenses:
        all_text.append(' '.join(' '.join(lic.itertext()).split()))
        all_text.extend(str(v) for child in lic.iter() for v in child.attrib.values())
    evidence = ' '.join(all_text)
    declared = CC_URI.findall(evidence)
    if declared:
        kinds = {kind.lower() for kind, _ in declared}
        if kinds != {'by'}:
            return False, 'RESTRICTED_OR_CONFLICTING_CC_URI'
        if TEXT_RESTRICTED.search(evidence):
            return False, 'CONFLICTING_RESTRICTION_TEXT'
        versions = {version for _, version in declared}
        return True, 'SOURCE_DECLARED_CC_BY_' + '_'.join(sorted(versions))
    if TEXT_RESTRICTED.search(evidence):
        return False, 'RESTRICTED_LICENSE_TEXT'
    if TEXT_CC_BY.search(evidence) or re.search(r'Creative Commons Attribution License\s*\(\s*CC\s+BY\s*\)', evidence, re.I):
        ver = re.search(r'Creative\s+Commons\s+Attribution\s+(\d+(?:\.\d+)?)\s+International\s+License', evidence, re.I)
        return True, 'SOURCE_DECLARED_CC_BY_' + ver.group(1) if ver else 'SOURCE_DECLARED_CC_BY_VERSION_UNSPECIFIED'
    return False, 'UNRESOLVED_LICENSE_TEXT'


def collect_eligible(project: Path):
    included, excluded = [], []
    m240 = project / 'topic_gap_batch_240_revised' / 'manifest.jsonl'
    if m240.is_file():
        for r in records(m240):
            # Minimal declared-rights filter. Staging remains unapproved.
            if r.get('source_type') != 'wikipedia_article' or not re.search(
                    r'creativecommons\.org/licenses/by-sa/4\.0/?', str(r.get('license_url', '')), re.I):
                excluded.append({'source_id': r.get('source_id'), 'reason': 'EXTERNAL_SOURCE_REQUIRES_SEPARATE_RIGHTS_REVIEW'})
                continue
            included.append((m240.parent, r, 'SOURCE_DECLARED_CC_BY_SA_4.0'))
    mhuman = project / 'topic_gap_human_expansion'
    for m in sorted(mhuman.glob('*/manifest.jsonl')) if mhuman.is_dir() else []:
        for r in records(m):
            sid = r.get('source_id')
            try:
                raw = safe_file(m.parent, str(r['raw_source_path'])).read_bytes()
                if r.get('raw_content_hash') and digest(raw) != r['raw_content_hash']:
                    raise ValueError('Raw XML hash mismatch')
                ok, status = license_from_jats(raw)
                if not ok:
                    excluded.append({'source_id': sid, 'reason': status})
                    continue
                included.append((m.parent, r, status))
            except (OSError, ET.ParseError, ValueError, KeyError) as exc:
                excluded.append({'source_id': sid, 'reason': f'RAW_XML_UNVERIFIED: {exc}'})
    if not included:
        raise ValueError('No usable staged candidate documents; check topic_gap_* manifests')
    seen_urls, seen_hashes, prepared = set(), set(), []
    for base, r, declaration in included:
        path = safe_file(base, str(r['path']))
        raw = path.read_bytes()
        source_hash = digest(raw)
        if source_hash != r.get('content_hash'):
            raise ValueError(f'Extracted text hash mismatch: {path}')
        url = normalize(str(r.get('source_url') or r.get('url') or ''))
        if url in seen_urls:
            raise ValueError(f'Duplicate normalized document URL: {url}')
        if source_hash in seen_hashes:
            raise ValueError(f'Duplicate source content hash: {path}')
        seen_urls.add(url)
        seen_hashes.add(source_hash)
        prepared.append({'source_id': str(r['source_id']), 'domain': str(r['domain']),
                         'title': str(r.get('title', path.stem)), 'url': url,
                         'license': declaration, 'source_license_status': r.get('license_status', 'UNREVIEWED'),
                         'source_quality_status': r.get('source_quality_status', 'UNREVIEWED'),
                         'sha256': source_hash, 'text_path': path})
    prepared.sort(key=lambda r: (r['domain'], r['source_id']))
    return prepared, excluded


def prepare(project: Path, destination: Path):
    prepared, excluded = collect_eligible(project)
    # Corpus at a separate path, never inside canonical dataset tree.
    canonical = project / 'expansformer_dataset'
    if destination.resolve() == canonical.resolve() or destination.resolve().is_relative_to(canonical.resolve()):
        raise ValueError('Refusing to write to canonical expansformer_dataset')
    manifest = []
    for i, r in enumerate(prepared):
        rel = f"data/{i:04d}_{r['source_id']}.txt"
        manifest.append({'path': rel, 'title': r['title'], 'domain': r['domain'],
                         'url': r['url'], 'license': r['license'], 'sha256': r['sha256'],
                         'source_id': r['source_id'], 'source_quality_status': r['source_quality_status'],
                         'source_license_status': r['source_license_status'],
                         'rights_independently_verified': False})
    snapshot = {'schema': SCHEMA, 'documents': len(prepared), 'excluded': excluded,
                'warning': 'Isolated unapproved research candidates; NOT a canonical dataset or rights clearance.',
                'sources': manifest}
    snapshot_hash = digest(json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode())
    if destination.exists():
        report_path = destination / 'experiment_provenance.json'
        if not report_path.is_file():
            raise FileExistsError(f'Destination exists without experiment manifest: {destination}')
        old = json.loads(report_path.read_text())
        if old.get('snapshot_hash') != snapshot_hash:
            raise ValueError('Staged data selection changed; use a NEW --corpus-out directory')
        for entry in manifest:
            if digest(safe_file(destination, entry['path']).read_bytes()) != entry['sha256']:
                raise ValueError('Cached research corpus was modified; use a new directory')
        print(f'Existing staged corpus verified: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.expansformer-stage-', dir=str(destination.parent)) as tmp:
            stage = Path(tmp)
            (stage / 'data').mkdir()
            for entry, original in zip(manifest, prepared):
                shutil.copyfile(original['text_path'], stage / entry['path'])
            (stage / 'manifest.jsonl').write_text(''.join(json.dumps(m, ensure_ascii=False, sort_keys=True) + '\n' for m in manifest), encoding='utf-8')
            (stage / 'experiment_provenance.json').write_text(json.dumps(snapshot | {'snapshot_hash': snapshot_hash}, ensure_ascii=False, indent=2), encoding='utf-8')
            os.rename(stage, destination)
    print('Research corpus prepared:', destination)
    print('Documents selected:', len(prepared), '| held for other source/rights conditions:', len(excluded))
    print('Domains:', dict(sorted(Counter(r['domain'] for r in prepared).items())))
    return len(prepared)


def model_command(project: Path, corpus: Path, a):
    cmd = [sys.executable, str(project / 'train_expansformer_v04.py'), 'train', '--data', str(corpus),
           '--out', str((project / a.out).resolve()), '--tokenizer', 'bpe',
           '--vocab-size', str(a.vocab_size), '--n-layers', '4', '--n-heads', '4',
           '--embed-dim', '256', '--context-length', str(a.context_length),
           '--batch-size', str(a.batch_size), '--grad-accum', '2',
           '--steps', str(a.steps), '--stop-after', str(a.stop_after),
           '--eval-every', '25', '--eval-windows', '32', '--save-every', '25',
           '--warmup-steps', '30', '--moe-experts', '4', '--moe-top-k', '2',
           '--auxiliary-exits', '--memory-slots', '16', '--memory-every', '2',
           '--next-token-heads', '--attention', 'sdpa', '--strict-provenance']
    ckpt = project / a.out / 'latest.pt'
    if ckpt.is_file():
        cmd.extend(['--resume', str(ckpt.resolve())])
    return cmd


def main():
    p = argparse.ArgumentParser(description='Install v0.4 neural modules into v0.3 and launch a bounded research training run')
    p.add_argument('action', choices=('install', 'prepare', 'train', 'smoke'))
    p.add_argument('--project', type=Path, default=Path.cwd())
    p.add_argument('--allow-unreviewed-experiment', action='store_true',
                   help='Required for staged candidate training; not approval or licensing clearance')
    p.add_argument('--corpus-out', type=Path, default=Path('runs/v04_staged_research_data'))
    p.add_argument('--out', default='runs/v04_integrated_pilot')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--stop-after', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--vocab-size', type=int, default=512)
    p.add_argument('--context-length', type=int, default=128)
    p.add_argument('--dry-run', action='store_true', help='Print the command without launching PyTorch training')
    a = p.parse_args()
    project = a.project.resolve()
    if not (project / 'train_expansformer_v03.py').is_file():
        p.error('--project must contain the v0.3 training script')
    if a.action in ('install', 'train', 'smoke'):
        install(project)
    if a.action == 'install':
        return
    if a.action == 'smoke':
        subprocess.run([sys.executable, 'train_expansformer_v04.py', 'smoke'], cwd=project, check=True)
        return
    if not a.allow_unreviewed_experiment:
        p.error('Staged corpus is unreviewed. Add --allow-unreviewed-experiment for an isolated exploratory run.')
    if a.steps < 1 or not 1 <= a.stop_after <= a.steps or a.batch_size < 1:
        p.error('Invalid steps/stop-after/batch-size')
    corpus = (project / a.corpus_out).resolve()
    count = prepare(project, corpus)
    if a.action == 'prepare':
        return
    if count < 2:
        p.error('Need at least two documents for document-level holdout')
    cmd = model_command(project, corpus, a)
    print('Starting integrated training command:', ' '.join(map(str, cmd)), flush=True)
    if a.dry_run:
        print('Dry run: no model training launched.')
        return
    subprocess.run(cmd, cwd=project, check=True)

if __name__ == '__main__':
    main()
