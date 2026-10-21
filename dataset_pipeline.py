#!/usr/bin/env python3
"""Local-first provenance, partitioning, and quality tooling for Expansformer data.

No network access or truth/legal verification is performed here. See
expansformer_dataset/README.md for the input contract and evidence limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

LAYOUT = (
    'raw/encyclopedic', 'raw/scientific', 'raw/engineering', 'raw/mathematics', 'raw/other_domains',
    'curated/pretraining/train', 'curated/pretraining/validation',
    'curated/reasoning', 'curated/epistemic_training',
    'evaluation/language', 'evaluation/factuality', 'evaluation/reasoning',
    'evaluation/generalization', 'evaluation/uncertainty_calibration', 'evaluation/abstention',
    'evidence/source_records', 'manifests', 'sources',
)
EVALUATION_TASK_TO_TRACK = {
    'language_modeling': 'language', 'factuality': 'factuality', 'reasoning': 'reasoning',
    'generalization': 'generalization', 'uncertainty_calibration': 'uncertainty_calibration',
    'abstention': 'abstention',
}
TRACKS = set(EVALUATION_TASK_TO_TRACK.values())
LICENSE_STATES = {'SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED', 'INDEPENDENTLY_REVIEWED', 'RESTRICTED', 'UNKNOWN'}
QUALITY_RUBRIC = {'source_authority', 'editorial_process', 'domain_relevance', 'factual_support', 'transparency'}
SOURCE_FIELDS = ('source_id', 'title', 'publisher', 'source_url', 'publication_date', 'license_id', 'license_url',
                 'license_status', 'domain', 'source_type', 'content_hash')


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().casefold()


def quality(text: str) -> dict:
    words = re.findall(r"\b[\w'-]+\b", text.casefold())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_counts = Counter(lines)
    return {
        'characters': len(text), 'words': len(words),
        'unique_word_fraction': len(set(words)) / len(words) if words else 0.0,
        'repeated_line_fraction': max(line_counts.values()) / len(lines) if len(lines) > 1 else 0.0,
    }


def preflight(input_dir: Path, metadata_file: Path) -> dict:
    """Check an incoming document batch without importing or accepting it.

    The report validates file/manifest correspondence and byte hashes, then
    emits screening diagnostics for human review. It deliberately does not
    determine legal rights, factual accuracy, or source quality.
    """
    if input_dir.is_symlink() or not input_dir.is_dir():
        raise ValueError('Input must be a real directory, not a symlink')
    records = read_jsonl(metadata_file)
    problems = []
    if not records:
        problems.append('Metadata JSONL contains no records')
    seen_paths = set()
    seen_ids = set()
    seen_hashes = defaultdict(list)
    reports = []
    disk_paths = set()
    for path in input_dir.rglob('*'):
        if path.is_symlink():
            problems.append(f'Input contains symlink: {path.relative_to(input_dir).as_posix()}')
        elif path.is_file() and path.suffix.lower() in {'.txt', '.md'}:
            disk_paths.add(path.relative_to(input_dir).as_posix())
    for row in records:
        label = str(row.get('path', '<unknown>'))
        missing = [key for key in SOURCE_FIELDS if not row.get(key)]
        if missing:
            problems.append(f'{label}: missing required metadata fields {missing}')
            continue
        nontext = [key for key in SOURCE_FIELDS if not isinstance(row.get(key), str)]
        if nontext:
            problems.append(f'{label}: metadata fields must be strings: {nontext}')
            continue
        if row['license_status'] not in LICENSE_STATES:
            problems.append(f'{label}: unsupported license_status {row["license_status"]!r}')
            continue
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', row['domain']):
            problems.append(f'{label}: invalid domain label {row["domain"]!r}')
            continue
        try:
            rel = safe_relative(row['path']).as_posix()
        except (TypeError, ValueError) as exc:
            problems.append(f'{label}: {exc}')
            continue
        if rel in seen_paths:
            problems.append(f'{rel}: duplicate metadata path')
            continue
        seen_paths.add(rel)
        if row['source_id'] in seen_ids:
            problems.append(f'{rel}: duplicate source_id {row["source_id"]}')
            continue
        seen_ids.add(row['source_id'])
        if PurePosixPath(rel).suffix.lower() not in {'.txt', '.md'}:
            problems.append(f'{rel}: only UTF-8 .txt/.md documents are supported')
            continue
        path = input_dir.joinpath(*PurePosixPath(rel).parts)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(input_dir.resolve()):
            problems.append(f'{rel}: file is absent, a symlink, or escapes input directory')
            continue
        raw = path.read_bytes()
        actual_hash = sha256(raw)
        claimed_hash = str(row.get('content_hash', '')).removeprefix('sha256:').lower()
        if actual_hash != claimed_hash:
            problems.append(f'{rel}: content hash mismatch')
            continue
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError:
            problems.append(f'{rel}: document is not UTF-8')
            continue
        if not content.strip():
            problems.append(f'{rel}: document is empty')
            continue
        diagnostics = quality(content)
        words = diagnostics['words']
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        flags = []
        if words < 250:
            flags.append('short_under_250_words')
        if len(lines) > 1 and diagnostics['repeated_line_fraction'] > 0.10:
            flags.append('repeated_lines_over_10_percent')
        heading_markers = len(re.findall(r'(?m)^\s*={2,}[^\n]*={2,}\s*$', content))
        emphasis_markers = len(re.findall(r"'{3,}", content))
        formula_markers = len(re.findall(r'\\(?:displaystyle|begin\{)', content))
        if formula_markers:
            flags.append('possible_formula_markup')
        if diagnostics['unique_word_fraction'] < 0.20:
            flags.append('low_unique_word_fraction_review_manually')
        normalized_hash = sha256(normalize(content).encode('utf-8'))
        seen_hashes[normalized_hash].append(rel)
        reports.append({
            'path': rel,
            'source_id': row['source_id'],
            'title': row['title'],
            'domain': row['domain'],
            'content_hash': actual_hash,
            'license_status': row['license_status'],
            'source_quality_status': row.get('source_quality_status', 'UNREVIEWED'),
            **diagnostics,
            'wiki_heading_marker_count': heading_markers,
            'wiki_emphasis_marker_count': emphasis_markers,
            'formula_marker_count': formula_markers,
            'flags': flags,
        })
    for path in sorted(disk_paths - seen_paths):
        problems.append(f'{path}: text file has no metadata row')
    for path in sorted(seen_paths - disk_paths):
        problems.append(f'{path}: metadata row has no matching text file')
    duplicate_groups = [sorted(paths) for paths in seen_hashes.values() if len(paths) > 1]
    return {
        'status': 'candidate_screen_only',
        'valid': not problems,
        'input_documents': len(reports),
        'input_text_files': len(disk_paths),
        'total_words': sum(report['words'] for report in reports),
        'domains': dict(sorted(Counter(report['domain'] for report in reports).items())),
        'license_statuses': dict(sorted(Counter(report['license_status'] for report in reports).items())),
        'source_quality_statuses': dict(sorted(Counter(report['source_quality_status'] for report in reports).items())),
        'exact_normalized_duplicate_groups': duplicate_groups,
        'flagged_documents': sum(bool(report['flags']) for report in reports),
        'documents': reports,
        'problems': problems,
        'limits': [
            'screening flags are diagnostic and do not automatically reject or accept text',
            'this command does not verify licenses, source quality, factual accuracy, or semantic duplication',
            'preflight does not import files or change the dataset',
        ],
    }


def prepare_wikipedia_batch(input_dir: Path, metadata_file: Path, output_dir: Path) -> dict:
    """Create a traceable, minimally normalized copy of Wikipedia extracts."""
    source = input_dir.resolve()
    output = output_dir.resolve()
    if output.exists():
        raise ValueError(f'Output already exists; refusing to overwrite: {output}')
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Input and output directories must not contain one another')
    screen = preflight(source, metadata_file)
    if not screen['valid']:
        raise ValueError(f'Input preflight failed: {screen["problems"][:10]}')
    rows = read_jsonl(metadata_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}.staging-', dir=output.parent))
    prepared = []
    changed = 0
    words_before = words_after = 0
    try:
        for row in rows:
            rel = safe_relative(row['path'])
            source_path = source.joinpath(*rel.parts)
            original = source_path.read_text(encoding='utf-8')
            original_hash = sha256(original.encode('utf-8'))
            cleaned = unicodedata.normalize('NFC', original)
            cleaned_lines = []
            for line in cleaned.splitlines():
                heading = re.match(r'^\s*(={2,})\s*(.*?)\s*\1\s*$', line)
                if heading:
                    line = '## ' + heading.group(2).strip()
                line = re.sub(r"'{2,5}", '', line)
                line = re.sub(r'[ \t]+', ' ', line).strip()
                cleaned_lines.append(line)
            cleaned = re.sub(r'\n{3,}', '\n\n', '\n'.join(cleaned_lines)).strip() + '\n'
            destination = stage.joinpath(*rel.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(cleaned, encoding='utf-8')
            derived_hash = sha256(cleaned.encode('utf-8'))
            changed += original != cleaned
            words_before += quality(original)['words']
            words_after += quality(cleaned)['words']
            derived_row = dict(row)
            derived_row['source_content_hash'] = 'sha256:' + original_hash
            derived_row['content_hash'] = 'sha256:' + derived_hash
            derived_row['processing_history'] = list(row.get('processing_history', [])) + [{
                'operation': 'conservative_wikipedia_text_normalization',
                'tool': 'dataset_pipeline.py prepare-wikipedia',
                'input_content_hash': 'sha256:' + original_hash,
                'output_content_hash': 'sha256:' + derived_hash,
                'operations': ['Unicode NFC', 'section heading normalization',
                               'Wikipedia emphasis marker removal', 'horizontal whitespace normalization'],
                'at': datetime.now(timezone.utc).isoformat(),
            }]
            prepared.append(derived_row)
        (stage / 'manifest.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in prepared),
            encoding='utf-8')
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        'output': str(output),
        'documents': len(prepared),
        'documents_changed': changed,
        'words_before': words_before,
        'words_after': words_after,
        'source_content_hashes_preserved': True,
        'license_and_quality_status_preserved_without_acceptance': True,
        'formula_or_equation_content_transformed': False,
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path}:{line_no}: invalid JSON: {exc}') from exc
        if not isinstance(row, dict):
            raise ValueError(f'{path}:{line_no}: each record must be a JSON object')
        rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(''.join(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n' for x in rows), encoding='utf-8')
    os.replace(temp, path)


def safe_relative(value: str) -> PurePosixPath:
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or any(x in ('..', '.') for x in p.parts) or '\\' in value:
        raise ValueError(f'Unsafe relative path: {value!r}')
    return p


def init(root: Path) -> None:
    if root.exists() and not root.is_dir():
        raise ValueError(f'Root exists and is not a directory: {root}')
    root.mkdir(parents=True, exist_ok=True)
    for name in LAYOUT:
        (root / name).mkdir(parents=True, exist_ok=True)
    config = root / 'dataset_config.json'
    if not config.exists():
        config.write_text(json.dumps({
            'dataset': 'Expansformer Knowledge & Reasoning Dataset', 'schema_version': 4,
            'created_at': datetime.now(timezone.utc).isoformat(), 'network_access': 'disabled',
            'training_evidence_separation': True,
            'reasoning_task_types': ['causal', 'counterfactual', 'tradeoff', 'cross_domain_synthesis', 'experimental_problem_solving'],
            'reasoning_example_required_fields': ['id', 'task_type', 'prompt', 'constraints', 'answer', 'rationale', 'source_ids'],
            'cross_domain_synthesis_required_fields': ['domain_tags', 'shared_mechanism', 'mechanism_evidence'],
            'experimental_problem_solving_required_fields': ['hypothesis', 'experiment_design', 'observed_outcome', 'interpretation', 'failed_prediction', 'model_revision'],
            'epistemic_example_required_fields': ['id', 'prompt', 'supported_claims', 'disputed_claims', 'assumptions', 'unknowns', 'conclusion', 'source_ids'],
            'evaluation_record_required_fields': ['id', 'task_type', 'input', 'expected', 'scoring', 'source_ids'],
            'evaluation_task_types': ['language_modeling', 'factuality', 'reasoning', 'generalization', 'uncertainty_calibration', 'abstention'],
            'evaluation_task_tracks': EVALUATION_TASK_TO_TRACK,
            'evaluation_scoring_required_fields': ['metric', 'protocol'],
            'license_manifest_extra_fields': ['license_terms_url', 'license_scope', 'additional_license_urls'],
            'source_review_record_fields': ['review_id', 'source_id', 'reviewer_id', 'reviewer_affiliation', 'relationship_to_source', 'reviewed_at', 'method', 'evidence_refs', 'ratings', 'decision'],
            'source_quality_rubric': sorted(QUALITY_RUBRIC),
            'source_review_threshold': 'Two accepting reviewers, distinct reviewer IDs and affiliations, self-declared independence, complete ratings of at least 1, and evidence references',
            'status_semantics': {'INDEPENDENTLY_REVIEWED': 'importer-supplied review assertion with evidence reference; not authenticated by this tool',
                                 'SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED': 'metadata claim supplied by importer',
                                 'UNKNOWN': 'rights status not established', 'RESTRICTED': 'do not train or redistribute without separate authorization'},
            'limits': ['hashes prove byte identity only', 'quality diagnostics do not establish accuracy',
                       'exact duplicate scan does not detect paraphrase or benchmark contamination']
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    else:
        config_data = json.loads(config.read_text(encoding='utf-8'))
        config_data['schema_version'] = 4
        config_data.pop('source_quality_review_fields', None)
        config_data.setdefault('source_review_record_fields', ['review_id', 'source_id', 'reviewer_id', 'reviewer_affiliation', 'relationship_to_source', 'reviewed_at', 'method', 'evidence_refs', 'ratings', 'decision'])
        config_data.setdefault('source_quality_rubric', sorted(QUALITY_RUBRIC))
        config_data.setdefault('source_review_threshold', 'Two accepting reviewers, distinct reviewer IDs and affiliations, self-declared independence, complete ratings, evidence references, and rubric scores of at least 1')
        config_data.setdefault('evaluation_task_types', ['language_modeling', 'factuality', 'reasoning', 'generalization', 'uncertainty_calibration', 'abstention'])
        config_data.setdefault('evaluation_task_tracks', EVALUATION_TASK_TO_TRACK)
        config_data.setdefault('evaluation_scoring_required_fields', ['metric', 'protocol'])
        config_data.setdefault('license_manifest_extra_fields', ['license_terms_url', 'license_scope', 'additional_license_urls'])
        config_data.setdefault('status_semantics', {})['INDEPENDENTLY_REVIEWED'] = 'importer-supplied review assertion with evidence reference; not authenticated by this tool'
        config.write_text(json.dumps(config_data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name in ('sources.jsonl', 'licenses.jsonl', 'quality_reports.jsonl', 'quality_reviews.jsonl', 'documents.jsonl', 'evaluation.jsonl',
                 'reasoning_examples.jsonl', 'epistemic_examples.jsonl'):
        p = root / 'manifests' / name
        p.touch(exist_ok=True)


def source_inventory(root: Path) -> list[dict]:
    return read_jsonl(root / 'manifests' / 'sources.jsonl')


def import_documents(root: Path, input_dir: Path, metadata_file: Path, partition: str,
                     domain: str | None, track: str | None) -> int:
    if partition not in {'training', 'evaluation'}:
        raise ValueError('partition must be training or evaluation')
    if partition == 'evaluation' and track not in TRACKS:
        raise ValueError(f'evaluation track must be one of {sorted(TRACKS)}')
    if partition == 'training' and track:
        raise ValueError('--track applies only to evaluation imports')
    if not root.is_dir():
        raise ValueError(f'Dataset root is missing; run init first: {root}')
    required_manifests = ('sources.jsonl', 'licenses.jsonl', 'quality_reports.jsonl',
                         'documents.jsonl', 'evaluation.jsonl')
    if any(not (root / 'manifests' / name).is_file() for name in required_manifests):
        raise ValueError(f'Dataset scaffold is incomplete; run init first: {root}')
    if input_dir.is_symlink() or not input_dir.is_dir():
        raise ValueError('Input must be a real directory, not a symlink')
    records = read_jsonl(metadata_file)
    if not records:
        raise ValueError('Metadata JSONL contains no records')
    existing = source_inventory(root)
    existing_paths = {x['path'] for x in existing}
    existing_ids = {x['source_id'] for x in existing}
    train_keys = {x['normalized_hash'] for x in existing if x['partition'] == 'training'}
    eval_keys = {x['normalized_hash'] for x in existing if x['partition'] == 'evaluation'}
    train_groups = {str(x.get('source_group') or x.get('source_id') or x['path'])
                    for x in existing if x['partition'] == 'training'}
    eval_groups = {str(x.get('source_group') or x.get('source_id') or x['path'])
                   for x in existing if x['partition'] == 'evaluation'}
    pending = []
    seen_hashes = set()
    for row in records:
        missing = [k for k in SOURCE_FIELDS if not row.get(k)]
        if missing:
            raise ValueError(f"Metadata for {row.get('path', '<unknown>')} missing required fields: {missing}")
        nontext = [field for field in SOURCE_FIELDS if not isinstance(row.get(field), str)]
        if nontext:
            raise ValueError(f"Metadata fields must be strings for {row.get('path', '<unknown>')}: {nontext}")
        if row['license_status'] not in LICENSE_STATES:
            raise ValueError(f"Unsupported license_status for {row['path']}")
        if row['source_id'] in existing_ids:
            raise ValueError(f"Duplicate source_id: {row['source_id']}")
        existing_ids.add(row['source_id'])
        if row['license_status'] != 'INDEPENDENTLY_REVIEWED':
            raise ValueError(f"Dataset admission requires source-specific INDEPENDENTLY_REVIEWED license status: {row['path']}")
        if not row.get('license_review_evidence'):
            raise ValueError(f"INDEPENDENTLY_REVIEWED requires license_review_evidence: {row['path']}")
        row['source_quality_status'] = 'UNREVIEWED'
        if row['license_status'] in {'UNKNOWN', 'RESTRICTED'} and partition == 'training':
            raise ValueError(f"Refusing training import with {row['license_status']} rights status: {row['path']}")
        rel = safe_relative(str(row['path']))
        if rel.suffix.lower() not in {'.txt', '.md'}:
            raise ValueError(f'Only UTF-8 .txt/.md documents are supported: {rel}')
        src = input_dir.joinpath(*rel.parts)
        if src.is_symlink() or not src.is_file() or not src.resolve().is_relative_to(input_dir.resolve()):
            raise ValueError(f'Input document is absent or escapes input directory: {rel}')
        raw = src.read_bytes()
        actual = sha256(raw)
        claimed = str(row['content_hash']).removeprefix('sha256:').lower()
        if claimed != actual:
            raise ValueError(f'Content hash mismatch for {rel}: metadata={claimed}, actual={actual}')
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError(f'Document is not UTF-8: {rel}') from exc
        if not content.strip():
            raise ValueError(f'Empty document: {rel}')
        normalized_hash = sha256(normalize(content).encode('utf-8'))
        if normalized_hash in seen_hashes:
            raise ValueError(f'Duplicate content within import batch: {rel}')
        seen_hashes.add(normalized_hash)
        if partition == 'training' and normalized_hash in eval_keys:
            raise ValueError(f'Training content overlaps existing evaluation document: {rel}')
        if partition == 'evaluation' and normalized_hash in train_keys:
            raise ValueError(f'Evaluation content overlaps existing training document: {rel}')
        if normalized_hash in (train_keys | eval_keys):
            raise ValueError(f'Duplicate content already present in dataset: {rel}')
        source_group = str(row.get('source_group') or row['source_id'])
        if partition == 'training' and source_group in eval_groups:
            raise ValueError(f'Training source_group overlaps existing evaluation: {source_group}')
        if partition == 'evaluation' and source_group in train_groups:
            raise ValueError(f'Evaluation source_group overlaps existing training: {source_group}')
        used_domain = domain or str(row['domain'])
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', used_domain):
            raise ValueError(f'Invalid domain label: {used_domain!r}')
        basename = f'{actual[:16]}_{re.sub(r"[^a-zA-Z0-9_-]+", "_", rel.stem)[:80]}.txt'
        if partition == 'training':
            category = str(row.get('raw_category') or used_domain.lower())
            if category not in {'encyclopedic', 'scientific', 'engineering', 'mathematics', 'other_domains'}:
                category = 'other_domains'
            destination_rel = PurePosixPath('raw') / category / used_domain / basename
        else:
            destination_rel = PurePosixPath('evaluation') / str(track) / basename
        dest_key = destination_rel.as_posix()
        if dest_key in existing_paths:
            raise ValueError(f'Destination collision: {dest_key}')
        item = dict(row)
        item.update({'path': dest_key, 'original_path': rel.as_posix(), 'content_hash': actual,
                     'normalized_hash': normalized_hash, 'domain': used_domain, 'partition': partition,
                     'source_group': source_group,
                     'track': track, 'license_status': row['license_status'],
                     'imported_at': datetime.now(timezone.utc).isoformat(),
                     'processing_history': list(row.get('processing_history', [])) + [{'operation': 'utf8_import', 'tool': 'dataset_pipeline.py', 'at': datetime.now(timezone.utc).isoformat()}],
                     'quality_diagnostics': quality(content)})
        pending.append((src, root / dest_key, item))
    # Publish staged files first; publish manifest last so an interrupted copy
    # cannot make incomplete sources appear as accepted dataset records.
    created = []
    created_records = []
    manifest_names = ['licenses.jsonl', 'quality_reports.jsonl',
                      'evaluation.jsonl' if partition == 'evaluation' else 'documents.jsonl',
                      'sources.jsonl']
    manifest_paths = [root / 'manifests' / name for name in manifest_names]
    previous_manifests = {path: path.read_bytes() for path in manifest_paths}
    try:
        for src, dest, _ in pending:
            if dest.exists():
                raise FileExistsError(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            created.append(dest)
        for _, _, item in pending:
            record_path = root / 'evidence' / 'source_records' / (sha256(item['source_id'].encode()) + '.json')
            if record_path.exists():
                raise FileExistsError(f'Source record already exists: {record_path.name}')
            record_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            created_records.append(record_path)
        write_jsonl_atomic(root / 'manifests' / 'licenses.jsonl', read_jsonl(root / 'manifests' / 'licenses.jsonl') +
                           [{'path': x[2]['path'], 'license_id': x[2]['license_id'], 'license_url': x[2]['license_url'],
                             'license_status': x[2]['license_status'], 'publisher': x[2]['publisher'],
                             'review_evidence': x[2].get('license_review_evidence', None),
                             'license_terms_url': x[2].get('license_terms_url'),
                             'license_scope': x[2].get('license_scope'),
                             'additional_license_urls': x[2].get('additional_license_urls', [])} for x in pending])
        write_jsonl_atomic(root / 'manifests' / 'quality_reports.jsonl', read_jsonl(root / 'manifests' / 'quality_reports.jsonl') +
                           [{'path': x[2]['path'], 'content_hash': x[2]['content_hash'], **x[2]['quality_diagnostics']} for x in pending])
        manifest_name = 'evaluation.jsonl' if partition == 'evaluation' else 'documents.jsonl'
        write_jsonl_atomic(root / 'manifests' / manifest_name, read_jsonl(root / 'manifests' / manifest_name) + [x[2] for x in pending])
        # sources.jsonl is the authoritative commit record and is replaced last.
        write_jsonl_atomic(root / 'manifests' / 'sources.jsonl', existing + [x[2] for x in pending])
    except Exception:
        for path, content in previous_manifests.items():
            restore = path.with_name(path.name + '.restore.tmp')
            restore.write_bytes(content)
            os.replace(restore, path)
        for path in created:
            path.unlink(missing_ok=True)
        for path in created_records:
            path.unlink(missing_ok=True)
        raise
    return len(pending)


def split_training(root: Path, seed: int, fraction: float, require_quality_review: bool = False) -> dict:
    if not 0 < fraction < .5:
        raise ValueError('validation fraction must be between 0 and 0.5')
    sources = [x for x in source_inventory(root) if x['partition'] == 'training']
    if len(sources) < 2:
        raise ValueError('At least two training documents are required')
    if require_quality_review:
        report = audit(root)
        if not report['valid']:
            raise ValueError(f"Dataset audit failed before quality-gated split: {report['problems'][:5]}")
        statuses = quality_review_status(sources, read_jsonl(root / 'manifests' / 'quality_reviews.jsonl'))
        pending = [sid for sid, status in statuses.items() if status != 'two-reviewer-acceptance-recorded']
        if pending:
            raise ValueError(f'Two-reviewer source quality acceptance is not recorded for {len(pending)} training sources')
    # Documents from one source work stay together when source_group is present.
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in sources:
        groups[str(item.get('source_group') or item.get('source_id') or item['path'])].append(item)
    if len(groups) < 2:
        raise ValueError('Need at least two independent source groups for a non-leaking holdout')
    target = max(1, min(len(sources) - 1, round(len(sources) * fraction)))
    rng = random.Random(seed)
    domain_totals = Counter(str(item.get('domain', 'unclassified')) for item in sources)
    groups_by_domain: dict[str, list[str]] = defaultdict(list)
    group_domains: dict[str, Counter] = {}
    for key, rows in groups.items():
        counts = Counter(str(item.get('domain', 'unclassified')) for item in rows)
        group_domains[key] = counts
        dominant_domain = min(counts, key=lambda domain: (-counts[domain], domain))
        groups_by_domain[dominant_domain].append(key)
    for queue in groups_by_domain.values():
        rng.shuffle(queue)
    domains = sorted(domain_totals)
    positions = {domain: 0 for domain in domains}
    selected, selected_set, count = [], set(), 0
    validation_domain_counts = Counter()
    # Select whole source groups from the domain furthest below the requested
    # validation fraction. Multi-domain groups update every affected domain.
    while count < target:
        available = []
        for domain in domains:
            queue = groups_by_domain.get(domain, [])
            pos = positions[domain]
            while pos < len(queue):
                key = queue[pos]
                if key in selected_set:
                    pos += 1
                    continue
                if len(sources) - count - len(groups[key]) < 1:
                    pos += 1
                    continue
                break
            positions[domain] = pos
            if pos < len(queue):
                available.append(domain)
        if not available:
            break
        domain = min(available, key=lambda d: (validation_domain_counts[d] / domain_totals[d], d))
        key = groups_by_domain[domain][positions[domain]]
        positions[domain] += 1
        selected.append(key)
        selected_set.add(key)
        count += len(groups[key])
        validation_domain_counts.update(group_domains[key])
    if not selected:
        raise ValueError('Unable to produce a validation split while keeping source groups intact')
    val_groups = selected_set
    train, val = [], []
    for item in sources:
        (val if str(item.get('source_group') or item.get('source_id') or item['path']) in val_groups else train).append(item)
    # Materialize standard corpora with per-directory manifests for v0.3.
    base = root / 'curated' / 'pretraining'
    stage = Path(tempfile.mkdtemp(prefix='.pretraining-stage-', dir=root / 'curated'))
    backup = None
    try:
        for split_name, rows in (('train', train), ('validation', val)):
            d = stage / split_name
            d.mkdir()
            output_rows = []
            for i, item in enumerate(rows):
                src = root / item['path']
                if not src.is_file() or sha256(src.read_bytes()) != item['content_hash']:
                    raise ValueError(f"Source changed since import: {item['path']}")
                name = f'{i:06d}_{item["content_hash"][:12]}.txt'
                shutil.copyfile(src, d / name)
                output_rows.append({'path': name, 'title': item['title'], 'domain': item['domain'],
                                    'license': item['license_id'], 'url': item['source_url'],
                                    'sha256': item['content_hash'], 'source_id': item['source_id'],
                                    'source_group': item.get('source_group', item['path']), 'split_seed': seed})
            (d / 'manifest.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in output_rows), encoding='utf-8')
        if base.exists() and any(base.iterdir()):
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            backup = base.with_name(f'pretraining.previous-{stamp}')
            if backup.exists():
                raise FileExistsError(f'Cannot preserve previous curated split; backup exists: {backup}')
            os.replace(base, backup)
        os.replace(stage, base)
    except Exception:
        if backup is not None and backup.exists() and not base.exists():
            os.replace(backup, base)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {'train_documents': len(train), 'validation_documents': len(val), 'seed': seed,
            'validation_fraction_actual': len(val) / len(sources),
            'training_domain_counts': dict(sorted(Counter(str(item.get('domain', 'unclassified')) for item in train).items())),
            'validation_domain_counts': dict(sorted(Counter(str(item.get('domain', 'unclassified')) for item in val).items())),
            'source_group_disjoint': True, 'evaluation_documents_separate': True,
            'quality_review_gate_enabled': require_quality_review}


def quality_review_status(source_rows: list[dict], review_rows: list[dict]) -> dict[str, str]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in review_rows:
        by_source[str(row.get('source_id'))].append(row)
    result = {}
    for source in source_rows:
        source_id = str(source.get('source_id'))
        reviews = by_source.get(source_id, [])
        if not reviews:
            result[source_id] = 'unreviewed'
        elif any(r.get('decision') == 'reject' for r in reviews):
            result[source_id] = 'rejected'
        elif (len({r.get('reviewer_id') for r in reviews if r.get('relationship_to_source') == 'independent'}) >= 2 and
              len({str(r.get('reviewer_affiliation', '')).casefold() for r in reviews if r.get('relationship_to_source') == 'independent'}) >= 2 and
              all(r.get('decision') == 'accept' and isinstance(r.get('ratings'), dict) and
                  set(r['ratings']) == QUALITY_RUBRIC and all(type(n) is int and n >= 1 for n in r['ratings'].values())
                  for r in reviews)):
            result[source_id] = 'two-reviewer-acceptance-recorded'
        else:
            result[source_id] = 'review-incomplete'
    return result


def audit(root: Path) -> dict:
    sources = source_inventory(root)
    review_rows = read_jsonl(root / 'manifests' / 'quality_reviews.jsonl')
    source_partitions = {str(x.get('source_id')): x.get('partition') for x in sources}
    seen_paths, seen_source_ids, train_hashes, eval_hashes = set(), set(), set(), set()
    domains, licenses = Counter(), Counter()
    words = total_bytes = short_docs = repetitive_docs = 0
    train_docs = eval_docs = 0
    problems = []
    review_ids, reviewer_source_pairs = set(), set()
    source_by_id = {str(row.get('source_id')): row for row in sources}
    for review in review_rows:
        required_review = ('review_id', 'source_id', 'reviewer_id', 'reviewer_affiliation',
                           'relationship_to_source', 'reviewed_at', 'method', 'evidence_refs', 'ratings', 'decision')
        if not all(review.get(field) for field in required_review):
            problems.append(f"incomplete source quality review: {review.get('review_id', '<unknown>')}")
            continue
        rid, sid = str(review['review_id']), str(review['source_id'])
        if rid in review_ids:
            problems.append(f'duplicate quality review id: {rid}')
        review_ids.add(rid)
        pair = (sid, str(review['reviewer_id']))
        if pair in reviewer_source_pairs:
            problems.append(f'duplicate reviewer for source: {sid} / {review["reviewer_id"]}')
        reviewer_source_pairs.add(pair)
        if sid not in source_by_id:
            problems.append(f'quality review references unknown source: {sid}')
        if review.get('relationship_to_source') != 'independent':
            problems.append(f'quality reviewer did not declare independence: {rid}')
        elif sid in source_by_id and str(review['reviewer_affiliation']).casefold() == str(source_by_id[sid].get('publisher', '')).casefold():
            problems.append(f'quality reviewer affiliation matches source publisher: {rid}')
        ratings = review.get('ratings')
        if not isinstance(ratings, dict) or set(ratings) != QUALITY_RUBRIC or any(
                type(score) is not int or score not in (0, 1, 2) for score in (ratings.values() if isinstance(ratings, dict) else [])):
            problems.append(f'invalid quality rubric ratings: {rid}')
        if review.get('decision') not in {'accept', 'reject', 'inconclusive'}:
            problems.append(f'invalid quality review decision: {rid}')
        if not isinstance(review.get('evidence_refs'), list) or not review['evidence_refs']:
            problems.append(f'quality review missing evidence references: {rid}')
        if not review.get('review_hash'):
            problems.append(f'quality review missing record hash: {rid}')
        else:
            payload = {k: v for k, v in review.items() if k not in {'review_hash', 'recorded_at'}}
            if sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()) != review['review_hash']:
                problems.append(f'quality review hash mismatch: {rid}')
        try:
            datetime.fromisoformat(str(review['reviewed_at']).replace('Z', '+00:00'))
        except ValueError:
            problems.append(f'quality review has invalid reviewed_at: {rid}')
    license_records = read_jsonl(root / 'manifests' / 'licenses.jsonl')
    quality_records = read_jsonl(root / 'manifests' / 'quality_reports.jsonl')
    license_by_path, quality_by_path = {}, {}
    for record, target, label in ((r, license_by_path, 'license') for r in license_records):
        path_key = record.get('path')
        if path_key in target:
            problems.append(f'duplicate {label} record: {path_key}')
        target[path_key] = record
    for record in quality_records:
        path_key = record.get('path')
        if path_key in quality_by_path:
            problems.append(f'duplicate quality record: {path_key}')
        quality_by_path[path_key] = record
    training_document_paths, evaluation_document_paths = set(), set()
    for item in sources:
        rel = safe_relative(item['path'])
        if rel.as_posix() in seen_paths:
            problems.append(f'duplicate manifest path: {rel}')
        seen_paths.add(rel.as_posix())
        source_id = str(item.get('source_id', ''))
        if source_id in seen_source_ids:
            problems.append(f'duplicate source_id: {source_id}')
        seen_source_ids.add(source_id)
        path = root.joinpath(*rel.parts)
        if not path.is_file():
            problems.append(f'missing file: {rel}')
            continue
        raw = path.read_bytes()
        if sha256(raw) != item.get('content_hash'):
            problems.append(f'content hash mismatch: {rel}')
            continue
        text = raw.decode('utf-8')
        norm = sha256(normalize(text).encode('utf-8'))
        if norm != item.get('normalized_hash'):
            problems.append(f'normalized content hash mismatch: {rel}')
        observed_quality = quality(text)
        total_bytes += len(raw)
        if observed_quality['words'] < 100:
            short_docs += 1
        if observed_quality['repeated_line_fraction'] > .5 and observed_quality['words'] > 100:
            repetitive_docs += 1
        quality_record = quality_by_path.get(rel.as_posix())
        if not quality_record:
            problems.append(f'missing quality record: {rel}')
        else:
            expected_quality = {'path': rel.as_posix(), 'content_hash': item['content_hash'], **observed_quality}
            if quality_record != expected_quality:
                problems.append(f'quality record mismatch: {rel}')
        license_record = license_by_path.get(rel.as_posix())
        if not license_record:
            problems.append(f'missing license record: {rel}')
        else:
            expected_license_fields = {
                'license_id': item.get('license_id'),
                'license_url': item.get('license_url'),
                'license_status': item.get('license_status'),
                'review_evidence': item.get('license_review_evidence'),
                'license_terms_url': item.get('license_terms_url'),
                'license_scope': item.get('license_scope'),
                'additional_license_urls': item.get('additional_license_urls', []),
            }
            if any(license_record.get(field) != value for field, value in expected_license_fields.items()):
                problems.append(f'license record mismatch: {rel}')
        if item['partition'] == 'training':
            train_docs += 1; training_document_paths.add(rel.as_posix())
            train_hashes.add(norm); words += observed_quality['words']
        elif item['partition'] == 'evaluation':
            eval_docs += 1; evaluation_document_paths.add(rel.as_posix()); eval_hashes.add(norm)
        else:
            problems.append(f'unknown partition: {rel}')
        domains[item.get('domain', 'unclassified')] += 1
        licenses[item.get('license_status', 'UNKNOWN')] += 1
        if not all(item.get(field) for field in SOURCE_FIELDS):
            problems.append(f'incomplete provenance metadata: {rel}')
        if item.get('license_status') != 'INDEPENDENTLY_REVIEWED' or not item.get('license_review_evidence'):
            problems.append(f'source-specific license review is absent or lacks evidence reference: {rel}')
        if item.get('source_quality_status') != 'UNREVIEWED':
            problems.append(f'source review status must be derived from the quality review manifest: {rel}')
        record_path = root / 'evidence' / 'source_records' / (sha256(str(item.get('source_id', '')).encode()) + '.json')
        try:
            source_record = json.loads(record_path.read_text(encoding='utf-8'))
            if source_record != item:
                problems.append(f'source record mismatch: {rel}')
        except (OSError, json.JSONDecodeError):
            problems.append(f'missing or invalid source record: {rel}')
    overlap = train_hashes & eval_hashes
    if overlap:
        problems.append(f'{len(overlap)} exact normalized training/evaluation content overlaps')
    train_source_groups = {str(item.get('source_group') or item.get('source_id') or item.get('path'))
                           for item in sources if item.get('partition') == 'training'}
    eval_source_groups = {str(item.get('source_group') or item.get('source_id') or item.get('path'))
                          for item in sources if item.get('partition') == 'evaluation'}
    group_overlap = train_source_groups & eval_source_groups
    if group_overlap:
        problems.append(f'{len(group_overlap)} source_group overlaps between training and evaluation')
    training_manifest_paths = {r.get('path') for r in read_jsonl(root / 'manifests' / 'documents.jsonl')}
    evaluation_manifest_paths = {r.get('path') for r in read_jsonl(root / 'manifests' / 'evaluation.jsonl') if 'id' not in r}
    if training_manifest_paths != training_document_paths:
        problems.append('training document manifest does not match sources manifest')
    if evaluation_manifest_paths != evaluation_document_paths:
        problems.append('evaluation document manifest does not match sources manifest')
    if set(license_by_path) != seen_paths:
        problems.append('license manifest does not match sources manifest')
    if set(quality_by_path) != seen_paths:
        problems.append('quality manifest does not match sources manifest')
    record_sets = (('reasoning_examples.jsonl', 'training'), ('epistemic_examples.jsonl', 'training'),
                   ('evaluation.jsonl', 'evaluation'))
    seen_record_ids = set()
    structured_by_family = {'reasoning': {}, 'epistemic': {}, 'evaluation': {}}
    for manifest_name, expected_partition in record_sets:
        for record in read_jsonl(root / 'manifests' / manifest_name):
            if 'id' not in record:
                continue  # document source entry in evaluation.jsonl
            rid = str(record['id'])
            if rid in seen_record_ids:
                problems.append(f'duplicate structured record id: {rid}')
            seen_record_ids.add(rid)
            family = record.get('family')
            if family in structured_by_family:
                structured_by_family[family][rid] = record
            if family == 'evaluation':
                task_type = record.get('task_type')
                expected_track = EVALUATION_TASK_TO_TRACK.get(task_type) if isinstance(task_type, str) else None
                if expected_track is None or record.get('track') != expected_track:
                    problems.append(f'evaluation task/track mismatch or unsupported task type: {rid}')
                scoring = record.get('scoring')
                if (not isinstance(scoring, dict) or not isinstance(scoring.get('metric'), str) or
                        not scoring['metric'].strip() or not isinstance(scoring.get('protocol'), str) or
                        not scoring['protocol'].strip()):
                    problems.append(f'evaluation record has incomplete scoring protocol: {rid}')
            for sid in map(str, record.get('source_ids', [])):
                if sid not in source_partitions:
                    problems.append(f'structured record {rid} references unknown source: {sid}')
                elif source_partitions[sid] != expected_partition:
                    problems.append(f'structured record {rid} crosses train/evaluation source boundary: {sid}')
            if record.get('record_hash'):
                payload = {k: v for k, v in record.items() if k not in {'record_hash', 'imported_at'}}
                if sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()) != record['record_hash']:
                    problems.append(f'structured record hash mismatch: {rid}')
    for family, records in structured_by_family.items():
        if family == 'evaluation':
            for track in TRACKS:
                expected_by_id = {rid: row for rid, row in records.items() if row.get('track') == track}
                local_records = read_jsonl(root / 'evaluation' / track / 'records.jsonl')
                local_by_id = {str(row.get('id')): row for row in local_records}
                if set(local_by_id) != set(expected_by_id):
                    problems.append(f'evaluation/{track} records do not match canonical track manifest')
                else:
                    for rid, row in expected_by_id.items():
                        if local_by_id[rid] != row:
                            problems.append(f'evaluation/{track} record copy mismatch: {rid}')
            continue
        directory = 'epistemic_training' if family == 'epistemic' else family
        local_records = read_jsonl(root / 'curated' / directory / 'records.jsonl')
        local_by_id = {str(row.get('id')): row for row in local_records}
        if set(local_by_id) != set(records):
            problems.append(f'{family} record files do not match canonical manifest')
        else:
            for rid, row in records.items():
                if local_by_id[rid] != row:
                    problems.append(f'{family} record copy mismatch: {rid}')
    training_sources = [row for row in sources if row.get('partition') == 'training']
    review_statuses = quality_review_status(sources, review_rows)
    training_domains = sorted({str(row.get('domain', 'unclassified'))
                               for row in training_sources})
    evaluation_records = list(structured_by_family['evaluation'].values())
    evaluation_track_counts = Counter(str(row.get('track', 'unknown')) for row in evaluation_records)
    evaluation_task_type_counts = Counter(str(row.get('task_type', 'unknown')) for row in evaluation_records)
    minimum_training_documents = 1000
    minimum_training_domains = 10
    expected_evaluation_tasks = {
        'language_modeling', 'factuality', 'reasoning', 'generalization',
        'uncertainty_calibration', 'abstention',
    }
    readiness_gaps = []
    if train_docs < minimum_training_documents:
        readiness_gaps.append(f'training corpus has {train_docs}/{minimum_training_documents} documents')
    if len(training_domains) < minimum_training_domains:
        readiness_gaps.append(f'training corpus covers {len(training_domains)}/{minimum_training_domains} domains')
    if not structured_by_family['reasoning']:
        readiness_gaps.append('structured reasoning examples are absent')
    if not structured_by_family['epistemic']:
        readiness_gaps.append('epistemic examples are absent')
    missing_evaluation_tasks = sorted(expected_evaluation_tasks - evaluation_task_type_counts.keys())
    if missing_evaluation_tasks:
        readiness_gaps.append('evaluation tasks are absent: ' + ', '.join(missing_evaluation_tasks))
    if not training_sources or not all(
            review_statuses.get(str(row.get('source_id'))) == 'two-reviewer-acceptance-recorded'
            for row in training_sources):
        readiness_gaps.append('two-reviewer source-quality acceptance is incomplete')
    # The workflow records self-reported reviews; it cannot authenticate reviewer
    # identity or inspect external evidence, so this remains an explicit gap.
    readiness_gaps.append('independent source-quality verification is not authenticated by this tool')
    readiness_gaps.append('factuality and semantic-contamination verification are not performed by this tool')
    two_reviewer_acceptance_recorded = bool(training_sources) and all(
        review_statuses.get(str(row.get('source_id'))) == 'two-reviewer-acceptance-recorded'
        for row in training_sources)
    coverage_gates = {
        'minimum_training_documents_met': train_docs >= minimum_training_documents,
        'minimum_training_domains_met': len(training_domains) >= minimum_training_domains,
        'reasoning_examples_present': bool(structured_by_family['reasoning']),
        'epistemic_examples_present': bool(structured_by_family['epistemic']),
        'all_required_evaluation_task_types_present': not missing_evaluation_tasks,
        'two_reviewer_acceptance_recorded': two_reviewer_acceptance_recorded,
    }
    objective_readiness = {
        'status': 'coverage_gates_incomplete' if not all(coverage_gates.values()) else 'coverage_gates_met_quality_unverified',
        'coverage_gates': coverage_gates,
        'minimum_training_documents': minimum_training_documents,
        'training_documents': train_docs,
        'minimum_training_domains': minimum_training_domains,
        'training_domains': training_domains,
        'reasoning_examples': len(structured_by_family['reasoning']),
        'epistemic_examples': len(structured_by_family['epistemic']),
        'evaluation_records': len(evaluation_records),
        'evaluation_records_by_track': dict(sorted(evaluation_track_counts.items())),
        'evaluation_records_by_task_type': dict(sorted(evaluation_task_type_counts.items())),
        'missing_evaluation_task_types': missing_evaluation_tasks,
        'independent_source_quality_verified': False,
        'factual_claims_verified': False,
        'semantic_contamination_verified': False,
        'gaps': readiness_gaps,
    }
    return {'documents': len(sources), 'training_documents': train_docs, 'evaluation_documents': eval_docs,
            'training_words': words, 'total_utf8_bytes': total_bytes,
            'short_documents_under_100_words': short_docs,
            'potentially_repetitive_documents_over_100_words': repetitive_docs,
            'domains': dict(sorted(domains.items())),
            'license_statuses': dict(sorted(licenses.items())), 'exact_normalized_split_overlap': len(overlap),
            'source_group_split_overlap': len(group_overlap),
            'recorded_source_quality_review_statuses': dict(sorted(Counter(review_statuses.values()).items())),
            'two_reviewer_source_quality_acceptance_recorded': two_reviewer_acceptance_recorded,
            'objective_readiness': objective_readiness,
            'independent_source_quality_verified': False, 'factual_claims_verified': False,
            'semantic_contamination_verified': False, 'problems': problems, 'valid': not problems}


def import_records(root: Path, input_file: Path, family: str, evaluation_track: str | None) -> int:
    if family not in {'reasoning', 'epistemic', 'evaluation'}:
        raise ValueError('family must be reasoning, epistemic, or evaluation')
    if family == 'evaluation' and evaluation_track not in TRACKS:
        raise ValueError(f'evaluation track must be one of {sorted(TRACKS)}')
    rows = read_jsonl(input_file)
    if not rows:
        raise ValueError('Input JSONL contains no records')
    if family == 'reasoning':
        required = {'id', 'task_type', 'prompt', 'constraints', 'answer', 'rationale', 'source_ids'}
        target = root / 'curated' / 'reasoning'
        manifest = root / 'manifests' / 'reasoning_examples.jsonl'
    elif family == 'epistemic':
        required = {'id', 'prompt', 'supported_claims', 'disputed_claims', 'assumptions', 'unknowns', 'conclusion', 'source_ids'}
        target = root / 'curated' / 'epistemic_training'
        manifest = root / 'manifests' / 'epistemic_examples.jsonl'
    else:
        required = {'id', 'task_type', 'input', 'expected', 'scoring', 'source_ids'}
        target = root / 'evaluation' / str(evaluation_track)
        manifest = root / 'manifests' / 'evaluation.jsonl'
    existing = read_jsonl(manifest)
    source_partitions = {str(r['source_id']): r['partition'] for r in source_inventory(root)}
    ids = {str(r.get('id')) for r in existing}
    for row in rows:
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Record {row.get('id', '<unknown>')} missing fields: {missing}")
        if family == 'reasoning':
            kinds = {'causal', 'counterfactual', 'tradeoff', 'cross_domain_synthesis', 'experimental_problem_solving'}
            if row['task_type'] not in kinds:
                raise ValueError(f"Unsupported reasoning task_type: {row['task_type']}")
            if row['task_type'] == 'cross_domain_synthesis':
                missing_cross = {'domain_tags', 'shared_mechanism', 'mechanism_evidence'} - row.keys()
                if missing_cross or not isinstance(row.get('domain_tags'), list) or len(set(row['domain_tags'])) < 2:
                    raise ValueError(f"Cross-domain example {row['id']} needs 2+ domain_tags, shared_mechanism, and mechanism_evidence")
            if row['task_type'] == 'experimental_problem_solving':
                experiment_fields = {'hypothesis', 'experiment_design', 'observed_outcome', 'interpretation', 'failed_prediction', 'model_revision'}
                missing_experiment = sorted(experiment_fields - row.keys())
                if missing_experiment:
                    raise ValueError(f"Experimental example {row['id']} missing fields: {missing_experiment}")
        if family == 'evaluation':
            task_type = row.get('task_type')
            expected_track = EVALUATION_TASK_TO_TRACK.get(task_type) if isinstance(task_type, str) else None
            if expected_track is None:
                raise ValueError(f"Unsupported evaluation task_type: {task_type!r}")
            if evaluation_track != expected_track:
                raise ValueError(f"Evaluation task_type {task_type!r} must use track {expected_track!r}, not {evaluation_track!r}")
            scoring = row.get('scoring')
            if (not isinstance(scoring, dict) or not isinstance(scoring.get('metric'), str) or
                    not scoring['metric'].strip() or not isinstance(scoring.get('protocol'), str) or
                    not scoring['protocol'].strip()):
                raise ValueError(f"Evaluation record {row['id']} scoring requires nonempty metric and protocol strings")
        if not isinstance(row['source_ids'], list) or not row['source_ids']:
            raise ValueError(f"Record {row['id']} must identify one or more source_ids")
        source_refs = set(map(str, row['source_ids']))
        unknown = sorted(source_refs - source_partitions.keys())
        if unknown:
            raise ValueError(f"Record {row['id']} references unknown source_ids: {unknown}")
        expected_partition = 'evaluation' if family == 'evaluation' else 'training'
        wrong_partition = sorted(sid for sid in source_refs if source_partitions[sid] != expected_partition)
        if wrong_partition:
            raise ValueError(f"Record {row['id']} references sources outside its {expected_partition} partition: {wrong_partition}")
        if str(row['id']) in ids:
            raise ValueError(f"Duplicate record id: {row['id']}")
        ids.add(str(row['id']))
        if family == 'epistemic' and not isinstance(row['unknowns'], list):
            raise ValueError(f"Epistemic record {row['id']} must explicitly list unknowns")
        row['family'] = family
        row['track'] = evaluation_track
        row['record_hash'] = sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode())
        row['imported_at'] = datetime.now(timezone.utc).isoformat()
    target.mkdir(parents=True, exist_ok=True)
    track_file = target / 'records.jsonl'
    write_jsonl_atomic(track_file, read_jsonl(track_file) + rows)
    write_jsonl_atomic(manifest, existing + rows)
    return len(rows)


def import_quality_reviews(root: Path, input_file: Path) -> int:
    rows = read_jsonl(input_file)
    if not rows:
        raise ValueError('Review JSONL contains no records')
    sources = {str(row['source_id']): row for row in source_inventory(root)}
    manifest = root / 'manifests' / 'quality_reviews.jsonl'
    existing = read_jsonl(manifest)
    review_ids = {str(row.get('review_id')) for row in existing}
    reviewer_pairs = {(str(row.get('source_id')), str(row.get('reviewer_id'))) for row in existing}
    required = {'review_id', 'source_id', 'reviewer_id', 'reviewer_affiliation',
                'relationship_to_source', 'reviewed_at', 'method', 'evidence_refs', 'ratings', 'decision'}
    for row in rows:
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Review {row.get('review_id', '<unknown>')} missing fields: {missing}")
        rid, sid = str(row['review_id']), str(row['source_id'])
        if rid in review_ids:
            raise ValueError(f'Duplicate review_id: {rid}')
        review_ids.add(rid)
        if sid not in sources:
            raise ValueError(f'Unknown source_id in review {rid}: {sid}')
        pair = (sid, str(row['reviewer_id']))
        if pair in reviewer_pairs:
            raise ValueError(f'Reviewer already reviewed this source: {pair}')
        reviewer_pairs.add(pair)
        if row['relationship_to_source'] != 'independent':
            raise ValueError(f'Reviewer must explicitly declare relationship_to_source=independent: {rid}')
        if str(row['reviewer_affiliation']).casefold() == str(sources[sid].get('publisher', '')).casefold():
            raise ValueError(f'Reviewer affiliation must differ from source publisher: {rid}')
        ratings = row['ratings']
        if not isinstance(ratings, dict) or set(ratings) != QUALITY_RUBRIC or any(
                type(value) is not int or value not in (0, 1, 2) for value in (ratings.values() if isinstance(ratings, dict) else [])):
            raise ValueError(f'Review {rid} requires integer 0/1/2 ratings for {sorted(QUALITY_RUBRIC)}')
        if row['decision'] not in {'accept', 'reject', 'inconclusive'}:
            raise ValueError(f'Invalid source quality decision in review {rid}')
        if not isinstance(row['evidence_refs'], list) or not row['evidence_refs'] or not all(isinstance(x, str) and x.strip() for x in row['evidence_refs']):
            raise ValueError(f'Review {rid} requires evidence references')
        if not isinstance(row['method'], str) or not row['method'].strip():
            raise ValueError(f'Review {rid} requires a review method')
        try:
            datetime.fromisoformat(str(row['reviewed_at']).replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError(f'Review {rid} has invalid reviewed_at; use ISO 8601') from exc
        row['review_hash'] = sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode())
        row['recorded_at'] = datetime.now(timezone.utc).isoformat()
    write_jsonl_atomic(manifest, existing + rows)
    return len(rows)


def metadata_template(input_dir: Path) -> list[dict]:
    if input_dir.is_symlink() or not input_dir.is_dir():
        raise ValueError('Input must be a real directory, not a symlink')
    rows = []
    for path in sorted(p for p in input_dir.rglob('*') if p.is_file() and p.suffix.lower() in {'.txt', '.md'}):
        if path.is_symlink() or not path.resolve().is_relative_to(input_dir.resolve()):
            raise ValueError(f'Input document is a symlink or escapes root: {path}')
        rel = path.relative_to(input_dir).as_posix()
        rows.append({'source_id': 'source-' + sha256(path.read_bytes())[:16], 'path': rel, 'title': path.stem,
                     'publisher': 'REQUIRED', 'source_url': 'REQUIRED', 'publication_date': 'REQUIRED',
                     'license_id': 'REQUIRED', 'license_url': 'REQUIRED',
                     'license_status': 'SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED',
                     'source_quality_status': 'UNREVIEWED',
                     'domain': 'unclassified', 'source_type': 'REQUIRED',
                     'content_hash': 'sha256:' + sha256(path.read_bytes()),
                     'source_group': rel.split('/')[0] if '/' in rel else rel,
                     'processing_history': [{'operation': 'metadata_template',
                                             'note': 'Replace REQUIRED and placeholder values before import'}]})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description='Local-first Expansformer knowledge and reasoning dataset pipeline')
    sub = p.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('init'); i.add_argument('--root', required=True)
    imp = sub.add_parser('import'); imp.add_argument('--root', required=True); imp.add_argument('--input', required=True)
    imp.add_argument('--metadata', required=True); imp.add_argument('--partition', choices=['training', 'evaluation'], required=True)
    imp.add_argument('--domain'); imp.add_argument('--track', choices=sorted(TRACKS))
    s = sub.add_parser('split'); s.add_argument('--root', required=True); s.add_argument('--seed', type=int, default=42)
    s.add_argument('--validation-fraction', type=float, default=.15)
    s.add_argument('--require-quality-review', action='store_true', help='Require recorded two-reviewer acceptance for every training document')
    a = sub.add_parser('audit'); a.add_argument('--root', required=True)
    a.add_argument('--require-coverage-gates', action='store_true',
                   help='Exit nonzero unless objective document/domain/example/evaluation coverage gates are met; does not authenticate evidence')
    t = sub.add_parser('template', help='Create metadata skeletons and exact byte hashes; does not import')
    t.add_argument('--input', required=True)
    pf = sub.add_parser('preflight', help='Validate an incoming batch and emit non-acceptance screening diagnostics')
    pf.add_argument('--input', required=True); pf.add_argument('--metadata', required=True)
    pw = sub.add_parser('prepare-wikipedia', help='Write a separate minimally normalized candidate batch with provenance')
    pw.add_argument('--input', required=True); pw.add_argument('--metadata', required=True); pw.add_argument('--out', required=True)
    r = sub.add_parser('import-records'); r.add_argument('--root', required=True); r.add_argument('--input', required=True)
    r.add_argument('--family', choices=['reasoning', 'epistemic', 'evaluation'], required=True)
    r.add_argument('--track', choices=sorted(TRACKS))
    q = sub.add_parser('import-quality-reviews'); q.add_argument('--root', required=True); q.add_argument('--input', required=True)
    args = p.parse_args()
    root = Path(args.root).resolve() if hasattr(args, 'root') else None
    if args.cmd == 'init':
        init(root); result = {'initialized': str(root), 'network_access': 'disabled'}
    elif args.cmd == 'import':
        result = {'imported_documents': import_documents(root, Path(args.input).resolve(), Path(args.metadata).resolve(), args.partition, args.domain, args.track)}
    elif args.cmd == 'split':
        result = split_training(root, args.seed, args.validation_fraction, args.require_quality_review)
    elif args.cmd == 'audit':
        result = audit(root)
    elif args.cmd == 'preflight':
        result = preflight(Path(args.input).resolve(), Path(args.metadata).resolve())
    elif args.cmd == 'prepare-wikipedia':
        result = prepare_wikipedia_batch(Path(args.input).resolve(), Path(args.metadata).resolve(), Path(args.out).resolve())
    elif args.cmd == 'import-records':
        result = {'imported_records': import_records(root, Path(args.input).resolve(), args.family, args.track)}
    elif args.cmd == 'import-quality-reviews':
        result = {'imported_reviews': import_quality_reviews(root, Path(args.input).resolve())}
    else:
        print(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                      for row in metadata_template(Path(args.input).resolve())), end='')
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.cmd == 'audit' and not result['valid']:
        raise SystemExit(1)
    if args.cmd == 'preflight' and not result['valid']:
        raise SystemExit(1)
    if (args.cmd == 'audit' and args.require_coverage_gates and
            result['objective_readiness']['status'] == 'coverage_gates_incomplete'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
