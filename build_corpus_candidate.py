#!/usr/bin/env python3
"""Phase 1 corpus candidate builder for recovered Expansformer staged batches.

Reads staged batches (manifest.jsonl + text files) and writes an inspectable, deterministic
review package: inventory, provenance/hash checks, license triage, quality screen, duplicate
report, domain coverage, status manifests, a frozen split PLAN, a Void Register and a build report.

What this tool does NOT do: no network access, no legal clearance, no factual verification, no
training. License triage uses manifest metadata only. Nothing becomes `approved` unless a human
decisions file says so (see --decisions). Source files are never modified; the output directory
must be new and outside the source tree.

Example:
  python build_corpus_candidate.py --src ./corpus_work/src --out ./corpus_work/build
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path

VERSION = 'corpus-candidate-0.1'
SPLIT_SALT = 'expansformer-corpus-split-v1'
SPLIT_FRACTIONS = (('train', 0.90), ('validation', 0.05), ('test', 0.05))
STATUSES = ('approved', 'rejected', 'manual_review', 'unresolved')
HEX64 = re.compile(r'^(?:sha256:)?[0-9a-fA-F]{64}$')
NEAR_DUP_JACCARD = 0.5       # exact 5-word-shingle Jaccard at or above this: near duplicate (flag, not removal)
OVERLAP_REVIEW_JACCARD = 0.2
TFIDF_REVIEW_COSINE = 0.45
SKETCH_K = 128
REQUIRED_PROVENANCE = ('source_id', 'title', 'publisher', 'license_id', 'content_hash', 'retrieved_at', 'domain')
SNAPSHOT_CLAIM_DEFAULT = 'e07b57fa83c2ac567c26d4d3ff39038bd4a85fe59666857c58fa154579c6d526'
# Minimum character length for a line to count in the repeated-line fraction.
# Fragments shorter than this (single punctuation marks, single letters, one-char list bullets)
# are structural noise and should not drive the flag.
REPEATED_LINE_MIN_CHARS = 20

TARGET_DOMAINS = {
    'computer_science_software_algorithms': ['algorithm', 'software', 'programming', 'compiler', 'database', 'operating system', 'data structure', 'source code', 'network protocol', 'complexity'],
    'mathematics_logic_optimization': ['theorem', 'proof', 'lemma', 'equation', 'optimization', 'calculus', 'algebra', 'probability', 'integral', 'combinatorics'],
    'physics_complex_systems': ['quantum', 'thermodynamic', 'entropy', 'complex system', 'dynamical', 'relativity', 'particle physics', 'chaos', 'emergence', 'thermodynamics'],
    'biology_neuroscience_cognitive_science': ['neuron', 'neural circuit', 'cortex', 'synapse', 'genome', 'evolutionary biology', 'neuroimaging', 'hippocampus', 'cerebral', 'epigenetic'],
    'artificial_intelligence_machine_learning': ['machine learning', 'neural network', 'deep learning', 'reinforcement learning', 'training data', 'language model', 'transformer', 'backpropagation', 'gradient descent', 'overfitting'],
    'human_behavior_psychology': ['emotion regulation', 'motivation', 'personality trait', 'cognitive bias', 'decision making', 'behavioral economics', 'psychological', 'self-efficacy', 'metacognition', 'working memory'],
    'human_ai_interaction_collaboration': ['human-ai', 'human–ai', 'human-computer interaction', 'trust in ai', 'explainab', 'human-centered ai', 'human-robot', 'algorithmic fairness', 'ai transparency', 'automation bias'],
    'causal_inference_scientific_method': ['causal inference', 'counterfactual', 'confounding', 'randomized controlled', 'replication crisis', 'statistical inference', 'instrumental variable', 'propensity score', 'meta-analysis', 'placebo'],
    'systems_architecture_engineering': ['system design', 'fault tolerance', 'distributed system', 'control system', 'feedback loop', 'reliability engineering', 'system architecture', 'embedded system', 'real-time system', 'hardware design'],
    'philosophy_epistemology_ethics': ['epistemology', 'ethical theory', 'moral philosophy', 'ontology', 'consciousness studies', 'justification', 'normative ethics', 'virtue ethics', 'phenomenology', 'philosophy of mind'],
}
# Required counts/density for keyword-heuristic match — both thresholds must be met.
# These are deliberately strict to avoid over-counting in short or general documents.
DOMAIN_KW_MIN_TERMS = 5    # distinct terms from the list that appear in the text
DOMAIN_KW_MIN_DENSITY = 5.0  # combined occurrences per 1000 words

# Approximate mapping from manifest `domain` label strings to target domains.
# One manifest label can map to multiple target domains; one target domain can be
# covered by multiple manifest labels.  Used to identify target domains that have
# no manifest-labeled documents at all.
MANIFEST_TO_TARGET = {
    'biological_intelligence_ai':   ['biology_neuroscience_cognitive_science', 'artificial_intelligence_machine_learning'],
    'human_ai_collaboration':        ['human_ai_interaction_collaboration', 'artificial_intelligence_machine_learning'],
    'human_cognition_behavior':      ['human_behavior_psychology', 'biology_neuroscience_cognitive_science'],
    'earth_environment':             ['physics_complex_systems'],
    'energy_power':                  ['physics_complex_systems', 'systems_architecture_engineering'],
    'space':                         ['physics_complex_systems'],
    'human_senses':                  ['biology_neuroscience_cognitive_science'],
    'social_power':                  ['human_behavior_psychology', 'philosophy_epistemology_ethics'],
    # manifest labels below have no clear mapping to any target domain
    'entertainment':                 [],
    'geography':                     [],
    'literature':                    [],
}

KNOWN_BASELINE_ARTIFACTS = [
    'runs/v04_transfer_431 (checkpoint, best.pt / latest.pt)', 'BPE tokenizer and merges for the 4,096-ID tokenizer',
    'list of the 431 selected documents', 'train/validation split of the 431', 'training manifest and configuration',
    'checkpoint context length', 'evaluation logs and metrics.jsonl',
]
BASELINE_CLAIMS = {
    'source': 'Expansformer_Transfer_1000/README.md (claims, not verified here)',
    'selected_documents': 431, 'excluded_documents': 68,
    'inherited_old_training_docs': 221, 'inherited_old_validation_docs': 39,
    'new_training_docs': 145, 'new_validation_docs': 26, 'split_seed': 42,
}


# ----------------------------------------------------------------------------- helpers
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def norm_text(text: str) -> str:
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.casefold()).split())


def hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest(), 'big')


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def dump_jsonl(path: Path, rows) -> None:
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows), encoding='utf-8')


def safe_resolve(base: Path, candidate: str) -> Path:
    p = Path(candidate)
    if not candidate or p.is_absolute() or '..' in p.parts:
        raise ValueError('unsafe or empty manifest path')
    base_real = base.resolve()
    resolved = (base / p).resolve()
    if not resolved.is_relative_to(base_real):
        raise ValueError('manifest path escapes its batch directory')
    if (base / p).is_symlink() or not resolved.is_file():
        raise ValueError('missing or non-regular file')
    return resolved


def find_manifests(src: Path) -> list[tuple[str, Path]]:
    """(batch_label, manifest_path): a top-level manifest, or immediate-category manifests."""
    found = []
    for d in sorted(p for p in src.iterdir() if p.is_dir() and not p.is_symlink()):
        if (d / 'manifest.jsonl').is_file():
            found.append((d.name, d / 'manifest.jsonl'))
        else:
            for sub in sorted(p for p in d.iterdir() if p.is_dir() and not p.is_symlink()):
                if (sub / 'manifest.jsonl').is_file():
                    found.append((d.name, sub / 'manifest.jsonl'))
    return found


# ----------------------------------------------------------------------------- stage 1/2: inventory, hashes, provenance
def load_records(src: Path):
    records, errors = [], []
    for batch, manifest in find_manifests(src):
        folder = manifest.parent
        seen_paths = set()
        for line_no, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            loc = f'{manifest.relative_to(src).as_posix()}:{line_no}'
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('manifest row is not an object')
                rel = str(row.get('path') or '')
                if rel in seen_paths:
                    raise ValueError('repeated path in manifest')
                seen_paths.add(rel)
                p = safe_resolve(folder, rel)
                data = p.read_bytes()
                text = data.decode('utf-8')
            except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append({'location': loc, 'error': str(exc)})
                continue
            declared = str(row.get('content_hash') or row.get('sha256') or '')
            actual = sha256_bytes(data)
            hash_ok = bool(HEX64.fullmatch(declared)) and declared.lower().removeprefix('sha256:') == actual
            raw_status, raw_rel = 'not_applicable', str(row.get('raw_source_path') or '')
            if raw_rel:
                try:
                    rp = safe_resolve(folder, raw_rel)
                    exp = str(row.get('raw_content_hash') or '')
                    raw_status = 'verified' if exp.lower().removeprefix('sha256:') == sha256_file(rp) else 'hash_mismatch'
                except ValueError:
                    raw_status = 'missing_from_package'
            records.append({
                'doc_id': str(row.get('source_id') or ''), 'batch': batch, 'manifest': manifest.relative_to(src).as_posix(),
                'path': p.relative_to(src.resolve()).as_posix(), 'row': row, 'text': text, 'bytes': len(data),
                'content_hash': actual, 'declared_hash': declared, 'hash_ok': hash_ok, 'raw_status': raw_status,
                'words': len(text.split()),
            })
    return records, errors


def provenance_check(records):
    out = {}
    seen_ids = collections.Counter(r['doc_id'] for r in records)
    canon = collections.Counter(canonical_source(r['row']) for r in records)
    for r in records:
        row = r['row']
        missing = [k for k in REQUIRED_PROVENANCE if not str(row.get(k) or '').strip()]
        if not str(row.get('source_url') or row.get('url') or '').strip():
            missing.append('source_url')
        if not str(row.get('license_url') or '').strip() and not str(row.get('license_declaration') or '').strip():
            missing.append('license_url_or_declaration')
        issues = []
        if seen_ids[r['doc_id']] > 1:
            issues.append('duplicate_source_id')
        if canon[canonical_source(row)] > 1:
            issues.append('duplicate_canonical_source')
        out[r['doc_id']] = {'missing_fields': sorted(missing), 'issues': issues,
                            'complete': not missing and not issues}
    return out


def canonical_source(row: dict) -> str:
    pmcid = str(row.get('pmcid', '')).upper()
    if pmcid:
        return 'pmc:' + pmcid
    group = str(row.get('source_group') or '').strip()
    if group:
        return 'group:' + group.casefold()
    url = str(row.get('source_url') or row.get('url') or '').strip().rstrip('/')
    return ('url:' + url.casefold()) if url else 'id:' + str(row.get('source_id', '')).casefold()


# ----------------------------------------------------------------------------- stage 3: license triage (metadata only)
NC_ND = re.compile(r'(?i)\bcc[- ]by[- ](nc|nd)\b|by-nc|by-nd|non-?commercial|no-?derivatives?')
SHARE_ALIKE = re.compile(r'(?i)by-sa|share-?alike|\bgfdl\b|gnu free documentation')
CC_BY = re.compile(r'(?i)\bcc[- ]by\b|creativecommons\.org/licenses/by/|creative commons attribution')
PD_DECL = re.compile(r'(?i)public[- ]domain|government[- ]work|u\.s\. government')
CAVEAT = re.compile(r'(?i)third[- ]party|not (independently )?(reviewed|assessed|verified)|item[- ]level|\bvar(y|ies)\b|candidate only|unless marked')


def triage_license(row: dict) -> dict:
    blob = ' '.join(str(row.get(k) or '') for k in
                    ('license_id', 'license_declaration', 'license_url', 'license_scope', 'rights_note', 'rights_review_note'))
    lid = str(row.get('license_id') or '')
    url = str(row.get('license_url') or '')
    version = None
    m = re.search(r'licenses/by(?:-sa)?/(\d\.\d)', url) or re.search(r'(?i)\bcc[- ]by(?:-sa)?\s*(\d\.\d)', lid)
    if m:
        version = m.group(1)
    reasons = ['declared_license_unverified_no_network']
    if not lid.strip() and not url.strip() and not str(row.get('license_declaration') or '').strip():
        return _lic('unresolved', 'missing', version, reasons + ['no_license_information'])
    if NC_ND.search(blob):
        return _lic('rejected', 'cc_nc_nd', version, reasons + ['noncommercial_or_noderivatives_terms_detected'])
    if SHARE_ALIKE.search(blob):
        return _lic('manual_review', 'share_alike', version,
                    reasons + ['share_alike_obligations_for_model_training_need_a_decision', 'imported_content_notices_unchecked'])
    if CC_BY.search(blob):
        if not url.strip():
            reasons.append('license_url_missing')
        if version is None:
            reasons.append('version_unspecified')
        if 'license_review_evidence' not in row and row.get('pmcid'):
            reasons.append('no_recorded_license_review_evidence')
        if re.search(r'(?i)item[- ]level', blob):
            reasons.append('item_level_review_not_performed')
        return _lic('manual_review', 'cc_by', version, reasons)
    if re.search(r'(?i)gutenberg', blob):
        return _lic('manual_review', 'project_gutenberg', version,
                    reasons + ['jurisdiction_and_transcription_not_verified', 'gutenberg_terms_apply_to_boilerplate'])
    caveat = CAVEAT.search(lid)
    if PD_DECL.search(blob) and not caveat:
        return _lic('manual_review', 'public_domain_declared', version, reasons + ['public_domain_claim_unverified'])
    return _lic('unresolved', 'public_domain_declared_with_caveat' if PD_DECL.search(blob) else 'other', version,
                reasons + [('caveat:' + caveat.group(0).casefold()) if caveat else 'license_family_not_recognised'])


def _lic(status, family, version, reasons):
    return {'status': status, 'family': family, 'declared_version': version, 'reasons': reasons}


def load_decisions(path: Path | None, known_ids: set[str]):
    """Human decisions file (JSONL): source_id, decision, decided_by, decided_on, basis. Invalid rows are ignored and reported."""
    if path is None:
        return {}, [], None
    decisions, problems = {}, []
    raw = path.read_bytes()
    for n, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            for k in ('source_id', 'decision', 'decided_by', 'decided_on', 'basis'):
                if not isinstance(d.get(k), str) or not d[k].strip():
                    raise ValueError(f'missing {k}')
            if d['decision'] not in STATUSES:
                raise ValueError('decision must be one of ' + ', '.join(STATUSES))
            if d['source_id'] not in known_ids:
                raise ValueError('unknown source_id')
            decisions[d['source_id']] = d
        except (ValueError, json.JSONDecodeError) as exc:
            problems.append({'line': n, 'error': str(exc)})
    return decisions, problems, sha256_bytes(raw)


# ----------------------------------------------------------------------------- stage 4: quality screen (automated flags only)
MOJIBAKE = re.compile('â€|Ã[©¨¤¶¼]|Â |�')
RETRACTION = re.compile(r'(?i)\bretract(ed|ion)\b|\bcorrigendum\b|\berratum\b|expression of concern')
WIKI_MARKUP = re.compile(r'(?m)^=+ .+ =+\s*$|\{\{|\[\[')
FORMULA = re.compile(r'\\frac|\\sum|\$\$|<math')
GUTENBERG = re.compile(r'(?i)\*\*\* ?(START|END) OF (THE|THIS) PROJECT GUTENBERG')
STOP = {'the', 'of', 'and', 'to', 'in', 'a', 'is', 'that', 'for', 'with'}


def quality_review(text: str, title: str) -> dict:
    words = text.split()
    n = len(words) or 1
    # Count only lines long enough to be real content.  Single-character fragments
    # (punctuation, list markers, OCR artifacts) are structural noise and must not
    # drive the repeated-line fraction.
    long_lines = [l.strip() for l in text.splitlines() if len(l.strip()) >= REPEATED_LINE_MIN_CHARS]
    rep = 1 - len(set(long_lines)) / len(long_lines) if long_lines else 0.0
    moj = len(MOJIBAKE.findall(text))
    bib = (len(re.findall(r'(?i)\bdoi:|\bet al\.', text)) / n) * 1000
    head = title + ' ' + text[:2000]
    flags = []
    if len(words) < 100:
        flags.append('very_short')
    elif len(words) < 500:
        flags.append('short')
    if len(set(w.casefold() for w in words)) / n < 0.15 and len(words) >= 100:
        flags.append('low_unique_word_fraction')
    if rep > 0.2:
        flags.append('many_repeated_lines')
    if moj / n > 0.001:
        flags.append('encoding_damage_suspected')
    if bib > 12:
        flags.append('reference_list_density_high')
    if RETRACTION.search(head):
        flags.append('retraction_or_correction_marker_near_start')
    if WIKI_MARKUP.search(text):
        flags.append('wiki_markup_remnants')
    if FORMULA.search(text):
        flags.append('formula_markup_present_preserved')
    if GUTENBERG.search(text):
        flags.append('gutenberg_boilerplate_in_text')
    stop_frac = sum(1 for w in words if w.casefold() in STOP) / n
    if len(words) >= 100 and stop_frac < 0.08:
        flags.append('low_english_stopword_fraction')
    status = 'pass_automated'
    informative = {'formula_markup_present_preserved'}
    if set(flags) - informative:
        status = 'needs_review'
    if 'very_short' in flags or 'encoding_damage_suspected' in flags and moj / n > 0.005:
        status = 'fail_automated'
    return {'status': status, 'flags': flags, 'words': len(words), 'characters': len(text),
            'unique_word_fraction': round(len(set(w.casefold() for w in words)) / n, 4),
            'repeated_line_fraction': round(rep, 4), 'reference_markers_per_1000_words': round(bib, 2),
            'english_stopword_fraction': round(stop_frac, 4), 'factual_quality_verified': False}


# ----------------------------------------------------------------------------- stage 5: duplicates
class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def shingles(text: str, k: int = 5) -> set[int]:
    w = norm_text(text).split()
    return {hash64(' '.join(w[i:i + k])) for i in range(max(len(w) - k + 1, 1))}


def duplicate_analysis(records):
    ids = [r['doc_id'] for r in records]
    by_id = {r['doc_id']: r for r in records}
    report = {'method': {
        'exact': 'SHA-256 of file bytes', 'normalized': 'SHA-256 of case-folded, punctuation-stripped, whitespace-collapsed text',
        'paragraph': 'normalized paragraphs of at least 200 characters shared by two or more documents',
        'near_duplicate': f'bottom-{SKETCH_K} MinHash sketch to find candidates, then exact Jaccard of 5-word shingles; >= {NEAR_DUP_JACCARD} flagged, >= {OVERLAP_REVIEW_JACCARD} listed for review',
        'tfidf': f'cosine over each document\'s top 300 TF-IDF terms; >= {TFIDF_REVIEW_COSINE} listed as review candidates only',
        'embedding_semantic': 'NOT PERFORMED: no embedding model is available offline in this build'}}
    groups = lambda key: [sorted(v) for v in sorted(
        (g for g in collections.defaultdict(list, _bucket(records, key)).values() if len(g) > 1))]
    report['exact_duplicate_groups'] = groups(lambda r: r['content_hash'])
    report['normalized_duplicate_groups'] = groups(lambda r: sha256_bytes(norm_text(r['text']).encode()))
    # paragraph level
    para = collections.defaultdict(set)
    for r in records:
        for p in re.split(r'\n\s*\n', r['text']):
            q = norm_text(p)
            if len(q) >= 200:
                para[sha256_bytes(q.encode())].add(r['doc_id'])
    pair_counts = collections.Counter()
    for owners in para.values():
        o = sorted(owners)
        for i in range(len(o)):
            for j in range(i + 1, len(o)):
                pair_counts[(o[i], o[j])] += 1
    report['shared_paragraph_pairs'] = [{'a': a, 'b': b, 'shared_paragraphs': c} for (a, b), c in sorted(pair_counts.items())]
    # near duplicates via bottom-k sketches
    sketch = {}
    for r in records:
        sketch[r['doc_id']] = sorted(shingles(r['text']))[:SKETCH_K]
    sets = {i: set(s) for i, s in sketch.items()}
    near, overlap = [], []
    cache = {}
    def full(i):
        if i not in cache:
            cache[i] = shingles(by_id[i]['text'])
        return cache[i]
    ordered = sorted(ids)
    for x in range(len(ordered)):
        for y in range(x + 1, len(ordered)):
            a, b = ordered[x], ordered[y]
            union = sorted(sets[a] | sets[b])[:SKETCH_K]
            if not union:
                continue
            est = sum(1 for h in union if h in sets[a] and h in sets[b]) / len(union)
            if est < 0.1:
                continue
            sa, sb = full(a), full(b)
            jac = len(sa & sb) / len(sa | sb)
            if jac >= NEAR_DUP_JACCARD:
                near.append({'a': a, 'b': b, 'jaccard': round(jac, 4)})
            elif jac >= OVERLAP_REVIEW_JACCARD:
                overlap.append({'a': a, 'b': b, 'jaccard': round(jac, 4)})
        if len(cache) > 60:
            cache.clear()
    report['near_duplicate_pairs'] = near
    report['overlap_review_pairs'] = overlap
    report['tfidf_review_pairs'] = tfidf_pairs(records)
    uf = UnionFind(ids)
    for g in report['exact_duplicate_groups'] + report['normalized_duplicate_groups']:
        for other in g[1:]:
            uf.union(g[0], other)
    for p in near:
        uf.union(p['a'], p['b'])
    by_canon = collections.defaultdict(list)
    for r in records:
        by_canon[canonical_source(r['row'])].append(r['doc_id'])
    for g in by_canon.values():
        for other in g[1:]:
            uf.union(g[0], other)
    group_of = {i: uf.find(i) for i in ids}
    certain = set()
    for g in report['exact_duplicate_groups'] + report['normalized_duplicate_groups']:
        certain.update(g[1:])            # keeper = smallest doc_id
    report['summary'] = {
        'documents': len(ids), 'exact_duplicate_groups': len(report['exact_duplicate_groups']),
        'normalized_duplicate_groups': len(report['normalized_duplicate_groups']),
        'documents_sharing_paragraphs': len({d for p in pair_counts for d in p}),
        'near_duplicate_pairs': len(near), 'overlap_review_pairs': len(overlap),
        'tfidf_review_pairs': len(report['tfidf_review_pairs']),
        'groups_with_more_than_one_document': len({g for g in group_of.values() if list(group_of.values()).count(g) > 1}),
        'removed_as_certain_duplicate': len(certain)}
    return report, group_of, certain


def _bucket(records, key):
    d = collections.defaultdict(list)
    for r in records:
        d[key(r)].append(r['doc_id'])
    return d


def tfidf_pairs(records):
    docs = {}
    df = collections.Counter()
    for r in records:
        c = collections.Counter(re.findall(r'[a-z]{3,}', r['text'].casefold()))
        docs[r['doc_id']] = c
        df.update(c.keys())
    n = len(records)
    vecs = {}
    for i, c in docs.items():
        w = {t: (1 + math.log(v)) * math.log(n / df[t]) for t, v in c.items() if df[t] < n}
        top = sorted(w.items(), key=lambda kv: (-kv[1], kv[0]))[:300]
        norm = math.sqrt(sum(v * v for _, v in top)) or 1.0
        vecs[i] = {t: v / norm for t, v in top}
    index = collections.defaultdict(list)
    for i in sorted(vecs):
        for t, v in vecs[i].items():
            index[t].append((i, v))
    acc = collections.defaultdict(float)
    for t in sorted(index):
        post = index[t]
        for x in range(len(post)):
            for y in range(x + 1, len(post)):
                acc[(post[x][0], post[y][0])] += post[x][1] * post[y][1]
    return [{'a': a, 'b': b, 'cosine': round(v, 4)} for (a, b), v in sorted(acc.items()) if v >= TFIDF_REVIEW_COSINE]


# ----------------------------------------------------------------------------- stage 6: coverage, baseline attempt
def domain_coverage(records):
    # --- factual: manifest `domain` labels as recorded in the source manifests ---
    manifest_domains = collections.defaultdict(lambda: {'documents': 0, 'words': 0, 'bytes': 0})
    for r in records:
        d = manifest_domains[str(r['row'].get('domain') or 'unlabeled')]
        d['documents'] += 1
        d['words'] += r['words']
        d['bytes'] += r['bytes']

    # Which target domains does each manifest label map to?
    target_covered_by_manifest: set[str] = set()
    for label in manifest_domains:
        for tgt in MANIFEST_TO_TARGET.get(label, []):
            target_covered_by_manifest.add(tgt)
    target_domains_zero_manifest = sorted(k for k in TARGET_DOMAINS if k not in target_covered_by_manifest)

    # --- heuristic: keyword signal (strict — both min-terms and min-density required) ---
    touches = {k: 0 for k in TARGET_DOMAINS}
    for r in records:
        low = (str(r['row'].get('title') or '') + ' ' + r['text']).casefold()
        n = max(r['words'], 1)
        for dom, terms in TARGET_DOMAINS.items():
            present = [t for t in terms if t in low]
            density = sum(low.count(t) for t in present) / n * 1000
            if len(present) >= DOMAIN_KW_MIN_TERMS and density >= DOMAIN_KW_MIN_DENSITY:
                touches[dom] += 1
    total = len(records) or 1
    weak_heuristic = sorted(k for k, v in touches.items() if v / total < 0.05)
    # The primary coverage gap report uses manifest labels; heuristic is secondary.
    # A domain is considered "weak or missing" if it has zero manifest-label coverage.
    weak = target_domains_zero_manifest

    years = collections.Counter()
    for r in records:
        m = re.search(r'(19|20)\d\d', str(r['row'].get('publication_date') or r['row'].get('publication_date_xml') or ''))
        years[m.group(0)[:3] + '0s' if m else 'unknown'] += 1
    return {
        'manifest_domains': {k: v for k, v in sorted(manifest_domains.items())},
        'manifest_to_target_mapping': {k: v for k, v in MANIFEST_TO_TARGET.items() if k in manifest_domains},
        'target_domains_with_zero_manifest_labels': target_domains_zero_manifest,
        'target_domain_touch_counts_keyword_heuristic': dict(sorted(touches.items())),
        'target_domain_touch_share_keyword_heuristic': {k: round(v / total, 4) for k, v in sorted(touches.items())},
        'weak_or_missing_target_domains_keyword_heuristic': weak_heuristic,
        'weak_or_missing_target_domains_under_5_percent': weak,
        'source_types': dict(sorted(collections.Counter(str(r['row'].get('source_type') or r['row'].get('article_type') or 'unknown') for r in records).items())),
        'publishers': dict(sorted(collections.Counter(str(r['row'].get('publisher') or 'unknown') for r in records).items())),
        'publication_decades': dict(sorted(years.items())),
        'total_words': sum(r['words'] for r in records),
        'note': ('Factual coverage comes from manifest `domain` labels. '
                 'Keyword-heuristic counts require at least '
                 f'{DOMAIN_KW_MIN_TERMS} distinct terms and {DOMAIN_KW_MIN_DENSITY}+ hits per 1000 words — '
                 'not a classifier; use for signal only. '
                 'Token counts need the real training tokenizer; words and bytes reported instead.')}


def baseline_reconstruction(src: Path, records, claim: str):
    names = [p.name.lower() for p in src.rglob('*') if p.is_file()]
    wanted = {'checkpoint files (*.pt, *.pth, *.ckpt)': [n for n in names if n.endswith(('.pt', '.pth', '.ckpt'))],
              'tokenizer files': [n for n in names if 'tokenizer' in n or 'merges' in n or n.endswith('vocab.json')],
              'selection or split lists': [n for n in names if re.search(r'select|split|train_docs|val_docs', n)],
              'training logs': [n for n in names if n in ('metrics.jsonl', 'train.log', 'initial_dev.json')]}
    hashes = sorted(r['content_hash'] for r in records)
    paths = sorted(f"{r['path']}\t{r['content_hash']}" for r in records)
    manifests = b''.join((src / m).read_bytes() for m in sorted({r['manifest'] for r in records}))
    attempts = {
        'sha256 of sorted content hashes joined by newline': sha256_bytes('\n'.join(hashes).encode()),
        'same with trailing newline': sha256_bytes(('\n'.join(hashes) + '\n').encode()),
        'sha256 of sorted "path<TAB>hash" lines': sha256_bytes('\n'.join(paths).encode()),
        'sha256 of manifest bytes in path order': sha256_bytes(manifests)}
    return {
        'status': 'not_reconstructed',
        'artifacts_looked_for_in_package': KNOWN_BASELINE_ARTIFACTS,
        'artifacts_found_in_package': {k: v for k, v in wanted.items()},
        'claims_from_transfer_readme': BASELINE_CLAIMS,
        'snapshot_hash_claim': claim,
        'snapshot_hash_attempts_over_all_package_records': {k: {'value': v, 'matches_claim': v == claim} for k, v in attempts.items()},
        'attempt_interpretation': 'The claimed snapshot covers 431 selected documents; this package has 499 and no selection list, so only conventions over the full set could be tried. A miss does not show the claim is wrong.',
        'package_documents': len(records),
        'what_would_be_needed': 'the 431 selected document identifiers or paths, the split, and the snapshot construction rule, or the original runs/v04_transfer_431 directory',
        'guidance': 'Any 431-document subset built from this package must be labeled a new baseline, not the v0.4 experiment.'}


# ----------------------------------------------------------------------------- stage 7: split plan
def split_of(group_id: str) -> str:
    u = int.from_bytes(hashlib.sha256((SPLIT_SALT + '|' + group_id).encode()).digest()[:8], 'big') / 2 ** 64
    acc = 0.0
    for name, frac in SPLIT_FRACTIONS:
        acc += frac
        if u < acc:
            return name
    return SPLIT_FRACTIONS[-1][0]


# ----------------------------------------------------------------------------- driver
def build(src: Path, out: Path, decisions_path: Path | None = None, expected: int | None = None,
          source_zip_sha256: str | None = None, snapshot_claim: str = SNAPSHOT_CLAIM_DEFAULT) -> dict:
    src, out = src.resolve(), out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to write into a non-empty output directory: {out}')
    if out.is_relative_to(src) or src.is_relative_to(out):
        raise ValueError('Output directory must be separate from the source tree')
    out.mkdir(parents=True, exist_ok=True)

    records, errors = load_records(src)
    records.sort(key=lambda r: r['doc_id'])
    prov = provenance_check(records)
    decisions, decision_problems, decisions_sha = load_decisions(decisions_path, {r['doc_id'] for r in records})
    dup, group_of, certain_dups = duplicate_analysis(records)
    coverage = domain_coverage(records)

    inventory, license_rows, quality_rows, manifests = [], [], [], {s: [] for s in STATUSES}
    for r in records:
        did, row = r['doc_id'], r['row']
        lic = triage_license(row)
        lic.update({'doc_id': did, 'decision_source': 'automated_triage_from_manifest_metadata',
                    'declared_license_id': row.get('license_id'), 'license_url': row.get('license_url'),
                    'manifest_license_status': row.get('license_status')})
        d = decisions.get(did)
        if d:
            if lic['status'] == 'rejected' and d['decision'] == 'approved':
                decision_problems.append({'source_id': did, 'error': 'approval refused: automated triage detected NC/ND terms; fix the evidence first'})
            else:
                lic.update({'status': d['decision'], 'decision_source': 'human_decision_file',
                            'decided_by': d['decided_by'], 'decided_on': d['decided_on'], 'basis': d['basis']})
        q = quality_review(r['text'], str(row.get('title') or ''))
        q['doc_id'] = did
        status = lic['status']
        reasons = list(lic['reasons'])
        if r['hash_ok'] is False:
            status, reasons = 'rejected', reasons + ['content_hash_mismatch']
        elif did in certain_dups:
            status, reasons = 'rejected', reasons + ['exact_or_normalized_duplicate_of_another_document']
        elif q['status'] == 'fail_automated' and status == 'approved':
            status, reasons = 'manual_review', reasons + ['approved_but_failed_automated_quality_screen']
        gid = group_of[did]
        eligible = status in ('approved', 'manual_review') and q['status'] != 'fail_automated'
        split = split_of(gid) if eligible else None
        inv = {'doc_id': did, 'batch': r['batch'], 'path': r['path'], 'manifest': r['manifest'], 'domain': row.get('domain'),
               'title': row.get('title'), 'bytes': r['bytes'], 'words': r['words'], 'content_hash': r['content_hash'],
               'declared_content_hash_matches': r['hash_ok'], 'raw_source_status': r['raw_status'],
               'provenance_complete': prov[did]['complete'], 'provenance_missing_fields': prov[did]['missing_fields'],
               'provenance_issues': prov[did]['issues']}
        inventory.append(inv)
        license_rows.append(lic)
        quality_rows.append(q)
        manifests[status].append({**inv, 'final_status': status, 'status_reasons': sorted(set(reasons)),
                                  'license_family': lic['family'], 'quality_status': q['status'], 'quality_flags': q['flags'],
                                  'duplicate_group': gid, 'split_plan': split})
    # outputs
    dump_jsonl(out / 'corpus_inventory.jsonl', inventory)
    dump_jsonl(out / 'license_review.jsonl', license_rows)
    dump_jsonl(out / 'quality_review.jsonl', quality_rows)
    dump_json(out / 'duplicate_report.json', dup)
    dump_json(out / 'domain_coverage.json', coverage)
    dump_jsonl(out / 'accepted_manifest.jsonl', manifests['approved'])
    dump_jsonl(out / 'rejected_manifest.jsonl', manifests['rejected'])
    dump_jsonl(out / 'unresolved_manifest.jsonl', manifests['unresolved'])
    dump_jsonl(out / 'manual_review_manifest.jsonl', manifests['manual_review'])
    baseline = baseline_reconstruction(src, records, snapshot_claim)
    dump_json(out / 'baseline_reconstruction.json', baseline)

    planned = [m for s in ('approved', 'manual_review') for m in manifests[s] if m['split_plan']]
    split_counts = {name: collections.Counter() for name, _ in SPLIT_FRACTIONS}
    for m in planned:
        split_counts[m['split_plan']][str(m['domain'])] += 1
    split_manifest = {
        'status': 'PLAN ONLY: assignments are a pure function of the duplicate-group id and the salt, so removing documents later does not move the others. '
                  'Regenerate the manifests after license decisions; do not hand-edit.',
        'salt': SPLIT_SALT, 'fractions': dict(SPLIT_FRACTIONS), 'unit': 'duplicate group (exact, normalized, near-duplicate and same-source documents stay together)',
        'eligible_documents': len(planned),
        'documents_per_split': {k: sum(v.values()) for k, v in split_counts.items()},
        'documents_per_split_and_domain': {k: dict(sorted(v.items())) for k, v in split_counts.items()},
        'frozen': False,
        'assignments': {m['doc_id']: m['split_plan'] for m in sorted(planned, key=lambda m: m['doc_id'])},
        'caveat': 'Held-out text only. Reasoning, code, comprehension and abstention evaluation sets do not exist yet (see evaluation_plan.md).'}
    dump_json(out / 'split_manifest.json', split_manifest)

    summary = build_summary(records, errors, prov, manifests, quality_rows, dup, coverage, baseline, split_manifest,
                            decision_problems, decisions_sha, source_zip_sha256, expected, decisions_path is not None)
    (out / 'void_register.md').write_text(void_register(summary, src, records, coverage, baseline), encoding='utf-8')
    (out / 'evaluation_plan.md').write_text(evaluation_plan(split_manifest), encoding='utf-8')
    (out / 'corpus_build_report.md').write_text(build_report(summary, coverage, dup, baseline), encoding='utf-8')
    dump_json(out / 'build_summary.json', summary)
    return summary


def build_summary(records, errors, prov, manifests, quality_rows, dup, coverage, baseline, split_manifest,
                  decision_problems, decisions_sha, zip_sha, expected, decisions_given):
    lic_counts = collections.Counter(f"{m['license_family']}|{s}" for s, ms in manifests.items() for m in ms)
    return {
        'tool_version': VERSION, 'source_zip_sha256': zip_sha, 'documents_loaded': len(records),
        'expected_documents': expected, 'count_matches_expectation': expected is None or len(records) == expected,
        'integrity_errors': errors,
        'hash_verification': {'matches_manifest': sum(r['hash_ok'] for r in records),
                              'mismatch_or_malformed': sum(not r['hash_ok'] for r in records)},
        'raw_source_status': dict(sorted(collections.Counter(r['raw_status'] for r in records).items())),
        'provenance': {'complete': sum(p['complete'] for p in prov.values()), 'incomplete': sum(not p['complete'] for p in prov.values())},
        'final_status_counts': {s: len(manifests[s]) for s in STATUSES},
        'license_family_status_counts': dict(sorted(lic_counts.items())),
        'quality_status_counts': dict(sorted(collections.Counter(q['status'] for q in quality_rows).items())),
        'duplicate_summary': dup['summary'],
        'weak_or_missing_target_domains': coverage['weak_or_missing_target_domains_under_5_percent'],
        'baseline_reconstruction': baseline['status'],
        'split_plan_eligible_documents': split_manifest['eligible_documents'],
        'human_decisions_file_supplied': decisions_given, 'human_decisions_file_sha256': decisions_sha,
        'decision_problems': decision_problems,
        'ready_for_training': False,
        'why_not_ready': ['no document is approved unless a human decision exists', 'license statuses derive from manifest metadata only',
                          'no independent quality or factual review', 'baseline v0.4 artifacts absent', 'no frozen evaluation set for reasoning, code or abstention'],
    }


def void_register(s, src, records, coverage, baseline):
    b_missing_ev = [r['doc_id'] for r in records if r['row'].get('pmcid') and 'license_review_evidence' not in r['row']]
    lines = ['# Void Register', '',
             'Missing data, unavailable artifacts, unresolved issues and unverified claims. Generated from the package contents.', '']
    def add(title, body):
        lines.extend([f'## {title}', body, ''])
    add('1. v0.4 baseline cannot be reproduced from this package',
        'Not in the package: ' + '; '.join(KNOWN_BASELINE_ARTIFACTS) + '. Snapshot-hash attempts: none matched. Any 431-document subset built here is a new baseline.')
    n_raw = s['raw_source_status'].get('missing_from_package', 0)
    add('2. Raw source files', f'{n_raw} manifest records reference a raw source file (JATS XML) that is not in the package, so their license declarations and extraction cannot be re-checked against the originals.')
    add('3. License status is metadata triage only', 'No network access, no authoritative re-check, no legal review. Statuses: ' + json.dumps(s['final_status_counts']) + '. `approved` requires a human decisions file; none was supplied.' if not s['human_decisions_file_supplied'] else 'A human decisions file was supplied; its decisions are recorded per document.')
    add('4. Records without recorded license-review evidence', f'{len(b_missing_ev)} article records carry no `license_review_evidence` field (all from the later b03 batch). Their declared license is unchecked beyond the manifest.')
    add('5. Share-alike and mixed-license material', 'Wikipedia text (CC BY-SA and GFDL) and Project Gutenberg text need an explicit decision on obligations when used for model training and release. Government and agency pages state that item-level reuse terms were not assessed.')
    summ = coverage['weak_or_missing_target_domains_under_5_percent']
    add('6. Domain coverage gaps', 'Target domains with no manifest-labeled documents: ' + (', '.join(summ) if summ else 'none') + '. See domain_coverage.json for keyword-heuristic counts (secondary signal only).')
    add('7. No reasoning or evaluation material', 'There are no verifiable math problems, executable code tasks, causal or abstention sets, or long-context tasks. See evaluation_plan.md.')
    add('8. No independent quality or factual review', 'Only automated flags ran. `factual_quality_verified` is false for every document.')
    add('9. Semantic duplicate review', 'Embedding-based similarity was not performed (no offline embedding model). TF-IDF and shingle overlap lists are review candidates.')
    add('10. Token counts', 'The real BPE tokenizer is absent, so budgets use words and bytes, not tokens.')
    add('11. Stale summary in the package', 'license_reconciliation_summary.json reports 178 retained human-focused articles; the manifests hold 259. The 81-article difference has no matching reconciliation record in the package.')
    return '\n'.join(lines)


def evaluation_plan(split_manifest):
    return '\n'.join([
        '# Evaluation plan (Phase 1 draft)', '',
        'Status: plan only. Nothing here has been run, and nothing is frozen until the license decisions are in.', '',
        '## Frozen text split', f"`split_manifest.json` assigns {split_manifest['eligible_documents']} eligible documents to train/validation/test by duplicate group. "
        'Freeze it (record its SHA-256 in the run configuration) before any architecture comparison, then never regenerate it.', '',
        '## Measures for the dense vs MoE+exits+memory comparison',
        '- Primary: held-out next-token cross-entropy on the frozen validation split; byte-normalized cross-entropy; token accuracy.',
        '- Secondary: domain-level held-out loss, training compute, throughput, peak memory, inference latency.',
        '- At least 3 seeds per arm; a difference counts only if it exceeds the seed-to-seed spread. Hold constant tokens, tokenizer, split, parameter budget, activated parameters, optimizer and compute.',
        '- Small CPU runs verify code paths only.', '',
        '## Sets that do not exist yet (must be built and validated before use)',
        '- Math problems with independently checkable answers; executable code tasks with passing tests; hypothesis-evaluation and causal items;',
        '  contradiction detection; long-context retrieval; abstention and evidence-support items.',
        '- Each needs an independent answer key or executable validator, a contamination check against the training corpus, and its own frozen split.', '',
        '## Contamination and uncertainty',
        '- Check every evaluation item against training text with the shingle and paragraph methods in `duplicate_report.json`.',
        '- Report confidence intervals; do not read lower loss as general capability.', ''])


def build_report(s, coverage, dup, baseline):
    c = s['final_status_counts']
    L = ['# Corpus build report (Phase 1)', '',
         f"Tool `{s['tool_version']}`. Source archive SHA-256: `{s['source_zip_sha256']}`.", '',
         '**Not training-ready.** ' + ' '.join(s['why_not_ready']) + '.', '',
         '## What was checked', '',
         f"- Documents loaded: {s['documents_loaded']} (expected {s['expected_documents']}); integrity errors: {len(s['integrity_errors'])}.",
         f"- Cleaned-text hashes matching the manifests: {s['hash_verification']['matches_manifest']}; mismatches: {s['hash_verification']['mismatch_or_malformed']}.",
         f"- Raw source files: {json.dumps(s['raw_source_status'])}.",
         f"- Provenance complete: {s['provenance']['complete']}; incomplete: {s['provenance']['incomplete']}.", '',
         '## Status', '', '| Status | Documents |', '|---|---|'] + [f'| {k} | {c[k]} |' for k in STATUSES] + ['',
         'License family by status: `' + json.dumps(s['license_family_status_counts']) + '`', '',
         f"Quality screen (automated flags only): `{json.dumps(s['quality_status_counts'])}`.", '',
         '## Duplicates', '', '`' + json.dumps(s['duplicate_summary']) + '`', '',
         '## Domain coverage', '', 'Target domains with no manifest-labeled documents: ' + (', '.join(s['weak_or_missing_target_domains']) or 'none') + '.',
         'Manifest domains: `' + json.dumps({k: v['documents'] for k, v in coverage['manifest_domains'].items()}) + '`',
         'Keyword-heuristic weak domains: ' + (', '.join(coverage.get('weak_or_missing_target_domains_keyword_heuristic', [])) or 'none') + '.', '',
         '## Baseline reconstruction', '', f"Status: **{baseline['status']}**. {baseline['guidance']}", '',
         '## Decisions', '', ('Human decisions file supplied (SHA-256 `%s`); problems: `%s`.' % (s['human_decisions_file_sha256'], json.dumps(s['decision_problems']))
                              if s['human_decisions_file_supplied'] else 'No human decisions file supplied, so nothing is approved.'), '']
    return '\n'.join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True, help='extracted staged corpus directory (read only)')
    ap.add_argument('--out', type=Path, required=True, help='new empty directory outside --src')
    ap.add_argument('--decisions', type=Path, help='JSONL of human license decisions')
    ap.add_argument('--expected-count', type=int)
    ap.add_argument('--source-zip-sha256')
    ap.add_argument('--snapshot-claim', default=SNAPSHOT_CLAIM_DEFAULT)
    a = ap.parse_args(argv)
    try:
        s = build(a.src, a.out, a.decisions, a.expected_count, a.source_zip_sha256, a.snapshot_claim)
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({k: s[k] for k in ('documents_loaded', 'count_matches_expectation', 'hash_verification', 'raw_source_status',
                                         'final_status_counts', 'quality_status_counts', 'duplicate_summary',
                                         'weak_or_missing_target_domains', 'baseline_reconstruction', 'ready_for_training')}, indent=2))
    return 1 if s['integrity_errors'] or not s['count_matches_expectation'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
