"""Tests for build_corpus_candidate.py — Phase 1 corpus pipeline.

Run from the repository root:
    python -m unittest discover -s tests -v

Each test has a declared contract:
    Difference: what observable outcome proves this test works.
    Guard fires when: the condition that makes the test fail.
All tests use only the standard library and a lightweight in-memory build
(no network access, no external corpus).  Slow checks (full build) are
marked as integration tests and can be skipped by setting SKIP_SLOW=1.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
import build_corpus_candidate as bcc


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

MINIMAL_ROW = {
    'source_id': 'doc-001',
    'title': 'A test document about machine learning',
    'publisher': 'Test Publisher',
    'license_id': 'CC BY 4.0',
    'license_url': 'https://creativecommons.org/licenses/by/4.0/',
    'content_hash': '',          # filled per document
    'retrieved_at': '2024-01-01',
    'domain': 'computer_science_software_algorithms',
    'source_url': 'https://example.com/doc001',
}

LOREM = (
    'Artificial intelligence and machine learning form the backbone of modern '
    'language processing systems. Neural networks are trained using backpropagation '
    'and gradient descent to minimize loss. Deep learning architectures like transformers '
    'have achieved strong results on language modeling tasks. Training data quality '
    'matters as much as architecture choice, and overfitting is a persistent challenge. '
) * 10   # ~500 words, enough to pass the short-document guard


def make_src(tmp: Path, docs: list[dict]) -> Path:
    """Write a minimal staged-corpus directory with one batch."""
    batch = tmp / 'batch01'
    batch.mkdir()
    manifest_rows = []
    for d in docs:
        text = d['text']
        data = text.encode('utf-8')
        digest = hashlib.sha256(data).hexdigest()
        fname = d['source_id'] + '.txt'
        (batch / fname).write_bytes(data)
        row = {**MINIMAL_ROW, **d.get('overrides', {})}
        row['source_id'] = d['source_id']
        row['content_hash'] = digest
        row['path'] = fname
        row.pop('text', None)
        row.pop('overrides', None)
        manifest_rows.append(row)
    (batch / 'manifest.jsonl').write_text(
        ''.join(json.dumps(r) + '\n' for r in manifest_rows), encoding='utf-8')
    return tmp / 'src'


def make_single_doc_src(tmp: Path, source_id='doc-001', text=None, row_overrides=None) -> Path:
    src = tmp / 'src_single'
    src.mkdir()
    batch = src / 'batch01'
    batch.mkdir()
    t = text or LOREM
    data = t.encode('utf-8')
    digest = hashlib.sha256(data).hexdigest()
    fname = source_id + '.txt'
    (batch / fname).write_bytes(data)
    row = {**MINIMAL_ROW, 'source_id': source_id, 'content_hash': digest, 'path': fname}
    if row_overrides:
        row.update(row_overrides)
    (batch / 'manifest.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    return src


# ---------------------------------------------------------------------------
# 1. Safety: path traversal is refused
# ---------------------------------------------------------------------------

class TestPathTraversal(unittest.TestCase):
    """Contract: safe_resolve raises ValueError on any path that could escape the batch directory.
    Guard fires when: a path like '../other_batch/secret.txt' is accepted."""

    def _check(self, candidate):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / 'batch'
            base.mkdir()
            with self.assertRaises(ValueError, msg=f'should reject: {candidate!r}'):
                bcc.safe_resolve(base, candidate)

    def test_absolute_path_rejected(self):
        self._check('/etc/passwd')

    def test_dotdot_escapes_rejected(self):
        self._check('../other/file.txt')

    def test_empty_path_rejected(self):
        self._check('')


# ---------------------------------------------------------------------------
# 2. Hash mismatch detection
# ---------------------------------------------------------------------------

class TestHashMismatch(unittest.TestCase):
    """Contract: a file whose bytes differ from the manifest content_hash is marked hash_ok=False.
    Guard fires when: a tampered file is silently accepted."""

    def test_tampered_file_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            batch = Path(d) / 'batch01'
            batch.mkdir()
            fname = 'doc-001.txt'
            (batch / fname).write_text('original content ' + LOREM, encoding='utf-8')
            # Declare a hash that does NOT match the actual file bytes
            wrong_hash = hashlib.sha256(b'completely different content').hexdigest()
            row = {**MINIMAL_ROW, 'source_id': 'doc-001', 'content_hash': wrong_hash, 'path': fname}
            (batch / 'manifest.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
            records, errors = bcc.load_records(Path(d))
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]['hash_ok'], 'tampered file must have hash_ok=False')

    def test_correct_hash_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            src = make_single_doc_src(Path(d))
            records, errors = bcc.load_records(src)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]['hash_ok'])


# ---------------------------------------------------------------------------
# 3. License triage table
# ---------------------------------------------------------------------------

class TestLicenseTriage(unittest.TestCase):
    """Contract: triage_license maps specific manifest fields to the correct status and family.
    Guard fires when: a NC/ND license is not rejected, or share-alike goes straight to approved."""

    def test_cc_by_nc_is_rejected(self):
        r = bcc.triage_license({'license_id': 'CC BY-NC 4.0', 'license_url': 'https://creativecommons.org/licenses/by-nc/4.0/'})
        self.assertEqual(r['status'], 'rejected')
        self.assertEqual(r['family'], 'cc_nc_nd')

    def test_cc_by_nd_is_rejected(self):
        r = bcc.triage_license({'license_id': 'CC BY-ND 4.0'})
        self.assertEqual(r['status'], 'rejected')

    def test_cc_by_sa_is_manual_review(self):
        r = bcc.triage_license({'license_id': 'CC BY-SA 4.0', 'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/'})
        self.assertEqual(r['status'], 'manual_review')
        self.assertEqual(r['family'], 'share_alike')

    def test_cc_by_is_manual_review_not_approved(self):
        r = bcc.triage_license({'license_id': 'CC BY 4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/'})
        self.assertEqual(r['status'], 'manual_review')
        self.assertEqual(r['family'], 'cc_by')

    def test_empty_license_is_unresolved(self):
        r = bcc.triage_license({})
        self.assertEqual(r['status'], 'unresolved')

    def test_public_domain_without_caveat_is_manual_review(self):
        r = bcc.triage_license({'license_id': 'public domain', 'license_declaration': 'This work is in the public domain.'})
        self.assertEqual(r['status'], 'manual_review')
        self.assertEqual(r['family'], 'public_domain_declared')

    def test_gutenberg_is_manual_review(self):
        r = bcc.triage_license({'license_id': 'Project Gutenberg license'})
        self.assertEqual(r['status'], 'manual_review')
        self.assertEqual(r['family'], 'project_gutenberg')


# ---------------------------------------------------------------------------
# 4. Approval only via valid decisions file; NC/ND cannot be approved
# ---------------------------------------------------------------------------

class TestDecisions(unittest.TestCase):
    """Contract: a document's status is `approved` ONLY when a valid human decision says so;
    an NC/ND detected document CANNOT be approved even with a decision.
    Guard fires when: automated triage is bypassed by a decisions file entry."""

    def _build_with_decision(self, tmp, lic_row_overrides, decision):
        src = make_single_doc_src(Path(tmp), source_id='doc-001', row_overrides=lic_row_overrides)
        out = Path(tmp) / 'out'
        dec_file = Path(tmp) / 'decisions.jsonl'
        dec_file.write_text(json.dumps(decision) + '\n', encoding='utf-8')
        s = bcc.build(src, out, decisions_path=dec_file)
        return s

    def test_no_decisions_file_means_nothing_approved(self):
        with tempfile.TemporaryDirectory() as d:
            src = make_single_doc_src(Path(d))
            out = Path(d) / 'out'
            s = bcc.build(src, out)
        self.assertEqual(s['final_status_counts']['approved'], 0)

    def test_valid_decision_approves_cc_by(self):
        with tempfile.TemporaryDirectory() as d:
            decision = {
                'source_id': 'doc-001',
                'decision': 'approved',
                'decided_by': 'test@example.com',
                'decided_on': '2024-06-01',
                'basis': 'CC BY 4.0 confirmed acceptable',
            }
            s = self._build_with_decision(
                d,
                {'license_id': 'CC BY 4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/'},
                decision)
        self.assertEqual(s['final_status_counts']['approved'], 1)

    def test_nc_nd_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as d:
            decision = {
                'source_id': 'doc-001',
                'decision': 'approved',
                'decided_by': 'test@example.com',
                'decided_on': '2024-06-01',
                'basis': 'trying to approve NC',
            }
            s = self._build_with_decision(
                d,
                {'license_id': 'CC BY-NC 4.0'},
                decision)
        # NC/ND approval is refused; document stays rejected
        self.assertGreater(s['decision_problems'].__len__(), 0, 'NC/ND override must produce a decision_problem')
        self.assertEqual(s['final_status_counts']['approved'], 0)


# ---------------------------------------------------------------------------
# 5. Exact and normalized duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicates(unittest.TestCase):
    """Contract: two documents with identical bytes end up in an exact_duplicate_group;
    the second (by doc_id sort) is marked as a certain duplicate and rejected.
    Guard fires when: identical content passes silently into the candidate set."""

    def _make_dup_src(self, tmp: Path) -> Path:
        src = tmp / 'src_dup'
        src.mkdir()
        batch = src / 'batch01'
        batch.mkdir()
        text = LOREM
        data = text.encode('utf-8')
        digest = hashlib.sha256(data).hexdigest()
        rows = []
        for i, sid in enumerate(['doc-001', 'doc-002']):
            fname = sid + '.txt'
            (batch / fname).write_bytes(data)
            row = {**MINIMAL_ROW, 'source_id': sid, 'content_hash': digest, 'path': fname,
                   'source_url': f'https://example.com/{sid}'}
            rows.append(row)
        (batch / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        return src

    def test_exact_duplicates_are_grouped(self):
        with tempfile.TemporaryDirectory() as d:
            src = self._make_dup_src(Path(d))
            records, _ = bcc.load_records(src)
            _, _, certain = bcc.duplicate_analysis(records)
        # Exactly one of the two is a certain duplicate
        self.assertEqual(len(certain), 1)

    def test_exact_duplicate_is_rejected_in_full_build(self):
        with tempfile.TemporaryDirectory() as d:
            src = self._make_dup_src(Path(d))
            out = Path(d) / 'out'
            s = bcc.build(src, out)
            # Read inside the with-block so the temp dir still exists
            rows = [json.loads(l) for l in (out / 'rejected_manifest.jsonl').read_text().splitlines() if l.strip()]
        self.assertGreater(len(rows), 0, 'rejected_manifest must be non-empty when a duplicate is present')
        dup_reasons = [r for row in rows for r in row['status_reasons'] if 'duplicate' in r]
        self.assertTrue(dup_reasons, 'rejected document must have a duplicate reason')


# ---------------------------------------------------------------------------
# 6. Split stability: same group_id always maps to the same split
# ---------------------------------------------------------------------------

class TestSplitStability(unittest.TestCase):
    """Contract: split_of is a pure deterministic function of its input.
    Guard fires when: the split assignment changes between calls (non-determinism)."""

    def test_deterministic(self):
        group_ids = [f'group-{i:04d}' for i in range(200)]
        first = [bcc.split_of(g) for g in group_ids]
        second = [bcc.split_of(g) for g in group_ids]
        self.assertEqual(first, second)

    def test_all_three_splits_appear(self):
        results = {bcc.split_of(f'g{i}') for i in range(500)}
        self.assertEqual(results, {'train', 'validation', 'test'})

    def test_fractions_are_approximately_correct(self):
        from collections import Counter
        counts = Counter(bcc.split_of(f'g{i}') for i in range(2000))
        # train should be ~90% (allow 5pp tolerance)
        self.assertGreater(counts['train'] / 2000, 0.85)
        self.assertLess(counts['validation'] / 2000, 0.10)
        self.assertLess(counts['test'] / 2000, 0.10)


# ---------------------------------------------------------------------------
# 7. Refusal to overwrite a non-empty output directory
# ---------------------------------------------------------------------------

class TestOutputSafety(unittest.TestCase):
    """Contract: build() raises FileExistsError when the output directory is non-empty.
    Guard fires when: a previous run's outputs are silently overwritten."""

    def test_non_empty_out_raises(self):
        with tempfile.TemporaryDirectory() as d:
            src = make_single_doc_src(Path(d))
            out = Path(d) / 'out'
            out.mkdir()
            (out / 'existing.txt').write_text('something', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                bcc.build(src, out)

    def test_out_inside_src_raises(self):
        with tempfile.TemporaryDirectory() as d:
            src = make_single_doc_src(Path(d))
            out = src / 'nested_out'
            with self.assertRaises(ValueError):
                bcc.build(src, out)


# ---------------------------------------------------------------------------
# 8. Quality flag: many_repeated_lines ignores short lines
# ---------------------------------------------------------------------------

class TestRepeatedLineFlag(unittest.TestCase):
    """Contract: lines shorter than REPEATED_LINE_MIN_CHARS are not counted in the
    repeated-line fraction, so a document whose only 'repeated' lines are single-character
    fragments does NOT trip the many_repeated_lines flag.
    Guard fires when: short-fragment repetition raises a false positive."""

    def test_short_fragments_do_not_trigger_flag(self):
        # Create a document with 200 repeated single-character "lines" surrounded by
        # TWO DISTINCT long-content blocks so no long line is repeated.
        # The only repetition is the short comma fragments, which must be ignored.
        block_a = (
            'The first unique section of this test document covers experimental methodology '
            'and establishes the baseline measurements for the quality-flag evaluation. '
            'Each procedure follows a deterministic protocol to ensure reproducibility. '
        )
        block_b = (
            'The second unique section discusses statistical analysis and interpretation '
            'of the results obtained from each test case. Confidence intervals are computed '
            'using bootstrap resampling across all experimental conditions. '
        )
        fragments = '\n'.join([','] * 200)
        text = block_a + '\n' + fragments + '\n' + block_b
        q = bcc.quality_review(text, 'Test document')
        self.assertNotIn('many_repeated_lines', q['flags'],
                         'Short single-character fragments must not trigger many_repeated_lines')

    def test_genuinely_repeated_long_lines_do_trigger_flag(self):
        # One long line repeated 50 times in a short document
        repeated_line = 'This is a line that is definitely longer than twenty characters and is repeated many times.'
        unique_prefix = LOREM
        text = unique_prefix + '\n' + '\n'.join([repeated_line] * 50)
        q = bcc.quality_review(text, 'Test')
        self.assertIn('many_repeated_lines', q['flags'],
                      'Genuinely repeated long lines must still trigger the flag')

    def test_threshold_is_correct_constant(self):
        # The constant is documented; this protects against accidental changes
        self.assertEqual(bcc.REPEATED_LINE_MIN_CHARS, 20)


# ---------------------------------------------------------------------------
# 9. Domain coverage: manifest labels are the factual source
# ---------------------------------------------------------------------------

class TestDomainCoverage(unittest.TestCase):
    """Contract: domain_coverage reports manifest `domain` labels as factual,
    and target_domains_with_zero_manifest_labels lists only targets with no
    manifest-label coverage (not keyword-heuristic failures).
    Guard fires when: keyword-heuristic noise drives the primary coverage report."""

    def _make_records_with_domain(self, domain_label: str) -> list:
        """Minimal record list with a single document labeled with domain_label."""
        row = {**MINIMAL_ROW, 'domain': domain_label}
        return [{'doc_id': 'doc-001', 'row': row, 'text': LOREM,
                 'words': len(LOREM.split()), 'bytes': len(LOREM.encode())}]

    def test_manifest_domain_appears_in_manifest_domains(self):
        records = self._make_records_with_domain('computer_science_software_algorithms')
        cov = bcc.domain_coverage(records)
        self.assertIn('computer_science_software_algorithms', cov['manifest_domains'])

    def test_zero_manifest_label_domains_are_reported(self):
        # A corpus with only one manifest domain cannot cover all target domains
        records = self._make_records_with_domain('human_cognition_behavior')
        cov = bcc.domain_coverage(records)
        zero = cov['target_domains_with_zero_manifest_labels']
        # Must have at least some; the exact set depends on MANIFEST_TO_TARGET
        self.assertTrue(len(zero) > 0, 'Not all target domains can be covered by human_cognition_behavior alone')

    def test_manifest_label_that_maps_to_a_target_is_not_in_zero_list(self):
        # biological_intelligence_ai maps to biology_neuroscience_cognitive_science and AI
        records = self._make_records_with_domain('biological_intelligence_ai')
        cov = bcc.domain_coverage(records)
        zero = set(cov['target_domains_with_zero_manifest_labels'])
        self.assertNotIn('biology_neuroscience_cognitive_science', zero)
        self.assertNotIn('artificial_intelligence_machine_learning', zero)

    def test_unlabeled_domains_are_not_mapped_to_targets(self):
        records = self._make_records_with_domain('entertainment')
        cov = bcc.domain_coverage(records)
        # entertainment has no mapping, so all target domains are uncovered
        self.assertEqual(len(cov['target_domains_with_zero_manifest_labels']),
                         len(bcc.TARGET_DOMAINS))


# ---------------------------------------------------------------------------
# 10. Byte-identical reruns (determinism)  [integration — needs full src]
# ---------------------------------------------------------------------------

@unittest.skipIf(os.environ.get('SKIP_SLOW') == '1', 'SKIP_SLOW=1')
class TestDeterminism(unittest.TestCase):
    """Contract: two consecutive runs over the same source produce byte-identical outputs.
    Guard fires when: a timestamp, random seed, or hash-map ordering breaks determinism."""

    SRC = Path('/home/claude/corpus_work/src')

    @classmethod
    def setUpClass(cls):
        if not cls.SRC.exists():
            raise unittest.SkipTest('corpus src not available in this environment')

    def test_two_runs_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            out1 = Path(d) / 'run1'
            out2 = Path(d) / 'run2'
            bcc.build(self.SRC, out1, expected=499)
            bcc.build(self.SRC, out2, expected=499)
            # Compare every output file
            files = sorted(p.name for p in out1.iterdir())
            for fname in files:
                b1 = (out1 / fname).read_bytes()
                b2 = (out2 / fname).read_bytes()
                self.assertEqual(b1, b2, f'{fname} differs between runs')


if __name__ == '__main__':
    unittest.main()
