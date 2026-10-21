"""Filesystem-level dataset pipeline checks; generated content is synthetic."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from corpus_builder import read_documents
from dataset_pipeline import (EVALUATION_TASK_TO_TRACK, audit, import_documents,
                              import_quality_reviews, import_records, init, split_training)


class DatasetPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'expansformer_dataset'
        self.input = self.base / 'input'
        self.input.mkdir()
        init(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def write_doc(self, rel: str, content: str):
        path = self.input / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def metadata(self, source_id: str, rel: str, digest: str, **kwargs):
        row = {
            'source_id': source_id, 'path': rel, 'title': rel, 'publisher': 'Synthetic fixture',
            'source_url': 'https://example.invalid/' + source_id, 'publication_date': '2026-01-01',
            'license_id': 'CC0-1.0 (synthetic fixture only)', 'license_url': 'https://example.invalid/license',
            'license_status': 'INDEPENDENTLY_REVIEWED', 'license_review_evidence': 'synthetic-fixture/rights-review.txt',
            'domain': 'mathematics', 'source_type': 'synthetic_fixture', 'content_hash': 'sha256:' + digest,
            **kwargs,
        }
        return row

    def test_2000_documents_import_audit_and_grouped_holdout(self):
        records = []
        for index in range(2000):
            rel = f'docs/doc-{index:04d}.txt'
            content = (f'Synthetic document {index}. It contains unique fixture material.\n'
                       f'Group {(index // 2):04d} domain {index % 5}.\n')
            digest = self.write_doc(rel, content)
            domains = ('mathematics', 'engineering', 'computing', 'biology', 'history')
            records.append(self.metadata(f'synthetic-{index:04d}', rel, digest,
                                         domain=domains[index % len(domains)],
                                         source_group=f'work-{index // 2:04d}'))
        metadata_file = self.base / 'sources.jsonl'
        metadata_file.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')

        self.assertEqual(import_documents(self.root, self.input, metadata_file, 'training', None, None), 2000)
        initial = audit(self.root)
        self.assertTrue(initial['valid'], initial['problems'][:5])
        self.assertEqual(initial['training_documents'], 2000)
        self.assertTrue(initial['objective_readiness']['coverage_gates']['minimum_training_documents_met'])
        self.assertFalse(initial['objective_readiness']['coverage_gates']['minimum_training_domains_met'])
        self.assertEqual(len(initial['domains']), 5)
        self.assertEqual(initial['training_words'], 24000)

        result = split_training(self.root, seed=73, fraction=.2)
        self.assertEqual(result['train_documents'] + result['validation_documents'], 2000)
        self.assertEqual(result['validation_documents'], 400)
        train = read_documents(self.root / 'curated' / 'pretraining' / 'train')
        validation = read_documents(self.root / 'curated' / 'pretraining' / 'validation')
        self.assertEqual((len(train), len(validation)), (1600, 400))
        train_manifest = self.root / 'curated' / 'pretraining' / 'train' / 'manifest.jsonl'
        val_manifest = self.root / 'curated' / 'pretraining' / 'validation' / 'manifest.jsonl'
        train_groups = {json.loads(line)['source_group'] for line in train_manifest.read_text().splitlines()}
        val_groups = {json.loads(line)['source_group'] for line in val_manifest.read_text().splitlines()}
        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertTrue({d.sha256 for d in train}.isdisjoint({d.sha256 for d in validation}))
        self.assertTrue(audit(self.root)['valid'])

    def test_empty_scaffold_is_structurally_valid_but_objective_incomplete(self):
        report = audit(self.root)
        self.assertTrue(report['valid'], report['problems'])
        self.assertEqual(report['objective_readiness']['status'], 'coverage_gates_incomplete')
        self.assertFalse(report['objective_readiness']['coverage_gates']['minimum_training_documents_met'])
        self.assertFalse(report['objective_readiness']['coverage_gates']['minimum_training_domains_met'])
        self.assertIn('independent source-quality verification is not authenticated by this tool',
                      report['objective_readiness']['gaps'])

    def test_audit_coverage_gate_fails_closed_on_empty_dataset(self):
        script = Path(__file__).resolve().parents[1] / 'dataset_pipeline.py'
        result = subprocess.run(
            [sys.executable, str(script), 'audit', '--root', str(self.root), '--require-coverage-gates'],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report['valid'])
        self.assertEqual(report['objective_readiness']['status'], 'coverage_gates_incomplete')

    def test_init_upgrades_existing_config_with_evaluation_track_schema(self):
        config_path = self.root / 'dataset_config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        config['schema_version'] = 2
        config.pop('evaluation_task_tracks', None)
        config.pop('evaluation_scoring_required_fields', None)
        config.pop('license_manifest_extra_fields', None)
        config_path.write_text(json.dumps(config), encoding='utf-8')

        init(self.root)

        upgraded = json.loads(config_path.read_text(encoding='utf-8'))
        self.assertEqual(upgraded['schema_version'], 4)
        self.assertEqual(upgraded['evaluation_task_tracks'], EVALUATION_TASK_TO_TRACK)
        self.assertEqual(upgraded['evaluation_scoring_required_fields'], ['metric', 'protocol'])
        self.assertEqual(upgraded['license_manifest_extra_fields'],
                         ['license_terms_url', 'license_scope', 'additional_license_urls'])

    def test_group_split_tracks_requested_fraction_per_domain(self):
        records = []
        specifications = (('mathematics', 5), ('history', 15), ('engineering', 30))
        for domain, amount in specifications:
            for index in range(amount):
                rel = f'{domain}/{domain}-{index:03d}.txt'
                digest = self.write_doc(rel, f'Unique source material in {domain}, sample {index}.')
                records.append(self.metadata(f'{domain}-{index}', rel, digest, domain=domain,
                                             source_group=f'{domain}-work-{index}'))
        metadata = self.base / 'domain-sources.jsonl'
        metadata.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
        import_documents(self.root, self.input, metadata, 'training', None, None)
        result = split_training(self.root, seed=17, fraction=.2)
        self.assertEqual(result['validation_documents'], 10)
        self.assertEqual(result['validation_domain_counts'],
                         {'engineering': 6, 'history': 3, 'mathematics': 1})

    def test_normalized_duplicate_is_rejected_across_training_and_evaluation(self):
        rel = 'one.txt'
        digest = self.write_doc(rel, 'Repeated content\nwith spaces.\n')
        metadata = self.base / 'training.jsonl'
        metadata.write_text(json.dumps(self.metadata('train-one', rel, digest)) + '\n', encoding='utf-8')
        import_documents(self.root, self.input, metadata, 'training', None, None)

        eval_input = self.base / 'evaluation-input'
        eval_input.mkdir()
        eval_content = '  Repeated   content with spaces. '
        (eval_input / rel).write_text(eval_content, encoding='utf-8')
        eval_hash = hashlib.sha256((eval_input / rel).read_bytes()).hexdigest()
        eval_metadata = self.base / 'evaluation.jsonl'
        eval_metadata.write_text(json.dumps(self.metadata('eval-one', rel, eval_hash)) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'overlaps existing training'):
            import_documents(self.root, eval_input, eval_metadata, 'evaluation', None, 'factuality')
        self.assertEqual(audit(self.root)['evaluation_documents'], 0)

    def test_source_group_overlap_is_rejected_even_when_document_text_differs(self):
        train_rel = 'chapter-1.txt'
        train_hash = self.write_doc(train_rel, 'First chapter has unique material and claims.')
        train_metadata = self.base / 'training.jsonl'
        train_metadata.write_text(json.dumps(self.metadata('book-chapter-1', train_rel, train_hash,
                                                           source_group='same-book')) + '\n', encoding='utf-8')
        import_documents(self.root, self.input, train_metadata, 'training', None, None)

        eval_dir = self.base / 'evaluation-input'
        eval_dir.mkdir()
        eval_rel = 'chapter-2.txt'
        eval_content = 'Second chapter is distinct text but comes from the same book.'
        (eval_dir / eval_rel).write_text(eval_content, encoding='utf-8')
        eval_hash = hashlib.sha256((eval_dir / eval_rel).read_bytes()).hexdigest()
        eval_metadata = self.base / 'evaluation.jsonl'
        eval_metadata.write_text(json.dumps(self.metadata('book-chapter-2', eval_rel, eval_hash,
                                                          domain='mathematics', source_group='same-book')) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'source_group overlaps existing training'):
            import_documents(self.root, eval_dir, eval_metadata, 'evaluation', None, 'factuality')
        self.assertEqual(audit(self.root)['evaluation_documents'], 0)

    def test_unknown_rights_and_unsubstantiated_review_are_rejected_for_training(self):
        rel = 'restricted.txt'
        digest = self.write_doc(rel, 'Unique source material.')
        metadata = self.base / 'unknown.jsonl'
        metadata.write_text(json.dumps(self.metadata('unknown-rights', rel, digest,
                                                     license_status='UNKNOWN')) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'requires source-specific INDEPENDENTLY_REVIEWED'):
            import_documents(self.root, self.input, metadata, 'training', None, None)

        metadata.write_text(json.dumps(self.metadata('declared-only', rel, digest,
                                                     license_status='SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED')) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'requires source-specific INDEPENDENTLY_REVIEWED'):
            import_documents(self.root, self.input, metadata, 'training', None, None)

        metadata.write_text(json.dumps(self.metadata('review-claim', rel, digest,
                                                     license_status='INDEPENDENTLY_REVIEWED',
                                                     license_review_evidence='')) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'license_review_evidence'):
            import_documents(self.root, self.input, metadata, 'training', None, None)
        self.assertEqual(audit(self.root)['documents'], 0)

    def test_structured_records_are_materialized_and_stay_in_their_partitions(self):
        train_rel = 'train.txt'
        train_hash = self.write_doc(train_rel, 'Training source contains causal relationships and outcomes.')
        training_metadata = self.base / 'training.jsonl'
        training_metadata.write_text(json.dumps(self.metadata('train-source', train_rel, train_hash)) + '\n', encoding='utf-8')
        import_documents(self.root, self.input, training_metadata, 'training', None, None)

        eval_dir = self.base / 'eval-input'
        eval_dir.mkdir()
        eval_rel = 'eval.txt'
        eval_content = 'Held-out source has unrelated factual content.'
        (eval_dir / eval_rel).write_text(eval_content, encoding='utf-8')
        eval_hash = hashlib.sha256((eval_dir / eval_rel).read_bytes()).hexdigest()
        eval_metadata = self.base / 'evaluation.jsonl'
        eval_metadata.write_text(json.dumps(self.metadata('eval-source', eval_rel, eval_hash,
                                                         domain='history')) + '\n', encoding='utf-8')
        import_documents(self.root, eval_dir, eval_metadata, 'evaluation', None, 'factuality')

        reasoning = self.base / 'reasoning.jsonl'
        reasoning.write_text(json.dumps({'id': 'reason-1', 'task_type': 'causal', 'prompt': 'Explain the mechanism.',
                                         'constraints': [], 'answer': 'A causes B through mechanism M.',
                                         'rationale': 'The source establishes M.', 'source_ids': ['train-source']}) + '\n', encoding='utf-8')
        epistemic = self.base / 'epistemic.jsonl'
        epistemic.write_text(json.dumps({'id': 'epistemic-1', 'prompt': 'What is supported?',
                                         'supported_claims': ['Claim A'], 'disputed_claims': [], 'assumptions': [],
                                         'unknowns': ['Claim B'], 'conclusion': 'Only A is supported.',
                                         'source_ids': ['train-source']}) + '\n', encoding='utf-8')
        evaluation = self.base / 'evaluation-records.jsonl'
        evaluation.write_text(json.dumps({'id': 'fact-1', 'task_type': 'claim_support',
                                          'input': 'Is claim X supported?', 'expected': 'unknown',
                                          'scoring': {'metric': 'exact_match', 'protocol': 'Compare normalized labels.'},
                                          'source_ids': ['eval-source']}) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unsupported evaluation task_type'):
            import_records(self.root, evaluation, 'evaluation', 'factuality')
        evaluation.write_text(json.dumps({'id': 'fact-1', 'task_type': 'factuality',
                                          'input': 'Is claim X supported?', 'expected': 'unknown',
                                          'scoring': {'metric': 'exact_match'},
                                          'source_ids': ['eval-source']}) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'requires nonempty metric and protocol'):
            import_records(self.root, evaluation, 'evaluation', 'factuality')
        import_records(self.root, reasoning, 'reasoning', None)
        import_records(self.root, epistemic, 'epistemic', None)
        for index, (task_type, track) in enumerate(EVALUATION_TASK_TO_TRACK.items()):
            evaluation.write_text(json.dumps({'id': f'eval-{index}', 'task_type': task_type,
                                              'input': 'Evaluate this held-out example.', 'expected': 'expected result',
                                              'scoring': {'metric': 'exact_match', 'protocol': 'Compare normalized labels.'},
                                              'source_ids': ['eval-source']}) + '\n', encoding='utf-8')
            import_records(self.root, evaluation, 'evaluation', track)
        self.assertEqual(len((self.root / 'curated' / 'reasoning' / 'records.jsonl').read_text().splitlines()), 1)
        self.assertEqual(len((self.root / 'curated' / 'epistemic_training' / 'records.jsonl').read_text().splitlines()), 1)
        for track in EVALUATION_TASK_TO_TRACK.values():
            self.assertEqual(len((self.root / 'evaluation' / track / 'records.jsonl').read_text().splitlines()), 1)
        report = audit(self.root)
        self.assertTrue(report['valid'], report['problems'])
        self.assertTrue(report['objective_readiness']['coverage_gates']['all_required_evaluation_task_types_present'])

        generalization_file = self.root / 'evaluation' / 'generalization' / 'records.jsonl'
        misplaced_record = json.loads(generalization_file.read_text(encoding='utf-8'))
        generalization_file.write_text('', encoding='utf-8')
        reasoning_file = self.root / 'evaluation' / 'reasoning' / 'records.jsonl'
        with reasoning_file.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(misplaced_record) + '\n')
        misplaced_report = audit(self.root)
        self.assertFalse(misplaced_report['valid'])
        self.assertTrue(any('evaluation/generalization records do not match' in problem
                            for problem in misplaced_report['problems']))

    def test_two_reviewer_quality_acceptance_is_recorded_but_not_authenticated(self):
        rel = 'source.txt'
        digest = self.write_doc(rel, 'An independently reviewable source document with useful content.')
        rel2 = 'source-2.txt'
        digest2 = self.write_doc(rel2, 'A second independently reviewable source document with useful content.')
        metadata = self.base / 'sources.jsonl'
        metadata.write_text(''.join(json.dumps(row) + '\n' for row in (
            self.metadata('reviewed-source', rel, digest),
            self.metadata('reviewed-source-2', rel2, digest2))), encoding='utf-8')
        import_documents(self.root, self.input, metadata, 'training', None, None)
        with self.assertRaisesRegex(ValueError, 'acceptance is not recorded'):
            split_training(self.root, seed=42, fraction=.4, require_quality_review=True)
        ratings = {'source_authority': 2, 'editorial_process': 2, 'domain_relevance': 2,
                   'factual_support': 2, 'transparency': 2}
        reviews = []
        for source in ('reviewed-source', 'reviewed-source-2'):
            for number, affiliation in ((1, 'Independent University'), (2, 'Independent Institute')):
                reviews.append({'review_id': f'{source}-review-{number}', 'source_id': source,
                                'reviewer_id': f'reviewer-{number}', 'reviewer_affiliation': affiliation,
                                'relationship_to_source': 'independent', 'reviewed_at': '2026-10-18T00:00:00Z',
                                'method': 'Independent sample audit', 'evidence_refs': [f'review-notes/{source}-{number}.md'],
                                'ratings': ratings, 'decision': 'accept'})
        review_file = self.base / 'reviews.jsonl'
        review_file.write_text(''.join(json.dumps(row) + '\n' for row in reviews), encoding='utf-8')
        self.assertEqual(import_quality_reviews(self.root, review_file), 4)
        report = audit(self.root)
        self.assertTrue(report['valid'], report['problems'])
        self.assertTrue(report['two_reviewer_source_quality_acceptance_recorded'])
        self.assertFalse(report['independent_source_quality_verified'])
        split_training(self.root, seed=42, fraction=.4, require_quality_review=True)


if __name__ == '__main__':
    unittest.main()
