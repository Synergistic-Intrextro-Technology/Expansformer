"""Offline acquisition checks; injected API replies ensure no network calls."""
from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from dataset_pipeline import audit, import_documents, init
from wikipedia_acquisition import collect, enumerate_pages


def offline_api(params, user_agent, timeout, max_bytes):
    if params.get('list') == 'categorymembers':
        category = params['cmtitle']
        if category == 'Category:Mathematics':
            return {'query': {'categorymembers': [
                {'ns': 0, 'title': 'Prime number'}, {'ns': 14, 'title': 'Category:Number theory'}]}}
        if category == 'Category:Number theory':
            return {'query': {'categorymembers': [
                {'ns': 0, 'title': 'Integer'}, {'ns': 0, 'title': 'Prime number'}]}}
        raise AssertionError(f'unexpected category request: {category}')
    title = params['titles']
    pageid = 101 if title == 'Prime number' else 202
    return {'query': {'pages': [{
        'pageid': pageid, 'title': title, 'fullurl': f'https://en.wikipedia.org/wiki/{title.replace(" ", "_")}',
        'revisions': [{'revid': pageid + 1000, 'timestamp': '2026-01-01T00:00:00Z'}],
        'extract': (f'{title} demonstrates mathematical structures. ' * 20),
    }]}}


class WikipediaAcquisitionTests(unittest.TestCase):
    def test_category_breadth_and_import_to_v03_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            categories = base / 'categories.jsonl'
            categories.write_text(json.dumps({'domain': 'mathematics', 'category': 'Mathematics', 'max_depth': 1}) + '\n', encoding='utf-8')
            acquisition = base / 'wikipedia'
            result = collect(acquisition, categories, 'Expansformer-test/1 (test@example.org)', 10,
                             delay=0, min_words=5, requester=offline_api, sleeper=lambda _: None)
            self.assertEqual(result['documents'], 2)
            rows = [json.loads(line) for line in (acquisition / 'manifest.jsonl').read_text().splitlines()]
            self.assertEqual({r['title'] for r in rows}, {'Prime number', 'Integer'})
            self.assertTrue(all(r['revision_id'] and r['history_url'] for r in rows))
            self.assertTrue(all(r['license_status'] == 'SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED' for r in rows))
            self.assertTrue(all(r['license_url'] == 'https://creativecommons.org/licenses/by-sa/4.0/' for r in rows))
            self.assertTrue(all(r['license_terms_url'].endswith('/Terms_of_Use') for r in rows))
            self.assertTrue(all(r['additional_license_urls'] == ['https://www.gnu.org/licenses/fdl-1.3.html'] for r in rows))

            dataset = base / 'dataset'
            init(dataset)
            # The collector's declaration is insufficient for dataset admission.
            for row in rows:
                row['license_status'] = 'INDEPENDENTLY_REVIEWED'
                row['license_review_evidence'] = 'synthetic-fixture/article-license-review.txt'
            (acquisition / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            imported = import_documents(dataset, acquisition, acquisition / 'manifest.jsonl', 'training', None, None)
            self.assertEqual(imported, 2)
            report = audit(dataset)
            self.assertTrue(report['valid'], report['problems'])
            self.assertEqual(report['training_documents'], 2)
            license_rows = [json.loads(line) for line in (dataset / 'manifests' / 'licenses.jsonl').read_text().splitlines()]
            self.assertEqual(len(license_rows), 2)
            self.assertTrue(all(row['license_url'] == 'https://creativecommons.org/licenses/by-sa/4.0/' for row in license_rows))
            self.assertTrue(all(row['license_terms_url'].endswith('/Terms_of_Use') for row in license_rows))
            self.assertTrue(all(row['additional_license_urls'] == ['https://www.gnu.org/licenses/fdl-1.3.html'] for row in license_rows))
            license_rows[0]['license_terms_url'] = 'https://example.invalid/not-terms'
            (dataset / 'manifests' / 'licenses.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in license_rows), encoding='utf-8')
            self.assertTrue(any('license record mismatch' in problem for problem in audit(dataset)['problems']))

    def test_category_subcategory_traversal_is_bounded_and_deduplicated(self):
        rows = enumerate_pages({'domain': 'mathematics', 'category': 'Mathematics', 'max_depth': 1},
                               'Expansformer-test/1 (test@example.org)', 1, 0, 10,
                               requester=offline_api, sleeper=lambda _: None)
        self.assertEqual([row['title'] for row in rows], ['Prime number', 'Integer'])

    def test_acquisition_budgets_pages_across_domains(self):
        def multi_domain_api(params, user_agent, timeout, max_bytes):
            if params.get('list') == 'categorymembers':
                domain = params['cmtitle'].removeprefix('Category:')
                return {'query': {'categorymembers': [
                    {'ns': 0, 'title': f'{domain} topic {number}'} for number in range(8)]}}
            title = params['titles']
            pageid = int(hashlib.sha256(title.encode()).hexdigest()[:8], 16) + 1
            return {'query': {'pages': [{
                'pageid': pageid, 'title': title,
                'revisions': [{'revid': pageid + 1, 'timestamp': '2026-01-01T00:00:00Z'}],
                'extract': f'{title} provides a distinct scientific subject. ' * 10,
            }]}}

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            categories = base / 'categories.jsonl'
            categories.write_text(''.join(json.dumps(row) + '\n' for row in (
                {'domain': 'mathematics', 'category': 'Mathematics'},
                {'domain': 'computing', 'category': 'Computing'},
            )), encoding='utf-8')
            result = collect(base / 'wiki', categories, 'Expansformer-test/1 (test@example.org)', 4,
                             delay=0, min_words=5, requester=multi_domain_api, sleeper=lambda _: None)
            self.assertEqual(result['documents'], 4)
            self.assertEqual(result['domains'], {'computing': 2, 'mathematics': 2})

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            categories = base / 'categories.jsonl'
            categories.write_text(json.dumps({'domain': 'mathematics', 'category': 'Mathematics'}) + '\n', encoding='utf-8')
            out = base / 'existing'
            out.mkdir()
            sentinel = out / 'keep.txt'
            sentinel.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                collect(out, categories, 'Expansformer-test/1 (test@example.org)', 10,
                        delay=0, min_words=5, requester=offline_api, sleeper=lambda _: None)
            self.assertEqual(sentinel.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
