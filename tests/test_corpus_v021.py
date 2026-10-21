"""Offline regression tests for the v0.2.1 corpus improvements."""
from __future__ import annotations
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from corpus_builder import TOPICS, balanced_topics, collect_wiki, read_documents, text_quality, wiki_fetch


def fake_wikipedia(title, user_agent, timeout, max_bytes):
    return {
        'title': title,
        'fullurl': 'https://en.wikipedia.org/wiki/' + title.replace(' ', '_'),
        'revisions': [{'revid': 1234, 'timestamp': '2026-10-01T00:00:00Z'}],
        'extract': (f'Explanation and observations concerning {title}. ' * 24),
    }


class CorpusV021Tests(unittest.TestCase):
    def test_balanced_default_includes_all_domains(self):
        topics = balanced_topics(list(TOPICS), 32)
        self.assertEqual(len(topics), 32)
        self.assertEqual(set(d for d, _ in topics), set(TOPICS))
        self.assertEqual(len(set(t for _, t in topics)), 32)
        counts = {d: sum(item[0] == d for item in topics) for d in TOPICS}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_small_selection_reaches_distinct_domains(self):
        topics = balanced_topics(['mathematics', 'medicine', 'arts'], 3)
        self.assertEqual([d for d, _ in topics], ['mathematics', 'medicine', 'arts'])

    def test_collection_contains_domain_and_license_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / 'wiki'
            collect_wiki(dest, 'Expansformer/0.2.1 (dev@example.org)', list(TOPICS), 32,
                         delay=0, fetcher=fake_wikipedia)
            documents = read_documents(dest)
            self.assertEqual(len(documents), 32)
            self.assertEqual({d.domain for d in documents}, set(TOPICS))
            rows = [json.loads(line) for line in (dest / 'manifest.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 32)
            self.assertTrue(all(row['license_status'] == 'SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED'
                                for row in rows))
            self.assertTrue(all(row['sha256'] == d.sha256 for row, d in zip(rows, documents)))
            self.assertTrue(all('quality_diagnostics' in row for row in rows))

    def test_no_partial_publication_when_network_returns_too_few_documents(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / 'wiki'
            def bad_fetch(title, ua, timeout, max_bytes):
                return {'extract': ''}
            with self.assertRaisesRegex(RuntimeError, 'fewer than two'):
                collect_wiki(dest, 'Expansformer/0.2.1 (dev@example.org)', ['history'], 2,
                             delay=0, fetcher=bad_fetch)
            self.assertFalse(dest.exists())
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_content_diagnostics_repeated_lines(self):
        diagnostics = text_quality(('repeated text line\n' * 100))
        self.assertEqual(diagnostics['words'], 300)
        self.assertEqual(diagnostics['repeated_line_fraction'], 1.0)

    def test_rate_limit_retry_after(self):
        payload = {'query': {'pages': [{'title': 'Probability theory', 'extract': 'Test content'}]}}
        class Reply:
            def __init__(self): self.data = io.BytesIO(json.dumps(payload).encode())
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, n): return self.data.read(n)
        err = urllib.error.HTTPError('https://en.wikipedia.org/w/api.php', 429,
                                     'Too Many Requests', {'Retry-After': '6'}, None)
        with patch('corpus_builder.urllib.request.urlopen', side_effect=[err, Reply()]) as getter, \
             patch('corpus_builder.time.sleep') as sleep:
            result = wiki_fetch('Probability theory', 'Expansformer/0.2.1 (dev@example.org)', 1, 3000)
        self.assertEqual(result['title'], 'Probability theory')
        self.assertEqual(getter.call_count, 2)
        sleep.assert_called_once_with(6)

    def test_reject_bad_api_response_without_repeated_requests(self):
        response = {'error': {'code': 'badtitle'}}
        class Reply:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, n): return json.dumps(response).encode()
        with patch('corpus_builder.urllib.request.urlopen', return_value=Reply()) as getter:
            with self.assertRaisesRegex(ValueError, 'badtitle'):
                wiki_fetch('bad', 'Expansformer/0.2.1 (dev@example.org)', 1, 3000)
        self.assertEqual(getter.call_count, 1)

if __name__ == '__main__':
    unittest.main()
