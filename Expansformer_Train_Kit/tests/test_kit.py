"""Offline regressions for the integration/training bridge."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('expansformer_train_kit', Path(__file__).resolve().parents[1] / 'run.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class KitTests(unittest.TestCase):
    def test_plain_by_text(self):
        raw = b'<article><license>Creative Commons Attribution License (CC BY).</license></article>'
        self.assertEqual(MOD.license_from_jats(raw), (True, 'SOURCE_DECLARED_CC_BY_VERSION_UNSPECIFIED'))

    def test_by_4_text(self):
        raw = b'<article><license>This work is licensed under a Creative Commons Attribution 4.0 International License.</license></article>'
        self.assertEqual(MOD.license_from_jats(raw), (True, 'SOURCE_DECLARED_CC_BY_4.0'))

    def test_by_nc_url(self):
        raw = b'<article xmlns:xlink="http://www.w3.org/1999/xlink"><license><ext-link xlink:href="https://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution License</ext-link></license></article>'
        self.assertFalse(MOD.license_from_jats(raw)[0])

    def test_by_nc_nd_text(self):
        raw = b'<article><license>CC BY-NC-ND 3.0</license></article>'
        self.assertFalse(MOD.license_from_jats(raw)[0])

    def test_unresolved_is_not_passed(self):
        self.assertFalse(MOD.license_from_jats(b'<article><license>All rights reserved.</license></article>')[0])

    def test_subfolder_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                MOD.safe_file(Path(temp), '../secret')

    def test_existing_corpus_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            m = p / 'topic_gap_batch_240_revised'
            (m / 'space').mkdir(parents=True)
            data = b'Text for self supervised research. ' * 20
            (m / 'space' / 'article.txt').write_bytes(data)
            row = {'source_id':'wiki-01','domain':'space','source_type':'wikipedia_article',
                   'title':'Space','path':'space/article.txt','content_hash':hashlib.sha256(data).hexdigest(),
                   'license_url':'https://creativecommons.org/licenses/by-sa/4.0/',
                   'source_url':'https://en.wikipedia.org/wiki/Space','license_status':'UNREVIEWED'}
            (m / 'manifest.jsonl').write_text(json.dumps(row) + '\n')
            out = p / 'runs' / 'research_corpus'
            self.assertEqual(MOD.prepare(p, out), 1)
            self.assertEqual(MOD.prepare(p, out), 1)
            self.assertFalse((p / 'expansformer_dataset').exists())
            self.assertEqual(len((out / 'manifest.jsonl').read_text().splitlines()), 1)

    def test_install_refuses_mismatched_files(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p / MOD.NEURAL_FILES[1]).write_text('not v04')
            with self.assertRaises(ValueError):
                MOD.install(p)
            self.assertFalse((p / MOD.NEURAL_FILES[0]).exists())


if __name__ == '__main__':
    unittest.main()
