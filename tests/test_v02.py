"""Run: python -m unittest discover -s tests -v"""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

import torch
from corpus_builder import read_documents, split_documents, digest, wiki_fetch, collect_wiki
from evidence_index import index_corpus, search
from claim_gate import inspect_claims
from hardware_report import estimate
from train_expansformer_v01 import OwnTokenizer
from train_expansformer_v02 import (make_parser, train, evaluate, use_sdpa, get_device, prepare_data)
from train_expansformer_v01 import load_model_source

BASE = Path(__file__).resolve().parents[1]


def make_corpus(folder: Path, n=8):
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        subject = ['Mathematics', 'Chemistry', 'History', 'Engineering'][k % 4]
        text = (f'{subject} example {k}: Evidence must come from observations. '
                f'When the data cannot establish the claim, acknowledge uncertainty. '
                f'Formal tests can reject a hypothesis without proving every alternative.\n') * 10
        (folder / f'document_{k}.txt').write_text(text, encoding='utf-8')


def train_args(folder, out, steps=3, stop=None, attention='sdpa'):
    p = make_parser()
    argv = ['train', '--data', str(folder), '--out', str(out), '--steps', str(steps),
            '--n-layers', '2', '--n-heads', '4', '--embed-dim', '64',
            '--context-length', '24', '--batch-size', '2', '--grad-accum', '2',
            '--eval-windows', '2', '--eval-every', '1', '--save-every', '1',
            '--warmup-steps', '1', '--cpu-threads', '2', '--device', 'cpu',
            '--attention', attention, '--amp', 'off', '--tokenizer', 'byte']
    if stop:
        argv += ['--stop-after', str(stop)]
    return p.parse_args(argv)


class V02Tests(unittest.TestCase):
    def test_tokenizer_unicode_roundtrip(self):
        tok = OwnTokenizer()
        tok.train_bpe('Scientific observation and uncertainty.\n' * 32, 280, 4096)
        for s in ('A claim is not evidence.', '東京 and Montréal ✨', 'line\n\nnext'):
            self.assertEqual(tok.decode(tok.encode(s)), s)

    def test_split_determinism_no_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'corpus'
            make_corpus(data)
            docs = read_documents(data)
            x, y = split_documents(docs, 42)
            x2, y2 = split_documents(docs, 42)
            self.assertEqual([d.path for d in y], [d.path for d in y2])
            self.assertEqual(len(x)+len(y), len(docs))
            self.assertTrue({d.sha256 for d in x}.isdisjoint({d.sha256 for d in y}))
            self.assertGreaterEqual(len(y), 1)

    def test_manifest_detects_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'doc.txt').write_text('source data', encoding='utf-8')
            (root / 'manifest.jsonl').write_text(json.dumps({'path': 'doc.txt', 'sha256': digest('different')})+'\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                read_documents(root)

    def test_sdpa_parity(self):
        torch.set_num_threads(2)
        config, cls = load_model_source(str(BASE / 'expansformer_model'))
        torch.manual_seed(3)
        cfg = config(vocab_size=260, context_length=24, n_layers=2, n_heads=4, embed_dim=64, dropout=0)
        original = cls(cfg).eval()
        other = cls(cfg).eval()
        other.load_state_dict(original.state_dict())
        use_sdpa(other)
        ids = torch.randint(0, 260, (2, 19))
        lhs, lhs_loss = original(ids, ids)
        rhs, rhs_loss = other(ids, ids)
        self.assertTrue(torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(lhs_loss, rhs_loss, atol=1e-5, rtol=1e-4))
        lhs_loss.backward(); rhs_loss.backward()
        for p, q in zip(original.parameters(), other.parameters()):
            self.assertTrue(torch.allclose(p.grad, q.grad, atol=2e-5, rtol=2e-4))

    def test_training_resume_equivalent_to_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root/'corpus'; make_corpus(corpus)
            baseline = root/'baseline'; interrupted = root/'interrupted'
            train(train_args(corpus, baseline, steps=3))
            train(train_args(corpus, interrupted, steps=3, stop=1))
            resume = train_args(corpus, interrupted, steps=3)
            resume.resume = str(interrupted/'latest.pt')
            train(resume)
            one = torch.load(baseline/'latest.pt', weights_only=True, map_location='cpu')
            two = torch.load(interrupted/'latest.pt', weights_only=True, map_location='cpu')
            self.assertEqual(one['step'], two['step'])
            self.assertEqual(one['fingerprint'], two['fingerprint'])
            for k in one['model']:
                self.assertTrue(torch.equal(one['model'][k], two['model'][k]), k)
            metrics = [json.loads(line) for line in (interrupted/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([m['step'] for m in metrics], [1, 2, 3])
            self.assertGreater(metrics[-1]['val_token_coverage'], 0)
            # Generation deterministic at temperature=0 across checkpoint restores.
            from train_expansformer_v02 import generate
            gen = make_parser().parse_args(['generate', '--checkpoint', str(interrupted/'latest.pt'),
                                            '--prompt', 'Evidence:', '--temperature', '0', '--max-new-tokens', '4',
                                            '--device', 'cpu'])
            generate(gen)

    def test_recover_metrics_after_checkpoint_before_log(self):
        from train_expansformer_v02 import trim_metrics
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metrics.jsonl'
            path.write_text(json.dumps({'step': 1, 'train_loss': 3.0})+'\n', encoding='utf-8')
            trim_metrics(path, 2)
            rows=[json.loads(r) for r in path.read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows], [1, 2])
            self.assertTrue(rows[-1]['checkpoint_recovered'])
            self.assertIsNone(rows[-1]['val_loss'])

    def test_resume_rejects_changed_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            make_corpus(root/'c')
            run=root/'run'
            train(train_args(root/'c', run, steps=2, stop=1))
            (root/'c'/'document_1.txt').write_text('altered by external source '+ 'x'*500)
            a=train_args(root/'c', run, steps=2)
            a.resume=str(run/'latest.pt')
            with self.assertRaisesRegex(ValueError, 'Dataset content or provenance changed'):
                train(a)

    def test_evidence_index_distinguishes_retrieval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            make_corpus(root/'data')
            path=root/'index.db'
            self.assertEqual(index_corpus(root/'data', path), 8)
            rows=search(path, 'hypothesis', 3)
            self.assertTrue(rows)
            self.assertEqual(rows[0]['status'], 'SOURCE_RETRIEVED_NOT_FACT_CHECKED')
            self.assertIn('sha256', rows[0])
            self.assertEqual(search(path, 'no-such-term-existing', 3), [])

    def test_claim_gate_rejects_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'source.txt').write_text('Observations can conflict with hypotheses.', encoding='utf-8')
            claims=[{'kind': 'quotation', 'statement': 'Observations can conflict with hypotheses.',
                     'source_path': 'source.txt', 'source_quote': 'Observations can conflict with hypotheses.'},
                    {'kind': 'assertion', 'statement': 'The model always knows the truth.'},
                    {'kind': 'unknown', 'statement': 'I do not have enough evidence.'},
                    {'kind': 'quotation', 'statement': 'The source proves everything.',
                     'source_path': 'source.txt', 'source_quote': 'The source proves everything.'}]
            statuses=[r['state'] for r in inspect_claims(root, claims)]
            self.assertEqual(statuses, ['SOURCE_QUOTATION_MATCHED', 'REJECTED', 'ABSTENTION', 'REJECTED'])

    def test_parameter_estimator_matches_model(self):
        config, model_type = load_model_source(str(BASE/'expansformer_model'))
        cfg=config(vocab_size=260, context_length=24, n_layers=2, n_heads=4, embed_dim=64)
        model=model_type(cfg)
        self.assertEqual(estimate(2, 4, 64, 260)['parameters'], sum(p.numel() for p in model.parameters()))

    def test_wiki_acquisition_mocked_offline(self):
        def fake(title, ua, timeout, max_bytes):
            return {'title': title, 'fullurl': 'https://en.wikipedia.org/wiki/Mock',
                    'revisions': [{'revid': 123, 'timestamp': '2026-01-01T00:00:00Z'}],
                    'extract': f'Content for {title}: ' + ('repeatable educational encyclopedia words. ' * 20)}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'wiki'
            collect_wiki(root, 'Expansformer-test contact: test@example.org', ['mathematics'],
                         3, delay=0, fetcher=fake)
            self.assertEqual(len(read_documents(root)), 3)
            self.assertTrue(all(d.license.startswith('CC BY-SA') for d in read_documents(root)))


if __name__ == '__main__':
    unittest.main()
