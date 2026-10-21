"""Offline regression suite for opt-in v0.3 neural extensions."""
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import torch

from train_expansformer_v01 import load_model_source
from train_expansformer_v03 import make_parser, train, generate, use_sdpa
from expansformer_v03_model import ResearchConfig, ResearchTransformerLM, SparseExperts
from compare_runs import compare
from evaluate_exits_v03 import measure

BASE = Path(__file__).resolve().parents[1]
load_model_source(str(BASE/'expansformer_model'))
from model import ModelConfig


def toy_corpus(folder):
    folder.mkdir(parents=True)
    for k in range(8):
        (folder / f'doc_{k}.txt').write_text(
            (f'Document {k}: Mathematical evidence can change a strategy. '
             f'No result establishes universal truth without examination. ' * 9), encoding='utf-8')


def arguments(data, out, mode='baseline', stop=None, steps=3):
    argv = ['train', '--data', str(data), '--out', str(out), '--steps', str(steps),
            '--n-layers','2','--n-heads','4','--embed-dim','64',
            '--context-length','24','--batch-size','2','--grad-accum','2',
            '--eval-windows','2','--eval-every','1','--save-every','1',
            '--warmup-steps','1','--cpu-threads','2','--device','cpu',
            '--attention','sdpa','--amp','off','--tokenizer','byte']
    if mode == 'moe':
        argv += ['--moe-experts','4','--moe-top-k','2']
    if mode == 'exit':
        argv += ['--auxiliary-exits']
    if mode == 'both':
        argv += ['--moe-experts','4','--moe-top-k','2','--auxiliary-exits']
    if stop:
        argv += ['--stop-after',str(stop)]
    return make_parser().parse_args(argv)


class V03Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_model_is_identical_when_disabled(self):
        _, original_cls = load_model_source(str(BASE/'expansformer_model'))
        cfg = ModelConfig(vocab_size=260,context_length=24,n_layers=2,n_heads=4,embed_dim=64,dropout=0)
        torch.manual_seed(17)
        baseline = original_cls(cfg)
        torch.manual_seed(17)
        enhanced = ResearchTransformerLM(cfg, ResearchConfig())
        self.assertEqual(set(baseline.state_dict()), set(enhanced.state_dict()))
        for key in baseline.state_dict():
            self.assertTrue(torch.equal(baseline.state_dict()[key], enhanced.state_dict()[key]),key)
        x = torch.randint(0,260,(2,16))
        a, a_loss = baseline(x,x)
        b, b_loss = enhanced(x,x)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(a_loss,b_loss))

    def test_moe_gradient_and_auxiliary_exit_gradient(self):
        cfg = ModelConfig(vocab_size=260,context_length=24,n_layers=3,n_heads=4,embed_dim=64,dropout=0)
        net = ResearchTransformerLM(cfg, ResearchConfig(moe_experts=4,moe_top_k=2,auxiliary_exits=True))
        x = torch.randint(0,260,(2,12))
        total, parts = net.training_loss(x,x)
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(float(parts['balance_loss']),-1e-8)
        total.backward()
        self.assertIsNotNone(net.blocks[0].ffn.router.weight.grad)
        self.assertGreater(net.blocks[0].ffn.router.weight.grad.abs().sum().item(),0)
        self.assertIsNotNone(net.exit_norms[0].weight.grad)
        self.assertGreater(net.exit_norms[0].weight.grad.abs().sum().item(),0)
        self.assertEqual(net.blocks[0].ffn.last_routing_counts.sum().item(),2*12*2)
        for depth in (1,2,3):
            self.assertEqual(net.logits_at_depth(x,depth).shape,(2,12,260))

    def test_untrained_intermediate_exit_forbidden(self):
        cfg = ModelConfig(vocab_size=260,context_length=24,n_layers=2,n_heads=4,embed_dim=64)
        net = ResearchTransformerLM(cfg, ResearchConfig())
        with self.assertRaisesRegex(ValueError,'not trained'):
            net.logits_at_depth(torch.randint(0,260,(1,8)),1)

    def test_causal_mask_prevents_future_leakage(self):
        cfg = ModelConfig(vocab_size=260,context_length=24,n_layers=2,n_heads=4,embed_dim=64,dropout=0)
        net = ResearchTransformerLM(cfg,ResearchConfig(moe_experts=4,auxiliary_exits=True)).eval()
        use_sdpa(net)
        x = torch.randint(0,260,(1,16))
        y = x.clone(); y[:,8:] = (y[:,8:]+7)%260
        with torch.no_grad():
            a,_ = net(x)
            b,_ = net(y)
        self.assertTrue(torch.allclose(a[:,:8],b[:,:8],atol=1e-5,rtol=1e-5))

    def test_research_config_validation(self):
        with self.assertRaises(ValueError):
            ResearchConfig(moe_experts=2,moe_top_k=3).validate(2)
        with self.assertRaises(ValueError):
            ResearchConfig(auxiliary_exits=True).validate(1)

    def test_train_resume_same_weights_and_generate_at_depth(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            toy_corpus(root/'corpus')
            complete=root/'complete'; interrupted=root/'interrupted'
            with redirect_stdout(StringIO()):
                train(arguments(root/'corpus',complete,mode='both'))
                train(arguments(root/'corpus',interrupted,mode='both',stop=1))
                resume=arguments(root/'corpus',interrupted,mode='both')
                resume.resume=str(interrupted/'latest.pt')
                train(resume)
            a=torch.load(complete/'latest.pt',weights_only=True)
            b=torch.load(interrupted/'latest.pt',weights_only=True)
            self.assertEqual(a['step'],b['step'])
            for name in a['model']:
                self.assertTrue(torch.equal(a['model'][name],b['model'][name]),name)
            rows=[json.loads(v) for v in (complete/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows],[1,2,3])
            self.assertTrue(all('train_lm_loss' in row and 'train_router_balance_loss' in row for row in rows))
            self.assertTrue(all(r['val_loss']>0 for r in rows))
            gen=make_parser().parse_args(['generate','--checkpoint',str(complete/'latest.pt'),
                                          '--prompt','Evidence:','--inference-depth','1',
                                          '--temperature','0','--max-new-tokens','4','--device','cpu'])
            with redirect_stdout(StringIO()) as output:
                generate(gen)
            self.assertTrue(output.getvalue().startswith('Evidence:'))
            depth_results = measure(complete/'latest.pt', root/'corpus',
                                    eval_windows=2, max_loss_increase=0.0, device_name='cpu')
            self.assertEqual(len(depth_results['depths']), 2)
            self.assertEqual(depth_results['selected_depth'], 2)
            baseline = root/'baseline'
            with redirect_stdout(StringIO()):
                train(arguments(root/'corpus',baseline,mode='baseline'))
            result=compare(baseline,complete)
            self.assertEqual(result['baseline']['fingerprint'],result['candidate']['fingerprint'])
            self.assertTrue(isinstance(result['candidate_minus_baseline_loss'],float))

    def test_mismatch_ablation_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); toy_corpus(root/'corpus')
            first=root/'first'; second=root/'second'
            with redirect_stdout(StringIO()):
                train(arguments(root/'corpus',first,mode='baseline',steps=2))
                train(arguments(root/'corpus',second,mode='moe',steps=3))
            with self.assertRaisesRegex(ValueError,'schedules differ'):
                compare(first,second)

    def test_cannot_resume_when_experiment_switch_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); toy_corpus(root/'corpus'); out=root/'run'
            with redirect_stdout(StringIO()):
                train(arguments(root/'corpus',out,mode='baseline',steps=2,stop=1))
            mismatch=arguments(root/'corpus',out,mode='moe',steps=2)
            mismatch.resume=str(out/'latest.pt')
            with self.assertRaisesRegex(ValueError,'Resume setting changed: moe_experts'):
                train(mismatch)


if __name__ == '__main__':
    unittest.main()
