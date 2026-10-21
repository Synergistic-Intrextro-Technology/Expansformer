"""Independent checks for new in-weight v0.4 additions; CPU only."""
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import torch
from train_expansformer_v01 import load_model_source
from train_expansformer_v04 import make_parser, train, generate, use_sdpa
from expansformer_v03_model import ResearchTransformerLM, ResearchConfig
from expansformer_v04_model import NeuralConfig, NeuralTransformerLM
from compare_runs_v04 import compare
from evaluate_adaptive_v04 import audit

ROOT = Path(__file__).resolve().parents[1]
load_model_source(str(ROOT/'expansformer_model'))
from model import ModelConfig


def args(data, folder, *, enhanced=False, stop=None, steps=3):
    cmd = ['train', '--data', str(data), '--out', str(folder), '--steps', str(steps),
           '--n-layers','2','--n-heads','4','--embed-dim','64', '--context-length','16',
           '--batch-size','2','--grad-accum','1','--eval-windows','2','--eval-every','1',
           '--save-every','1','--warmup-steps','1','--cpu-threads','2','--device','cpu',
           '--attention','sdpa','--amp','off','--tokenizer','byte']
    if enhanced:
        cmd += ['--memory-slots','8','--memory-every','1','--auxiliary-exits', '--next-token-heads']
    if stop:
        cmd += ['--stop-after',str(stop)]
    return make_parser().parse_args(cmd)


def corpus(data):
    data.mkdir()
    for i in range(8):
        (data/f'doc_{i}.txt').write_text((f'Provenance and strategy in experiment {i}. '
            'Evidence is required before accepting a claim. ')*18,encoding='utf-8')


class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def cfg(self):
        return ModelConfig(vocab_size=260,context_length=20,n_layers=2,n_heads=4,embed_dim=64,dropout=0)

    def test_zero_options_v03_exact_equivalence(self):
        torch.manual_seed(77)
        old=ResearchTransformerLM(self.cfg(),ResearchConfig())
        torch.manual_seed(77)
        new=NeuralTransformerLM(self.cfg(),NeuralConfig())
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        for k in old.state_dict():
            self.assertTrue(torch.equal(old.state_dict()[k],new.state_dict()[k]),k)
        x=torch.randint(0,260,(1,12))
        self.assertTrue(torch.equal(old(x)[0],new(x)[0]))

    def test_memory_and_prediction_have_actual_gradients(self):
        model=NeuralTransformerLM(self.cfg(),NeuralConfig(memory_slots=8,memory_every=1,
                                        auxiliary_exits=True,next_token_heads=True))
        x=torch.randint(0,260,(2,12))
        loss, parts=model.training_loss(x,x)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(parts['next_token_head_loss']),0)
        loss.backward()
        for p in [model.slot_memory.keys,model.slot_memory.values,
                  model.slot_memory.gates,model.slot_memory.query.weight,
                  model.next_token_heads[0].weight]:
            self.assertIsNotNone(p.grad)
            self.assertGreater(p.grad.abs().sum().item(),0)

    def test_memory_preserves_causality(self):
        model=NeuralTransformerLM(self.cfg(),NeuralConfig(memory_slots=8,memory_every=1,
                                        auxiliary_exits=True,next_token_heads=True)).eval()
        use_sdpa(model)
        x=torch.randint(0,260,(1,12)); changed=x.clone();changed[:,6:]=(changed[:,6:]+5)%260
        with torch.no_grad():
            a=model(x)[0]; b=model(changed)[0]
            c=model.adaptive_logits(x,.01)[0]; d=model.adaptive_logits(changed,.01)[0]
        self.assertTrue(torch.allclose(a[:,:6],b[:,:6],atol=1e-5,rtol=1e-5))
        self.assertTrue(torch.allclose(c[:,:6],d[:,:6],atol=1e-5,rtol=1e-5))

    def test_prediction_gate_never_claims_evidence(self):
        model=NeuralTransformerLM(self.cfg(),NeuralConfig(memory_slots=4,
                          auxiliary_exits=True,next_token_heads=True)).eval()
        x=torch.randint(0,260,(1,6))
        logits, depth, confidence = model.adaptive_logits(x,.01)
        self.assertEqual(depth,1)
        self.assertEqual(logits.shape,(1,6,260))
        self.assertTrue(0<=confidence<=1)
        with self.assertRaises(ValueError):
            model.adaptive_logits(x,1.1)

    def test_bad_settings_are_rejected(self):
        for opts in [NeuralConfig(next_token_heads=True), NeuralConfig(memory_slots=-1),
                     NeuralConfig(memory_every=0), NeuralConfig(next_token_head_weight=-.1),
                     NeuralConfig(memory_slots=4,memory_every=3),
                     NeuralConfig(auxiliary_exits=True,next_token_heads=True,next_token_head_weight=0)]:
            with self.assertRaises(ValueError):
                opts.validate(2)
        net=NeuralTransformerLM(self.cfg(),NeuralConfig(memory_slots=2))
        with self.assertRaisesRegex(ValueError,'trained'):
            net.logits_at_depth(torch.randint(0,260,(1,8)),1)

    def test_all_trainable_mechanisms_joint_gradient(self):
        model=NeuralTransformerLM(self.cfg(),NeuralConfig(
            moe_experts=4,moe_top_k=2,auxiliary_exits=True,
            memory_slots=4,memory_every=1,next_token_heads=True))
        x=torch.randint(0,260,(2,10))
        loss, parts=model.training_loss(x,x)
        loss.backward()
        self.assertGreater(model.blocks[0].ffn.router.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.exit_norms[0].weight.grad.abs().sum().item(),0)
        self.assertGreater(model.slot_memory.keys.grad.abs().sum().item(),0)
        self.assertGreater(model.next_token_heads[0].weight.grad.abs().sum().item(),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_train_resume_and_audit_real_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); corpus(root/'corpus')
            whole=root/'whole'; resumed=root/'resumed'; base=root/'base'
            with redirect_stdout(StringIO()):
                train(args(root/'corpus',whole,enhanced=True))
                train(args(root/'corpus',resumed,enhanced=True,stop=1))
                x=args(root/'corpus',resumed,enhanced=True); x.resume=str(resumed/'latest.pt'); train(x)
                train(args(root/'corpus',base,enhanced=False))
            a=torch.load(whole/'latest.pt',weights_only=True)
            b=torch.load(resumed/'latest.pt',weights_only=True)
            for k in a['model']:
                self.assertTrue(torch.equal(a['model'][k],b['model'][k]),k)
            self.assertEqual(a['step'],3)
            with self.assertRaisesRegex(ValueError,'Resume setting changed'):
                x=args(root/'corpus',resumed,enhanced=False); x.resume=str(resumed/'latest.pt');train(x)
            result=compare(base,whole)
            self.assertIsInstance(result['candidate_minus_baseline_loss'],float)
            with redirect_stdout(StringIO()) as out:
                g=make_parser().parse_args(['generate','--checkpoint',str(whole/'best.pt'),
                    '--prompt','Evidence:', '--experimental-adaptive-exit-threshold','.05',
                    '--max-new-tokens','3', '--temperature','0','--device','cpu'])
                generate(g)
            self.assertIn('Experimental depths:',out.getvalue())
            measure=audit(whole/'best.pt',root/'corpus',windows=2,device_name='cpu')
            self.assertEqual(measure['held_out_windows'],2)
            self.assertEqual(measure['full_depth'],2)
            with self.assertRaisesRegex(ValueError,'Data/provenance changed'):
                (root/'corpus'/'doc_0.txt').write_text('changed',encoding='utf-8')
                audit(whole/'best.pt',root/'corpus',windows=2,device_name='cpu')


if __name__=='__main__':
    unittest.main()
