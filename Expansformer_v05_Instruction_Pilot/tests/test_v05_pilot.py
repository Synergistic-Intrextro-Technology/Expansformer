"""Run after installing v0.5 trainer next to the original project modules.
python3 -m unittest discover -s Expansformer_v05_Instruction_Pilot/tests -v
"""
import json
from pathlib import Path
import tempfile
import unittest

from train_expansformer_v01 import OwnTokenizer, SPECIAL
from train_expansformer_v05_instruction import read_dataset, tokenize_examples, collate
import torch


class TestV05Pilot(unittest.TestCase):
    def test_answer_only_mask(self):
        tok = OwnTokenizer()
        ex = {'id':'x1','prompt':'Question: What is 2 plus 3?\nAnswer: ',
              'response':'5', 'category':'arithmetic', 'origin':'synthetic'}
        _, x, y = tokenize_examples([ex], tok, 128)[0]
        prefix_len = 1 + len(tok.encode(ex['prompt']))
        self.assertTrue(torch.all(y[:prefix_len-1] == -1).item())
        self.assertEqual(int(y[prefix_len-1]), tok.encode('5')[0])
        self.assertEqual(int(y[-1]), SPECIAL['<EOS>'])
        self.assertEqual(int((y!=-1).sum()), len(tok.encode('5')) + 1)

    def test_right_pad_never_trains(self):
        tok = OwnTokenizer()
        ex = lambda name, ans: {'id': name, 'prompt':f'Question: Say {name}.\nAnswer: ',
                                 'response': ans, 'category':'instruction','origin':'synthetic'}
        batch = tokenize_examples([ex('a','one'),ex('long name','different longer answer')], tok, 128)
        x,y = collate(batch, torch.device('cpu'))
        self.assertEqual(x.size(), y.size())
        self.assertEqual(int(y[0,-1]), -1)
        self.assertEqual(int(y[1,-1]), SPECIAL['<EOS>'])

    def test_reject_cross_split_prompt_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            def row(ident, text):
                return {'id':ident,'prompt':'Question: '+text+'\nAnswer: ',
                        'response':'yes','category':'test','origin':'synthetic'}
            for split, values in [('train',[row('t','Duplicated')]),
                                  ('dev',[row('d',' duplicated  ')]),
                                  ('test',[row('x','Different')])]:
                (p/(split+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in values)+'\n')
            with self.assertRaisesRegex(ValueError, 'Prompt leakage'):
                read_dataset(p)

    def test_reject_truncation(self):
        tok=OwnTokenizer()
        row={'id':'big','prompt':'Question: '+ ('x'*150)+'\nAnswer: ',
             'response':'hello','category':'test','origin':'synthetic'}
        with self.assertRaisesRegex(ValueError, 'context limit'):
            tokenize_examples([row],tok,64)


if __name__ == '__main__':
    unittest.main()
