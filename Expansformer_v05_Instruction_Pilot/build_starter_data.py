#!/usr/bin/env python3
"""Make a deterministic, unreviewed SYNTHETIC instruction-format smoke/pilot corpus.

NOT an instruction benchmark, not a legally cleared external corpus, not independent
of the author's templates. Test numbers are disjoint but task templates overlap.
Do not use this to claim factual, reasoning, or production capabilities.
"""
import hashlib
import json
from pathlib import Path


def row(split, ident, category, prompt, answer):
    return {'id': f'{split}-{ident}', 'category': category,
            'prompt': prompt, 'response': str(answer), 'origin': 'synthetic_deterministic_unreviewed_v05'}


def examples(split, values):
    result = []
    for n in values:
        offset = 2 + (n % 7)
        result += [
            row(split, f'addition-{n}', 'arithmetic',
                f'Question: What is {n} plus {offset}?\nAnswer: ', str(n + offset)),
            row(split, f'subtraction-{n}', 'arithmetic',
                f'Question: Subtract {offset} from {n + 20}.\nAnswer: ', str(n + 20 - offset)),
            row(split, f'parity-{n}', 'classification',
                f'Question: Is {n} even or odd?\nAnswer: ', 'even' if n % 2 == 0 else 'odd'),
            row(split, f'compare-{n}', 'comparison',
                f'Question: Which is larger, {n} or {n + offset}?\nAnswer: ', str(n + offset)),
        ]
    return result


def main():
    out = Path(__file__).with_name('data')
    out.mkdir(exist_ok=True)
    splits = {
        'train': examples('train', range(5, 65)),
        'dev': examples('dev', range(65, 80)),
        'test': examples('test', range(80, 95)),
    }
    # Distinct relationship/instruction tasks. No old audit prompts appear.
    topics = {
        'train': [('solar', 'The Sun provides energy to Earth.'),
                  ('gravity', 'Gravity is an attractive interaction between objects with mass.'),
                  ('photosynthesis', 'Photosynthesis converts light energy into chemical energy.'),
                  ('neuron', 'A neuron transmits information using electrical and chemical signals.')],
        'dev': [('evaporation', 'Evaporation is the change of a liquid into gas.')],
        'test': [('condensation', 'Condensation is the change of gas into liquid.')],
    }
    for split, facts in topics.items():
        for concept, explanation in facts:
            splits[split].append(row(split, f'definition-{concept}', 'definition',
                                     f'Question: Give one short definition of {concept}.\nAnswer: ', explanation))
    for split, records in splits.items():
        p = out / f'{split}.jsonl'
        p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    summary = {
        'origin': 'self-contained deterministic synthetic toy examples; unreviewed',
        'purpose': 'Exercise completion-only training and evaluate limited template transfer',
        'limitations': ['Only four numeric templates plus a few definitions',
                        'Same task templates recur across splits; number ranges are disjoint',
                        'Not sufficient evidence of instruction following or reasoning',
                        'Not an independent untouched benchmark unless never used for tuning'],
        'counts': {k: len(v) for k, v in splits.items()},
        'sha256': {k: hashlib.sha256((out / (k + '.jsonl')).read_bytes()).hexdigest() for k in splits},
    }
    (out / 'DATASET_NOTE.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
