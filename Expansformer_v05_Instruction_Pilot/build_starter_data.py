#!/usr/bin/env python3
"""Make a deterministic, unreviewed SYNTHETIC instruction-format smoke/pilot corpus.

NOT an instruction benchmark, not a legally cleared external corpus, not independent
of the author's templates. Test numbers are disjoint but task templates overlap.
Do not use this to claim factual, reasoning, or production capabilities.
"""
import argparse
import collections
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


def build_v1(out):
    """Original layout, unchanged. Kept byte-identical so v1 checkpoints stay valid."""
    out = Path(out)
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
    return summary


# ---------------------------------------------------------------------------
# Layout v2 (opt-in). Fixes three measurement problems found in v1:
#  1. 'compare' always answered with the second number, so copying scored 100%.
#  2. Number ranges were disjoint across splits for every numeric template, so
#     "unseen numbers" and "larger magnitude than ever trained" were confounded.
#  3. Definitions had 4/1/1 items, too few to measure anything per split.
# v2 writes to a separate directory; v1 'data/' is never touched. A checkpoint
# records dataset hashes, so v1 checkpoints keep working only on v1 data.
# ---------------------------------------------------------------------------
V2_ORIGIN = 'synthetic_deterministic_unreviewed_v05_layout2'
IN_RANGE = range(5, 80)          # every operand, shown number and answer stays < 100
EXTRAP = range(100, 130)         # every operand and answer is >= 100, none in train
IN_RANGE_PINNED_TO_TRAIN = (5, 79)   # range endpoints stay in train: dev/test never leave the trained range

TRAIN_DEFINITIONS = [
    ('solar', 'The Sun provides energy to Earth.'),
    ('gravity', 'Gravity is an attractive interaction between objects with mass.'),
    ('photosynthesis', 'Photosynthesis converts light energy into chemical energy.'),
    ('neuron', 'A neuron transmits information using electrical and chemical signals.'),
    ('friction', 'Friction is a force that resists the sliding of surfaces against each other.'),
    ('density', 'Density is the mass of a substance per unit of volume.'),
    ('ecosystem', 'An ecosystem is a community of living things interacting with their physical environment.'),
    ('magnet', 'A magnet is an object that produces a magnetic field and attracts certain metals.'),
    ('erosion', 'Erosion is the process by which wind, water or ice wears away and moves rock and soil.'),
    ('habitat', 'A habitat is the natural environment where an organism lives.'),
    ('atom', 'An atom is the smallest unit of an element that keeps the properties of that element.'),
    ('velocity', 'Velocity is the speed of an object together with its direction of motion.'),
    ('fossil', 'A fossil is the preserved remains or traces of an ancient organism.'),
    ('climate', 'Climate is the typical pattern of weather in a region over many years.'),
    ('molecule', 'A molecule is a group of atoms bonded together.'),
    ('respiration', 'Respiration is the process by which cells release energy from food.'),
    ('volcano', "A volcano is an opening in the Earth's crust through which molten rock and gas can erupt."),
    ('circuit', 'A circuit is a closed path along which electric current can flow.'),
    ('predator', 'A predator is an animal that hunts and eats other animals.'),
    ('orbit', 'An orbit is the curved path of one object around another under gravity.'),
    ('germination', 'Germination is the process by which a seed begins to grow into a plant.'),
    ('lever', 'A lever is a rigid bar that turns around a fixed point to help move a load.'),
    ('tide', "A tide is the regular rise and fall of sea level caused mainly by the Moon's gravity."),
    ('sediment', 'Sediment is solid material such as sand or silt that is moved and deposited by water, wind or ice.'),
]
DEV_DEFINITIONS = [
    ('evaporation', 'Evaporation is the change of a liquid into gas.'),
    ('pollination', 'Pollination is the transfer of pollen that allows plants to produce seeds.'),
    ('insulator', 'An insulator is a material that does not easily let heat or electric current pass through it.'),
    ('glacier', 'A glacier is a large body of ice that moves slowly over land.'),
    ('mineral', 'A mineral is a naturally occurring solid with a definite chemical composition.'),
    ('herbivore', 'A herbivore is an animal that eats mainly plants.'),
    ('acceleration', 'Acceleration is the rate at which velocity changes over time.'),
    ('cell', 'A cell is the basic structural and functional unit of living things.'),
    ('eclipse', 'An eclipse happens when one celestial body moves into the shadow of another.'),
    ('conductor', 'A conductor is a material that lets heat or electric current pass through it easily.'),
]
TEST_DEFINITIONS = [
    ('condensation', 'Condensation is the change of gas into liquid.'),
    ('precipitation', 'Precipitation is water that falls from clouds as rain, snow, sleet or hail.'),
    ('decomposer', 'A decomposer is an organism that breaks down dead plants and animals.'),
    ('weathering', 'Weathering is the breaking down of rock at the Earth\'s surface by wind, water, ice or living things.'),
    ('pressure', 'Pressure is force applied per unit of area.'),
    ('enzyme', 'An enzyme is a protein that speeds up a chemical reaction in a living thing.'),
    ('omnivore', 'An omnivore is an animal that eats both plants and animals.'),
    ('frequency', 'Frequency is the number of times a repeating event occurs in a given time.'),
    ('mass', 'Mass is the amount of matter in an object.'),
    ('nutrient', 'A nutrient is a substance that living things need in order to grow and stay healthy.'),
]


def _rank(tag, n):
    """Deterministic pseudo-random order that does not depend on parity or magnitude of n."""
    return hashlib.sha256(f'layout2|{tag}|{n}'.encode()).hexdigest()


def _assign(pool, sizes, pinned=()):
    """Split `pool` into named groups of exact `sizes`; the last group takes the remainder."""
    rest = sorted((n for n in pool if n not in pinned), key=lambda n: _rank('split', n))
    groups, i = {}, 0
    for name, size in sizes.items():
        groups[name] = rest[i:i + size]
        i += size
    return groups, rest[i:]


def _numeric_rows(split, ns, extrap):
    """Rows for one group of numbers. `extrap` only changes ids and categories."""
    swap_first = set(sorted(ns, key=lambda n: _rank('swap', n))[:len(ns) // 2])  # larger number comes first
    mark = '-x' if extrap else ''
    arith, comp = ('extrapolation_arithmetic', 'extrapolation_comparison') if extrap else ('arithmetic', 'comparison')
    rows = []
    for n in sorted(ns):
        offset = 2 + (n % 7)
        a, b = (n + offset, n) if n in swap_first else (n, n + offset)
        rows += [
            row(split, f'addition{mark}-{n}', arith,
                f'Question: What is {n} plus {offset}?\nAnswer: ', str(n + offset)),
            row(split, f'subtraction{mark}-{n}', arith,
                f'Question: Subtract {offset} from {n + 20}.\nAnswer: ', str(n + 20 - offset)),
            row(split, f'compare{mark}-{n}', comp,
                f'Question: Which is larger, {a} or {b}?\nAnswer: ', str(n + offset)),
        ]
        if not extrap:
            rows.append(row(split, f'parity-{n}', 'classification',
                            f'Question: Is {n} even or odd?\nAnswer: ', 'even' if n % 2 == 0 else 'odd'))
    for r in rows:
        r['origin'] = V2_ORIGIN
    return rows


def _compare_diagnostics(rows):
    out = {}
    for cat in ('comparison', 'extrapolation_comparison'):
        c = [r for r in rows if r['category'] == cat]
        if not c:
            continue
        first = sum(r['response'] == r['prompt'].split('larger, ')[1].split(' or ')[0] for r in c)
        out[cat] = {'n': len(c), 'answer_is_first_number': first, 'answer_is_second_number': len(c) - first}
    return out


def build_v2(out):
    out = Path(out)
    out.mkdir(exist_ok=True)
    in_range, train_n = _assign(IN_RANGE, {'dev': 15, 'test': 15}, IN_RANGE_PINNED_TO_TRAIN)
    train_n = sorted(train_n + list(IN_RANGE_PINNED_TO_TRAIN))
    extrap, _ = _assign(EXTRAP, {'dev': 15, 'test': 15})
    splits = {
        'train': _numeric_rows('train', train_n, False),
        'dev': _numeric_rows('dev', in_range['dev'], False) + _numeric_rows('dev', extrap['dev'], True),
        'test': _numeric_rows('test', in_range['test'], False) + _numeric_rows('test', extrap['test'], True),
    }
    defs = {'train': TRAIN_DEFINITIONS, 'dev': DEV_DEFINITIONS, 'test': TEST_DEFINITIONS}
    for split, facts in defs.items():
        for concept, explanation in facts:
            r = row(split, f'definition-{concept}', 'definition',
                    f'Question: Give one short definition of {concept}.\nAnswer: ', explanation)
            r['origin'] = V2_ORIGIN
            splits[split].append(r)
    for split, records in splits.items():
        p = out / f'{split}.jsonl'
        p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    summary = {
        'layout': 2,
        'origin': 'self-contained deterministic synthetic toy examples; unreviewed',
        'purpose': 'Exercise completion-only training and measure template transfer, kept separate from extrapolation',
        'fixes_relative_to_layout_1': [
            'compare operand order is balanced (answer is first or second number about half the time) so copying is not a winning strategy',
            'in-range dev/test numbers are interleaved with train numbers inside the same range (all values < 100)',
            'extrapolation is its own labeled category (extrapolation_arithmetic, extrapolation_comparison): every operand and answer >= 100, none in train',
            'definition has 24/10/10 items instead of 4/1/1',
        ],
        'limitations': ['Only four numeric templates plus definitions',
                        'Same task templates recur across splits',
                        'Definition exact-match on unseen concepts measures recall of text never seen in training; use masked answer CE/token accuracy for that category and treat exact match as uninformative',
                        'Definitions are short textbook statements written without independent review',
                        'Not sufficient evidence of instruction following or reasoning',
                        'Not an independent untouched benchmark unless never used for tuning',
                        'Layout 1 data/ is retained unchanged; checkpoints record dataset hashes, so layout 1 and layout 2 checkpoints are not interchangeable'],
        'counts': {k: len(v) for k, v in splits.items()},
        'category_counts': {k: dict(sorted(collections.Counter(r['category'] for r in v).items()))
                            for k, v in splits.items()},
        'compare_answer_position': {k: _compare_diagnostics(v) for k, v in splits.items()},
        'sha256': {k: hashlib.sha256((out / (k + '.jsonl')).read_bytes()).hexdigest() for k in splits},
    }
    (out / 'DATASET_NOTE.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--layout', choices=('v1', 'v2'), default='v1',
                        help='v1 (default) rewrites the original data/ exactly; v2 writes the corrected layout')
    parser.add_argument('--out', help='output directory (default: data for v1, data_v2 for v2)')
    args = parser.parse_args(argv)
    default = 'data' if args.layout == 'v1' else 'data_v2'
    out = Path(args.out) if args.out else Path(__file__).with_name(default)
    summary = build_v1(out) if args.layout == 'v1' else build_v2(out)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
