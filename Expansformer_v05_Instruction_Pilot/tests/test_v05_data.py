"""Regression guards for the v0.5 starter datasets.

Run from the repository root:
    python -m unittest discover -s Expansformer_v05_Instruction_Pilot/tests -v

The guard function `layout_violations` is the living proof. It is run on layout 2
(must be clean) and on layout 1 as a negative control (must flag the known
measurement flaws, so the guard is shown to be able to fail).
Everything except the real-loader test is standard library only.
"""
import hashlib
import json
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1]
ROOT = PILOT.parent
sys.path.insert(0, str(PILOT))
import build_starter_data as bsd  # noqa: E402

SPLITS = ('train', 'dev', 'test')
MIN_ITEMS_PER_MEASURED_CATEGORY = 10
NUMERIC = ('arithmetic', 'comparison', 'classification', 'extrapolation_arithmetic', 'extrapolation_comparison')
EXTRAP = ('extrapolation_arithmetic', 'extrapolation_comparison')


def load(directory):
    return {s: [json.loads(l) for l in (Path(directory) / f'{s}.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
            for s in SPLITS}


def ints(text):
    return [int(x) for x in re.findall(r'\d+', text)]


def compare_first_rate(rows):
    """Fraction of comparison rows whose answer is the first number in the prompt."""
    first = [r['response'] == re.search(r'larger, (\d+) or', r['prompt']).group(1) for r in rows]
    return sum(first) / len(first)


def layout_violations(data):
    """Return a list of human-readable violations; empty means the layout passes."""
    bad = []
    # 1. compare must not be solvable by copying a fixed position
    for split, rows in data.items():
        for cat in ('comparison', 'extrapolation_comparison'):
            c = [r for r in rows if r['category'] == cat]
            if c and not 0.4 <= compare_first_rate(c) <= 0.6:
                bad.append(f'compare-position: {split}/{cat} answer is the first number in {compare_first_rate(c):.0%} of rows')
    # 2. in-range dev/test rows must stay inside the trained range; extrapolation rows must be beyond it
    train_max = max(max(ints(r['prompt'] + ' ' + r['response'])) for r in data['train'] if r['category'] in NUMERIC)
    for split in ('dev', 'test'):
        for r in data[split]:
            if r['category'] in EXTRAP:
                if int(r['response']) <= train_max:
                    bad.append(f'extrapolation-not-beyond-train: {r["id"]}')
            elif r['category'] in NUMERIC and max(ints(r['prompt'] + ' ' + r['response'])) > train_max:
                bad.append(f'in-range-exceeds-train-range: {r["id"]}')
    # 3. every measured category in dev/test needs enough items to measure anything
    for split in ('dev', 'test'):
        for cat, n in Counter(r['category'] for r in data[split]).items():
            if n < MIN_ITEMS_PER_MEASURED_CATEGORY:
                bad.append(f'too-few-items: {split}/{cat} has {n}')
    # 4. definitions: disjoint concepts and no repeated answers
    concepts = {s: {r['id'].split('definition-')[1] for r in data[s] if r['category'] == 'definition'} for s in SPLITS}
    for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        if concepts[a] & concepts[b]:
            bad.append(f'definition-concept-overlap: {a}/{b}')
    answers = [r['response'] for s in SPLITS for r in data[s] if r['category'] == 'definition']
    if len(answers) != len(set(answers)):
        bad.append('definition-duplicate-answers')
    return bad


class TestLayout1Unchanged(unittest.TestCase):
    """v1 stays exactly as shipped so existing v1 checkpoints keep a valid dataset hash."""

    def test_committed_v1_matches_recorded_hashes(self):
        note = json.loads((PILOT / 'data' / 'DATASET_NOTE.json').read_text())
        for s in SPLITS:
            self.assertEqual(hashlib.sha256((PILOT / 'data' / f'{s}.jsonl').read_bytes()).hexdigest(), note['sha256'][s])

    def test_default_builder_output_is_byte_identical_to_committed_v1(self):
        with tempfile.TemporaryDirectory() as d:
            bsd.build_v1(Path(d) / 'v1')
            for name in [f'{s}.jsonl' for s in SPLITS] + ['DATASET_NOTE.json']:
                self.assertEqual((Path(d) / 'v1' / name).read_bytes(), (PILOT / 'data' / name).read_bytes(), name)

    def test_negative_control_guard_flags_known_v1_flaws(self):
        found = layout_violations(load(PILOT / 'data'))
        kinds = {v.split(':')[0] for v in found}
        self.assertIn('compare-position', kinds)            # answer is always the second number
        self.assertIn('in-range-exceeds-train-range', kinds)  # numeric ranges disjoint across splits
        self.assertIn('too-few-items', kinds)               # definition: one item in dev and test


class TestLayout2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = PILOT / 'data_v2'
        cls.data = load(cls.dir)

    def test_guards_pass(self):
        self.assertEqual(layout_violations(self.data), [])

    def test_copy_baselines_cannot_win_on_compare(self):
        for split in ('dev', 'test'):
            for cat in ('comparison', 'extrapolation_comparison'):
                rows = [r for r in self.data[split] if r['category'] == cat]
                rate = compare_first_rate(rows)
                self.assertLessEqual(max(rate, 1 - rate), 0.6, f'{split}/{cat}')

    def test_extrapolation_is_labeled_and_absent_from_train(self):
        self.assertFalse([r for r in self.data['train'] if r['category'] in EXTRAP])
        for split in ('dev', 'test'):
            cats = {r['category'] for r in self.data[split]}
            self.assertTrue(set(EXTRAP) <= cats, split)

    def test_each_split_has_the_documented_category_counts(self):
        note = json.loads((self.dir / 'DATASET_NOTE.json').read_text())
        for s in SPLITS:
            self.assertEqual(dict(Counter(r['category'] for r in self.data[s])), note['category_counts'][s])
            self.assertEqual(hashlib.sha256((self.dir / f'{s}.jsonl').read_bytes()).hexdigest(), note['sha256'][s])

    def test_no_prompt_or_id_overlap_across_splits(self):
        prompts, ids = set(), set()
        for s in SPLITS:
            for r in self.data[s]:
                key = ' '.join(r['prompt'].lower().split())
                self.assertNotIn(key, prompts, r['id'])
                self.assertNotIn(r['id'], ids)
                prompts.add(key)
                ids.add(r['id'])

    def test_answers_are_correct(self):
        for s in SPLITS:
            for r in self.data[s]:
                nums = ints(r['prompt'])
                cat = r['category']
                if cat in ('arithmetic', 'extrapolation_arithmetic'):
                    if 'plus' in r['prompt']:
                        self.assertEqual(int(r['response']), nums[0] + nums[1], r['id'])
                    else:
                        self.assertEqual(int(r['response']), nums[1] - nums[0], r['id'])
                elif cat in ('comparison', 'extrapolation_comparison'):
                    self.assertEqual(int(r['response']), max(nums), r['id'])
                elif cat == 'classification':
                    self.assertEqual(r['response'], 'even' if nums[0] % 2 == 0 else 'odd', r['id'])

    def test_rebuild_is_deterministic_and_matches_committed_files(self):
        with tempfile.TemporaryDirectory() as d:
            bsd.build_v2(Path(d) / 'v2')
            for name in [f'{s}.jsonl' for s in SPLITS] + ['DATASET_NOTE.json']:
                self.assertEqual((Path(d) / 'v2' / name).read_bytes(), (self.dir / name).read_bytes(), name)

    def test_real_trainer_loader_accepts_layout2(self):
        """Wiring check: the trainer's own fail-closed loader (leakage, duplicates, prompt format)."""
        try:
            sys.path[:0] = [str(ROOT), str(PILOT)]
            from train_expansformer_v05_instruction import read_dataset
        except Exception as exc:  # torch or project modules unavailable
            self.skipTest(f'trainer not importable here: {exc!r}')
        splits, hashes = read_dataset(self.dir)
        self.assertEqual({k: len(v) for k, v in splits.items()}, {s: len(self.data[s]) for s in SPLITS})
        self.assertEqual(set(hashes), set(SPLITS))


if __name__ == '__main__':
    unittest.main()
