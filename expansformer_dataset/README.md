# Expansformer Knowledge & Reasoning Dataset

This is a local-first dataset pipeline alongside the v0.3 model code. The
importer does not make network requests, decide legal rights, verify facts, or
assert source quality. An optional separate Wikipedia collector makes network
requests only when explicitly invoked. Training text and independent evaluation
data have separate trees and manifests; importing a document as training text
does not make it evidence for generated claims.

## Layout

`python dataset_pipeline.py init --root ./expansformer_dataset` creates the
repository layout. `raw/` holds source documents, `curated/` holds
document-level pretraining and validation splits plus typed reasoning and
epistemic examples, `evaluation/` holds language, factuality, reasoning,
generalization, uncertainty-calibration, and abstention sets, and
`evidence/source_records/` holds the
provenance record for each document. JSONL manifests are under `manifests/`.

## Importing local documents

Prepare a UTF-8 JSONL metadata file, one row for each `.txt` or `.md` document
in the input directory. Create a starting template with exact content hashes:

```bash
python dataset_pipeline.py template --input ./authorized_documents > ./authorized_documents.metadata.jsonl
```

Edit each row to replace `REQUIRED` with source-specific metadata and review the
provisional hash-derived `source_id`. Replace it with a durable source identifier
if it must remain stable across revisions. `path` is relative to the input directory.
Required fields:

```json
{"source_id":"example-limits-2024","path":"calculus/limits.md","title":"Limits","publisher":"Example Press","source_url":"https://example.org/limits","publication_date":"2024-01-01","license_id":"CC-BY-4.0","license_url":"https://creativecommons.org/licenses/by/4.0/","license_status":"SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED","domain":"mathematics","source_type":"textbook","content_hash":"sha256:<hex>"}
```

`content_hash` must match the exact source bytes. This guards transfer errors;
it does not verify truth, license validity, or authenticity. Import into
`raw/` or `evaluation/` with an explicit partition:

Before reviewing or importing a staged batch, run a read-only preflight:

```bash
python dataset_pipeline.py preflight \
  --input ./wikipedia_stage_100 \
  --metadata ./wikipedia_stage_100/manifest.jsonl
```

Preflight verifies that text files match metadata rows and byte hashes, then
reports word counts, domain/license/review status, exact normalized duplicates,
and screening flags for short text, repeated lines, likely formula markup, and
low unique-word fraction. It counts wiki heading and emphasis markers separately.
Flags are prompts for human inspection; they do not automatically accept or
reject a document. Preflight makes no dataset
changes and does not establish licensing, source quality, or factual accuracy.

For Wikipedia extracts, prepare a separate candidate copy after source preflight:

```bash
python dataset_pipeline.py prepare-wikipedia \
  --input ./wikipedia_stage_100 \
  --metadata ./wikipedia_stage_100/manifest.jsonl \
  --out ./wikipedia_stage_100_normalized
```

The output directory must not already exist. The command preserves the staged
source, records the original and derived hashes, and applies only Unicode NFC,
heading/emphasis-marker, and horizontal-whitespace normalization. It leaves
formula text unchanged. The derived manifest retains the original license and
quality-review states; this preparation step is not admission or review. Run
`preflight` against the derived directory before human review and import.

Before import, change every `license_status` to `INDEPENDENTLY_REVIEWED` only
after a source-specific human review, and add a `license_review_evidence`
reference for that review. The importer rejects source-declared, unknown, or
restricted license status. The evidence reference is not authenticated by this
tool and does not constitute legal clearance.

```bash
python dataset_pipeline.py import --root ./expansformer_dataset \
  --input ./authorized_documents --metadata ./authorized_documents.jsonl \
  --partition training --domain mathematics
```

Evaluation imports use `--partition evaluation --track factuality` (or
`language`, `reasoning`, `abstention`). Evaluation material is rejected if an
identical normalized document or `source_group` is already in the training
inventory, and training imports are rejected if they overlap evaluation. If
omitted, `source_group` defaults to the document's `source_id`; supply a shared
work/source group for chapters or other related material. The pipeline cannot
detect paraphrase or semantic leakage.

## Staged topic-gap candidates

The repository also contains `../topic_gap_batch_240_revised/`, a separate
candidate batch with 30 documents in each of eight topic labels. It combines
the original staged set with selected NASA, USGS, EPA, NOAA, EIA, PubMed Central,
Congressional Research Service mirror, and Project Gutenberg sources. Its
manifest marks sources as `UNREVIEWED` and licenses as
`SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED`; the batch has not been imported.
See `../topic_gap_batch_240_revised_QUALITY_REVIEW.md` and
`../topic_gap_source_inventory/candidate_manifest.jsonl` for the selection
review and the broader acquired candidate inventory. A valid preflight means
the files and manifest agree; it does not mean the documents passed factual,
rights, contamination, or human-quality review.

An additional human-focused candidate batch is staged separately at
`../topic_gap_human_expansion/`, with separate categories for human cognition
and behavior, human–AI collaboration, and biological intelligence/AI. It
currently contains 259 retained documents: 86 human cognition/behavior,
86 human–AI collaboration, and 87 biological intelligence/AI. Each selected
record has a raw XML snapshot, article-body text, source manifest, and
article-level license declaration. All records remain unreviewed and
unimported. The license audit found 238 plain CC BY declarations with
parseable URLs and 21 explicit plain CC BY declarations without parseable
URLs; neither group is independent legal clearance. See the quality review
above and `../topic_gap_human_expansion/dedup_review.json` for current
duplicate and diversity triage. Excluded acquisition artifacts are kept separately in
`../topic_gap_human_expansion_excluded/`; the category folders contain only
manifest-selected document text files. The CC BY declaration is source
metadata, not independent legal clearance.

## Optional Wikipedia acquisition

For an explicit network acquisition, review a category list first. The
candidate list at `expansformer_dataset/sources/wikipedia_categories.example.jsonl`
maps ten requested domains to seed categories with depth zero. The names are
not validated or curated by this repository; inspect each category page and
edit the list before collection. Use one row per category:

```json
{"domain":"mathematics","category":"Linear algebra","max_depth":0}
{"domain":"physics","category":"Thermodynamics","max_depth":1}
```

Then run the separate bounded collector (requests are sequential and capped):

```bash
python wikipedia_acquisition.py collect \
  --out ./wikipedia_acquisition_incoming \
  --categories ./expansformer_dataset/sources/wikipedia_categories.example.jsonl \
  --user-agent 'ExpansformerDataset/0.1 (REPLACE_WITH_YOUR_CONTACT_EMAIL_OR_URL)' \
  --max-pages 1000 --min-words 80 --delay 1.0
# Review each collected article's license notices and attribution. Edit its
# manifest row to mark INDEPENDENTLY_REVIEWED and attach a review evidence ref.
python dataset_pipeline.py import --root ./expansformer_dataset \
  --input ./wikipedia_acquisition_incoming \
  --metadata ./wikipedia_acquisition_incoming/manifest.jsonl \
  --partition training
```

The acquisition directory is a staging copy; the importer copies accepted text
into the canonical `raw/` tree. Keep it until the dataset audit and backup are
complete, then manage the temporary copy according to local retention policy.

This collector follows MediaWiki's category-member API and requires an
identifying User-Agent; see [Categorymembers](https://www.mediawiki.org/wiki/API:Categorymembers/en)
and [API etiquette](https://www.mediawiki.org/wiki/API:Etiquette/en). Wikimedia
states that most text is under CC BY-SA 4.0 and GFDL, with possible imported
content and article-specific notices. Retain page/history links and review
each article's notices and attribution conditions before training or
redistribution; see [Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Terms_of_Use).
The collector labels rights as `SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED` and
does not provide that review; direct import will fail until every row has been
reviewed and updated. Category depth can pull in unrelated pages, so
inspect domain counts and quality reports before admitting the data.
The staged manifest separates the CC BY-SA license URL from Wikimedia's Terms
of Use URL, records a GFDL reference, and flags the revision-specific review
scope. These fields are carried into `manifests/licenses.jsonl`; they are
provenance clues, not a rights determination.

After importing at least two documents:

```bash
python dataset_pipeline.py split --root ./expansformer_dataset --seed 42 --validation-fraction 0.15
python dataset_pipeline.py audit --root ./expansformer_dataset --require-coverage-gates
```

The audit's top-level `valid` field means that files, manifests, hashes, and
partition links are structurally consistent. It does not mean that the dataset
meets the project objective. The nested `objective_readiness` report measures
coverage gates separately: at least 1,000 training documents, 10 training
domains, structured reasoning and epistemic examples, all six evaluation task
types (`language_modeling`, `factuality`, `reasoning`, `generalization`,
`uncertainty_calibration`, and `abstention`), and recorded two-reviewer source
quality acceptance. The report continues to mark source-quality independence,
factuality, and semantic-contamination verification as unverified; file counts
and review metadata do not prove those claims.
`--require-coverage-gates` returns exit code 2 when one of the listed coverage
gates is missing; structural manifest failures return exit code 1. A zero exit
from the coverage gate means the recorded counts and example/task presence
thresholds are met. It does not establish source accuracy, reviewer identity,
factuality, or benchmark independence.

The split is deterministic, keeps each `source_group` intact, and selects
validation groups from the domain with the lowest current validation fraction.
It reports actual train and validation counts by domain; large or multi-domain
groups can force deviations from the requested fraction. It preserves
training/evaluation separation. The output remains ordinary text plus
`manifest.jsonl`, so it can be used as v0.3 `--data` / `--val-data` after review.

## Structured examples

Reasoning and epistemic examples are JSONL records imported into
`curated/reasoning/` and `curated/epistemic_training/`. Reasoning records use
task types `causal`, `counterfactual`, `tradeoff`, `cross_domain_synthesis`, or
`experimental_problem_solving`; cross-domain records require at least two
domain tags and a named shared mechanism. Experimental records retain the
hypothesis, experiment design, observed outcome, interpretation, and model
revision as separate fields. Epistemic records separately list supported and
disputed claims, assumptions, unknowns, and conclusions. Every record references
source IDs and records from the training family must point only to training
sources. Evaluation records must point only to evaluation sources.

These records are not silently converted to prose for pretraining. Evaluation
tracks use the same explicit record import path and remain outside curated
training.

Evaluation task types are `language_modeling`, `factuality`, `reasoning`,
`generalization`, `uncertainty_calibration`, and `abstention`; each must be
imported into its corresponding track (`language_modeling` uses `language`).
Every evaluation record must define a nonempty scoring `metric` and `protocol`.
This checks that a task is labeled and scorable at the schema level; it does not
validate whether the expected answers, metric, or protocol are sound. The
pipeline stores these protocols; it does not run model evaluation or calculate
calibration scores.

## Limits

Cheap diagnostics (length, repeated lines, vocabulary diversity, hashes, and
domain counts) are triage signals, not independent quality or factuality
verification. `SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED` records a claim
provided by the importer. The importer has no web calls and does not provide
legal clearance. A benchmark is independent only if its source and content
were kept out of training and tuning; this tool can check exact/normalized
document overlap, not prove that history.

Record independent source-quality reviews separately from source metadata. A
review JSONL row must include a stable review ID, source ID, reviewer ID and
affiliation, a self-declared `relationship_to_source: "independent"`, ISO
review date, method, nonempty evidence references, a decision, and ratings from
0 to 2 for `source_authority`, `editorial_process`, `domain_relevance`,
`factual_support`, and `transparency`. Two distinct reviewers from distinct
affiliations must both accept and score at least 1 in every category before
the audit records a two-reviewer acceptance:

```bash
python dataset_pipeline.py import-quality-reviews \
  --root ./expansformer_dataset --input ./quality_reviews.jsonl
python dataset_pipeline.py split --root ./expansformer_dataset \
  --require-quality-review
```

The audit checks the record shape and linkage. Reviewer independence and the
evidence references are self-reported and are not authenticated or inspected
by this software. A recorded acceptance is a workflow result, not proof that a
source's every claim is true.
