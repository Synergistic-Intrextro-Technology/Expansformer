# Corpus candidate acquisition and quality review

**Review date:** 2026-10-18  
**Status:** staged candidate material only. Nothing in either candidate batch has been admitted to `expansformer_dataset/` or accepted for training.

## What is staged

The original `topic_gap_batch_240/` remains untouched. Its revised sibling contains 240 text files, 30 each in eight domains, totaling 1,103,940 words by the dataset pipeline tokenizer. The pipeline preflight reported structural validity and no exact normalized duplicate groups; its diagnostic flags are screening prompts, not source-quality decisions. The revised batch remains Wikipedia-heavy: 172/240 records (71.7%). Other sources include Project Gutenberg, EIA, NASA, PubMed Central, CRS mirror copies, EPA, NOAA, USGS, and Census materials. License and source-quality status remain unreviewed for training admission.

The human-focused PMC/Europe PMC candidate batch now contains **259 retained articles**:

| Category | Retained |
| --- | ---: |
| Human cognition and behavior | 86 |
| Human–AI collaboration | 86 |
| Biological intelligence and AI | 87 |

The retained articles span 127 journal titles. Europe PMC metadata records article type and authors for the acquired batches; seven records still lack a DOI in the retrieved metadata. Article bodies contain 2,071–24,116 words. The current set includes research articles, reviews, and a small number of reports, protocols, editorials, and letters; article type alone is not a quality judgment.

Across both staged batches, the read-only review found **499 files with valid recorded hashes**, 499 unique exact content hashes, no integrity errors, and no unlisted text. Six duplicate flags concern multiple distinct articles from the same publisher host in the 240-document batch; they are source-group flags, not exact content duplicates. The human-focused batch has 259 unique PMCID values and no repeated exact or whitespace/case-normalized article text. A title-token overlap screen at Jaccard 0.60 returned no pairs; none of these automated checks proves semantic uniqueness.

## Acquisition and exclusions

The human-focused batch was acquired in three bounded Europe PMC rounds. Each retrieved article was required to carry an article-specific JATS license declaration explicitly permitting plain CC BY, have an abstract of at least 250 characters, and contain at least 1,500 extracted body words. Raw JATS XML is retained, and extraction omits references, figures, tables, acknowledgments, footnotes, and supplements. Search terms covered cognition/behavior, human–AI teamwork and decision-making, and biological/neural inspiration for AI. Linguistics and math were excluded by scope filters and follow-up screening.

The current independent XML audit scanned 259 retained records: **238 have parseable CC BY license URLs, and 21 explicitly declare plain CC BY in XML without a parseable license URL**. There are zero manifest/XML license mismatches and zero manifest errors. The text-only declarations were preserved without inventing license versions or URLs. Earlier non-CC-BY NC/ND records and scope-mismatched articles are kept in `topic_gap_human_expansion_excluded/` with reasons. Five records from the latest round were excluded after title/scope review: three language/linguistics items and two category mismatches.

The collector initially reused the `b02` filename prefix on the latest round and overwrote six earlier files in cognition and human–AI folders. The six original records were refetched by PMCID, their extracted text and raw XML restored, and the new round renamed with `b03` paths. The retained manifests then passed hash checks. The incident also overwrote the prior batch02 detailed screening reports and collection plan; `batch02_collection_plan_reconstructed.json` records what could be recovered, and the original detailed rejection lists are unavailable. `batch03_repair_log.json` records the repair. The collector now derives a new batch number and writes batch-specific paths and reports.

## License, factuality, and objective limits

- The JATS declaration audit is source evidence only. **It is not independent legal clearance.** Third-party content, rights in embedded materials, intended-use conditions, and jurisdiction-specific questions remain unreviewed. All current candidate rows remain `SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED`.
- `source_quality_status` remains `UNREVIEWED`. Peer-reviewed appearance, PMC indexing, journal title, and article type do not establish factual accuracy, balance, representativeness, or fitness for pretraining. The 259 PMC records are concentrated in one repository even though journal titles vary.
- No factual claims were independently checked, and no two-reviewer acceptance was recorded.
- These are source documents. Structured reasoning and epistemic examples were not authored or independently labeled.
- No separate held-out evaluation set was built; training/evaluation contamination was not measured.
- The canonical dataset remains empty. Its audit reports 0/1,000 training documents, 0/10 training domains, no reasoning/epistemic examples, no six-task evaluation coverage, and no two-reviewer acceptance.
- The staged read-only review was rerun after this acquisition; no unit tests or model training were run this turn.

The 499 candidates are useful acquisition material, not a ready training corpus. The next useful step is source-specific factual and license review, then structured reasoning/epistemic annotation and separate evaluation construction.
