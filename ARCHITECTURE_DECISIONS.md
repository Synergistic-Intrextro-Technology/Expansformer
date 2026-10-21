# Decision record — Architecture additions before expensive pretraining

**Scope:** The original Expansformer transformer backbone and audited v0.2 code are immutable in this release. New functionality is opt-in via a new model/trainer and checkpoint format; no external agent frameworks are imported or allowed to mutate weights or training data.

**Three candidates considered:**

1. Train dense model first and attach external orchestrators after: safest, but excludes joint training of trainable expert routing/exit heads.
2. Import all five experimental Python modules into the language model before pretraining: rejected. Their APIs, optional dependencies, heuristic scores and execution semantics do not form a unified, tested neural computation graph.
3. **Selected:** Preserve a dense control, introduce two small, explicit trainable mechanisms, benchmark ablations, then pretrain the selected model; add retrieval/cognition later behind separately tested contracts.

## Module disposition

| Source | Reusable concept | Current disposition |
| --- | --- | --- |
| `expansformer_model/model.py` | Decoder-only backbone, causal attention, RMSNorm, SwiGLU | Original source preserved; subclassed in `expansformer_v03_model.py` |
| `expansformer_model/MoE.py` | Mixture of experts exploration | Separate classifier example; **not** plugged into LM. New per-token sparse experts built and tested instead. |
| `enhanced_agi_core.py` | Multiple pathways, experimental adaptive depth, vector memory | Trainable intermediate exits only; no imported AGI core, pretrained model dependencies, or memory store |
| `cognitive_orchestrator_v2.py` | Adaptive work allocation, pathway selection | Inference orchestration is deferred, not represented by an uncalibrated `confidence` score |
| `autonomous_ai_development_engine.py` | Experiment logs, comparison, optimizer proposal | Deterministic independently testable logs/comparison; no automatic self-modifying code |
| `enhanced_innovation_engine.py` | Architecture hypothesis generation | Offline research only until a candidate's gains are independently measured |
| `ApexModal(1).py` | Multimodal event processing and experience buffer | Later external system, not a language-model training modality |

## Acceptance contract

- **Difference:** new trainable MoE and exits can compute training gradients, train and resume, save/reload, and evaluate without altering the original architecture path.
- **Baseline:** identical model config, data split, tokenizer, seed, optimizer schedule, training/eval windows, and step count.
- **Metric:** primary held-out next-token cross-entropy; lower is better. Capture expert parameter cost and measured latency before choosing production architecture.
- **Threshold:** do not automatically select candidate on a single tiny experiment. Demand repeatability on more seeds, adequate domain-diverse held-out data, and tolerable inference/training cost before accepting architectural migration. This release makes no claims about such gains.
- **Safety:** never call generated predictions verified facts; disallow silent cross-version checkpoint resume; preserve original files; no unknown network downloads; failures preserve recoverable checkpoints.
- **Revalidation:** after changing data, tokenizer, model flags, PyTorch version, or runtime backend, rerun smoke, resume, and ablation tests.
- **Authority:** selecting architectural trade-offs and any large pretraining expenditure remains the user's decision.

## Future gates

1. Larger licensed dataset and streaming tokenizer/data loader. The tiny 32-document Wikipedia set is for prototyping.
2. Proper untampered held-out evaluation on language and reasoning with factual abstention separately measured.
3. Additional efficient attention / KV-cache experiments with latency tests.
4. Optional external cognitive orchestration and evidence library once explicit contracts and end-to-end evidence are available.
5. Ollama compatibility requires model-specific inference backend support, not merely renaming weights or writing GGUF.
