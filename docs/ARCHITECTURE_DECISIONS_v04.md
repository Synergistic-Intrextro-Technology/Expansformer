# Expansformer v0.4 — original-module analysis and neural integration decision

## Scope and evidence

Based on the **five original user-provided scripts** plus their existing `Expansformer_v03_Research` project. The original files are used as **architectural references, not copied into the runtime**. The intact prior trainer `train_expansformer_v03.py`, model `expansformer_v03_model.py`, tests, and original transformer are retained. This v0.4 release is a separate experimental code path.

### Component-by-component findings

| First uploaded source | Real code inspected | Correct conclusion | v0.4 decision |
|---|---|---|---|
| `enhanced_agi_core.py` | `CustomTransformerCore`, `AdaptiveDepthController`, `NeuralMemorySystem`, `PathwayManager`, `_train_on_batch` | Implements actual `nn.Module` layers and SQLite-backed memory. However, `_train_on_batch` **only clears** its data, and pathway quality rewards response length. The neural memory's embedding encoder/decoder/attention are not evidence that a combined model learns reliably. | New **trainable, static parametric slots**, with a learned gate and gradient tests; no SQLite memory inside model. Existing intermediate exits retained. |
| `cognitive_orchestrator_v2.py` | `AdaptiveDepthController`, `CapabilityMatcher`, `MultiPathwayExecutor`, `SelfOptimizingArchitecture` | Context-length/keyword heuristics are not calibrated indications that a shorter reasoning pathway suffices. | Add intermediate **next-token accuracy predictor** supervised by actual corpus targets. Adaptive exits remain *disabled by default*. |
| `enhanced_innovation_engine.py` | `EnhancedInnovationDiscoveryEngine._initialize_advanced_embeddings`, synthetic components, synergy rankings | Creates randomly initialized, keyword-biased vectors and many symbolic capability names. Not a trained discovery model or validated novelty measure. | Use its hypothesis-generation idea for **external experimental design only**. No random ‘knowledge scores’ in the model or training objective. |
| `autonomous_ai_development_engine.py` | `ProjectScanner`, `ExperimentManager._optimize_learning_rate`, `_search_architecture` | Has useful project and experiment representations, but some training paths operate on dummy/random data and depend on unavailable project trainers. | Provide **actual checkpointed, comparable training runs**, rather than importing experiment results as model capability. |
| `ApexModal(1).py` | `FusionHub`, `ThresholdLearner`, `DriftDetector`, `ExperienceBuffer`, `LifelongLearningCore` | Genuine event-stream, rule-engine and bounded-buffer design, primarily for equipment telemetry. Its threshold learner uses proxy labels and does not train language model weights. | Preserve for later external event/retrieval processing; do **not** combine with LM token stream or use proxy precision as truth labels. |

### Three candidate strategies considered

1. Import all first-five modules into the model: **rejected**. Many are external runtime systems with varying assumptions, side effects, and proxy metrics. No verified common neural interface.
2. Train v0.3 then retro-fit all modules: **rejected for trainable mechanisms**, which would introduce newly random layers and require more training.
3. **Selected:** keep the stable decoder as control; adapt only explicitly trainable ideas into two opt-in modules, measure them separately, and choose the larger training architecture only after meaningful hold-out benchmarks.

### Specific neural additions

**TrainableSlotMemory**: Learned slot keys and values read by per-token queries, gated residual added after every N transformer blocks. It has no retrieval, citation, external data ingestion, or remembered conversations. Its parameters are optimized by next-token loss. Each token consults fixed slots, so it does not introduce access to future tokens. Optional `--memory-slots 0` disables it completely.

**Intermediate next-token predictor**: Independent head per intermediate layer trained by binary cross-entropy to estimate whether that layer's top predicted token matches the *observed* next training token. This replaces simplistic text-length confidence heuristics with a measurable label. It is **not calibrated factual confidence**, proof of reasoning, hallucination detection, or source verification. The signal is isolated from backbone gradients (`hidden.detach()`); `--next-token-heads` requires `--auxiliary-exits`. Adaptive depth inference is separate opt-in, requires held-out evaluation, and defaults to OFF.

**Preserved**: v0.3's trainable sparse MoE routing and auxiliary exits, original causal transformer, audit-able corpus loader, checkpoint/resume, SDPA, and original test suite.

### Toy experiment observation — do not generalize

See `ablations_v04_toy.json`. All arms used the same tokenizer, split, optimizer settings, seed and 12 CPU optimizer steps on repetitive toy text.

- Dense + exits: 123,520 parameters; final held-out loss **5.281183**.
- Memory + exits: 132,738 parameters; loss **5.283247** (**+0.002064 worse** than dense).
- Memory + exits + predictor: 132,803 parameters; loss **5.282079** (slightly lower than memory alone, but **higher** than dense).

Therefore **neither new addition has demonstrated superiority**. Next decision: run a larger controlled comparison on a truly diverse non-contaminated corpus, with multiple seeds, compute/latency logging and factuality evaluation independent of training. If there is no meaningful improvement, keep the dense baseline.

### Guardrails and voids

- No pretrained weights, no unvalidated automatic self-modification, no implicit network calls, no license proof, no truth assertion by logits, no Ollama export.
- User's v0.2 trained checkpoint remains intact and is *not* upgradable in place to v0.4. Retrain experimental variants from random initialization.
- V0.4 checkpoint format cannot be resumed as v0.3 and vice versa; CLI rejects mismatches.
- Tested on CPU here; user's CUDA environment, throughput, VRAM and third-party model inference compatibility are **not yet verified for v0.4**.
- Early-exit predictor quality and adaptive runtime speedups require held-out calibration and explicit comparison; single toy corpus does not establish these properties.
