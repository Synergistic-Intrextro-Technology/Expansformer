# Expansformer: integrated neural training kit

This kit installs the **v0.4 trainable neural implementation** into your **existing v0.3 project** without changing its original trainer, model, acquisition scripts, or canonical dataset. The original source modules contributed ideas, not an automatic copy-and-execute integration.

Built into the trainable v0.4 model:
- `enhanced_agi_core.py` → trainable parameter memory slots and intermediate depth (adapted; not the original implementation).
- `cognitive_orchestrator_v2.py` → trainable intermediate next-token predictor for experimental adaptive depth.
- `enhanced_innovation_engine.py` → experimental routing inspiration; sparse learned mixture-of-experts replaces speculative heuristic scoring.
- `autonomous_ai_development_engine.py` → reproducible training measurements, checkpointing and separate comparisons, not code self-modification.
- `ApexModal(1).py` → monitoring/experience ideas stay outside the neural forward pass; raw industrial/sensor code is **not** trainable on a text corpus and is not imported.

## Commands (WSL)

Unzip `Expansformer_Train_Kit.zip` into your existing v0.3 project directory. The ZIP contains only `Expansformer_Train_Kit/` and does not overwrite existing project files.

```bash
cd ~/ACTIVE_PROJECTS/Expansformer_v03_Research
source ../Expansformer_v02_Research/.venv/bin/activate

python Expansformer_Train_Kit/run.py install --project .
python Expansformer_Train_Kit/run.py smoke --project .

# Source-declared candidate data; isolated experimental copy; no admission/approval changes.
python Expansformer_Train_Kit/run.py prepare --project . --allow-unreviewed-experiment

# First training segment: all four trainable capabilities enabled.
python Expansformer_Train_Kit/run.py train --project . \
  --allow-unreviewed-experiment --steps 1000 --stop-after 100

# Continue the *same* experiment later:
python Expansformer_Train_Kit/run.py train --project . \
  --allow-unreviewed-experiment --steps 1000 --stop-after 250

# Generate text after the 100-step run:
python train_expansformer_v04.py generate \
  --checkpoint ./runs/v04_integrated_pilot/best.pt \
  --prompt 'Scientific reasoning requires' --max-new-tokens 100 --temperature 0.7
```

By default this kit copies **only** staged Wikipedia pages with recorded CC BY-SA 4.0 URLs and PMC research articles with local JATS declarations of plain CC BY. It excludes the known NC/ND variants and all other external-source records pending separate review. The expected **approximate** selection on the latest local corpus is 210 Wikipedia + 88 PMC = **298**, but actual numbers depend on the local manifests and XML. A v0.3 archive without the newer human batches will instead select only the present eligible subset.

**Training is experimental and unreviewed.** These source-declaration checks are not independent rights verification, source-quality review, or canonical training admission. The tool never modifies `expansformer_dataset/` or marks a source approved. Don't redistribute an experimental text corpus or trained model as rights-cleared based on these checks.

For training on your prior small pilot corpus without candidate-corpus copying, you can use `train_expansformer_v04.py train --data ../Expansformer_v02_Research/corpus_wiki` directly after installation, with the v0.4 flags shown in the `run.py` training command.

The v0.2 checkpoint **cannot** be resumed as a v0.4 model; new neural parameters require training a new model. This kit reuses v0.3 utilities, but does not claim to import the other four original projects' full runtime behavior into transformer weights.
