# MP1 code — installation and usage

Read [the project guide](../GUIDE.md) for the assignment, assessment, deadlines and peer review. This README contains the running instructions and technical rules. The student's experimental analysis is in [REPORT.md](REPORT.md), with a [three-page PDF version](REPORT.pdf).

All commands below run from **code/**. Data and the tokenizer are included. No API key, pretrained weights or additional dataset download is needed; after installing dependencies, training and evaluation work offline.

## Selected student model and direct reproduction

The selected model is `student.py` with RoPE, four blocks, four heads, width 192, context 256, and the configuration in `configs/rope_width192.json`. Its frozen checkpoint produces **1.6653628497818085 full-test BPB** with the supplied CPU FP32 evaluator. The checkpoint SHA-256 is `aa35c0455a95d5cc4cbe124b4e57ec257b96012b056160a60c784764ec03bff1`; the matching `student.py` SHA-256 is `a90a2d9ebd047e791425e350ce9a226bf5a9a1ed3b3e29227d0893ae94f0e328`.

After the installation steps below, download the checkpoint from the final course submission and run this command from `code/`. Training is **not** required to reproduce the score:

```bash
python evaluate.py --checkpoint /path/to/downloaded/checkpoint.pt --device cpu --precision fp32 --split test
```

To repeat the training run from random initialization, use a new empty output directory:

```bash
python train.py --implementation student --config configs/rope_width192.json --device cpu --precision fp32 --threads 4 --seed 17 --steps 4800 --batch-size 32 --eval-every 1200 --run-dir runs/reproduced-rope-w192-s17
```

The selected run processed 39,321,600 training targets in about 2,246 CPU seconds on the student's Mac. Validation BPB was 1.642035. The exact test score and run metadata are in `runs/rope-w192-4800-s17/test_cpu_fp32.json` and `metrics.json` locally; `runs/` is ignored by Git, so the checkpoint must be supplied separately. `PACKAGE_MANIFEST.json` records hashes of the **original starter package**, so hashes for intentionally edited `student.py` and this README differ. The final checkpoint and source hashes above identify the submitted predictor.

## 1. Install

Use **Python 3.12**. From the extracted package directory:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install PyTorch for **one** device:

```bash
# Linux/Windows CPU: recommended; no GPU needed
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a compatible driver, use this command **instead**:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

For macOS, install `torch==2.7.1` from the default PyPI index and run on CPU. After installing PyTorch, install the remaining dependencies and check the model:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Linux CPU commands were verified with Python 3.12 and PyTorch 2.7.1+cpu. Windows/macOS timings have not been measured.

## 2. Train and evaluate

**Quick installation check** — 10 training steps, then full-test evaluation:

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/checkpoint.pt --split test
```

This checks that the pipeline works; its score is **not** the full baseline. Each training run needs a new output directory.

**Full baseline** — 1,200 training updates, then evaluation:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

The baseline has four GPT blocks, width 128, four attention heads and **1,088,256 parameters**, and achieves approximately **2.10 test BPB**. On the reference four-thread Xeon Platinum 8457C, measured training took about **311 seconds** and scoring **5.92 seconds**, excluding installation and loading. These are reference measurements, not laptop guarantees or a fixed time allowance.

**Your model** — edit `student.py` and supporting files, then:

```bash
python train.py --implementation student --seed 17 --eval-every 300 --run-dir runs/my-model
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split validation
# Freeze the final method before testing:
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split test
```

Training writes `checkpoint.pt` and `metrics.json`. Evaluation writes `test_cpu_fp32.json` (or the corresponding device/split name) and per-window losses. Submit the **bpb** value from the complete-test JSON, not token perplexity or validation BPB. Default evaluation is FP32. Add `--device cuda` for GPU runs; training can use BF16, but ranked evaluation must use FP32 and remain reproducible on CPU. The supplied CUDA runner caps PyTorch allocation at 20 GB; driver overhead is additional.

## 3. Files and model interface

| Files | Use |
|---|---|
| `model.py`, `configs/baseline.json` | Runnable baseline; preserve for comparisons. |
| `student.py`, `train.py` | Your model factory and training recipe; add supporting code as needed. |
| `common.py`, `evaluate.py` | Fixed data checks, windows and scorer; keep unchanged. |
| `data/` | Supplied splits, tokenizer and dataset hashes; keep unchanged. |
| `tests/test_contract.py` | Checks your model's causality, normalization, independence and gradients. |
| `RUN_LOG_TEMPLATE.csv` | Optional experiment-log template. |
| `PACKAGE_MANIFEST.json` | Release hashes; paths are relative to the package root containing code/ and guide/. |

- `build_model(config)` returns a PyTorch model with `context=256`.
- The supplied trainer calls `forward(ids)` for unnormalized logits; the scorer calls `predict_log_probs(ids)` for finite, normalized natural-log probabilities. Both outputs have shape `[batch, time, 2048]`.
- A prediction at position t may use only the observed prefix through t. Reset temporary state between independent windows, examples and scoring passes. Compact training-derived assets may be reused across windows; evaluation-prefix state may not.
- Checkpoints record the implementation module and configuration. Include that module and every required asset so the evaluator can reconstruct the submitted predictor. No optimizer state is required for direct evaluation.
- Training length, architecture, optimizer, regularization, self-trained weight averaging and ensembles may change within the guide's constraints. Log all seeds, processed training targets, checkpoint ancestry and search costs; reusing a checkpoint does not erase its training cost. No particular seed or score improvement is mandated.

## 4. Benchmark and resource measurements

**Fixed score.** Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, train-fitted BPE-2048, independent windows of 256 targets, including the final short window. Every target except the first token of each split is scored once. Input windows share a boundary token but carry no state. BPB is summed negative log-base-2 next-token probability divided by the split's entire raw UTF-8 byte length, including the first token's bytes.

| Split | Scored targets | UTF-8 bytes |
|---|---:|---:|
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

Use validation for all development and checkpoint/mixture selection. Weights, statistics and retrieval entries must derive only from training text. The public test text enables reproduction; it must not be used to tune the method. Once frozen, the same predictor may be evaluated repeatedly for timing or reproduction. Token perplexity is not directly comparable with published word-level perplexity.

Measure all three limits for the same frozen predictor. Local measurements for the selected width-192 checkpoint are:

- **CPU time ≤5× baseline:** 9.44 seconds for full-test CPU FP32 scoring versus 6.68 seconds for the local baseline, about 1.41×.
- **Peak RAM ≤4 GiB:** about 1.42 GiB peak resident memory during full-test scoring on macOS.
- **Inference assets ≤64 MiB uncompressed:** the checkpoint is 8,709,749 bytes (8.31 MiB); no additional model inference assets are required.

## 5. Prepare your submission and reproduce a peer

The [guide](../GUIDE.md) specifies the deadline and website workflow. Include the following in your immutable code repository:

- **Report, at most 10 pages including figures, tables and references** 
- **Reproduction instructions**

Your final website submission must link to this code and the matching complete checkpoint bundle. The website generates the Issue JSON automatically. Keep all inference assets downloadable for verification.

To check a peer, obtain their exact code version and checkpoint, follow their installation instructions, and run their frozen model with the supplied evaluator:

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

Compare reproduced BPB with the reported score. Submit **Peer Review Report** with the reproduced score; optionally include the command, environment, difference and evidence/log link.  The instructor adjudicates discrepancies. Confirmed discrepancies during the seven-day review earn bonus credit under the announced marking policy.

## Student work, reused sources, and AI assistance

The course supplied `model.py`, `train.py`, `common.py`, `evaluate.py`, the tests, data, tokenizer, and initial configuration. The student's `student.py` adapts the rotary-position idea from [Su et al., *RoFormer*](https://arxiv.org/abs/2104.09864) to that baseline; the RoPE idea is credited to its authors. The original data notices are retained below.

**Substantive AI assistance disclosure:** OpenAI Codex helped design and implement the RoPE variant, suggest and run the width-192 comparison, and inspect correctness and resource measurements. It also assisted with writing `REPORT.md` and these README additions. No pretrained weights or external training text were used.

## 6. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These dataset notices do not assign a new license to the surrounding classroom code.
