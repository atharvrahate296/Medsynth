# Medsynth: Synthetic Dermatological Image Studio

Medsynth fine-tunes Stable Diffusion 1.5 with LoRA across three benchmark skin lesion datasets and ships a dedicated **Streamlit** clinical studio for generating synthetic dermatological images. Training and inference are fully decoupled: `train.py` prepares the datasets, trains, and exports the standalone pipeline; `main.py` runs the studio to synthesize images, tune generation parameters per thread, and manage clinical conversation histories in SQLite.

> **✅ Current project status:** Training is **complete** — the model finished a full 15,000-step run and the standalone pipeline is exported at `medsynth-model/pipeline/`. If you just want to generate images, skip straight to [Using the Studio](#using-the-studio): you only need the dependencies, not the source datasets. Re-running `python train.py` is safe — it detects the finished model and does nothing.

---

## Contents

- [At a Glance](#at-a-glance)
- [The Three Modalities & Datasets](#the-three-modalities--datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Using the Studio](#using-the-studio)
- [Studio Features & UI Guide](#studio-features--ui-guide)
  - [Sidebar: Threads, Search & Settings](#sidebar-threads-search--settings)
  - [Appearance & Hardware Status](#appearance--hardware-status)
  - [Clinical Modalities & Grammar](#clinical-modalities--grammar)
  - [Token-Safe Prompting](#token-safe-prompting)
  - [Parameters & Modifiers](#parameters--modifiers)
  - [Chat Cards, Metadata & Download](#chat-cards-metadata--download)
- [Training the Model](#training-the-model)
  - [Step 1 — Prerequisites](#step-1--prerequisites)
  - [Step 2 — Get the Datasets](#step-2--get-the-datasets)
  - [Step 3 — Run the Trainer](#step-3--run-the-trainer)
  - [What One Run Does](#what-one-run-does)
  - [Outputs You Get](#outputs-you-get)
  - [Resuming, Retraining & Smoke Test](#resuming-retraining--smoke-test)
  - [GPU Memory Requirements](#gpu-memory-requirements)
- [Training Configuration](#training-configuration)
- [Quality Checks](#quality-checks)
- [Troubleshooting](#troubleshooting)
- [Responsible & Ethical Use](#responsible--ethical-use)

---

## At a Glance

| Task | Entry Point | Requirements |
|---|---|---|
| **Generate images (Studio)** | `streamlit run main.py` | Exported `medsynth-model/pipeline/` (already present), Python 3.10–3.12 |
| **Train or resume training** | `python train.py` | Source datasets in `data/raw/`, CUDA-capable NVIDIA GPU, internet access for base SD 1.5 |
| **Prepare datasets** | Built into `train.py` | Place raw dataset archives into `data/raw/` (see [Step 2](#step-2--get-the-datasets)) |

```bash
# Install dependencies (once)
pip install -r requirements.txt

# Launch the studio (no dataset needed — model is already trained)
streamlit run main.py
```

---

## The Three Modalities & Datasets

A single unified model learns all three dermatological views. Each view is conditioned with its trained caption prefix, which the studio applies automatically from the **Image type** selector:

| View Modality | Source Dataset | Caption Prefix | Example Clinical Prompt |
|---|---|---|---|
| **Dermoscopy** | **HAM10000** (~10k images, 7 diagnoses) | `dermoscopy image of` | `dermoscopy image of melanoma, malignant, back` |
| **Smartphone Close-Up** | **PAD-UFES-20** (~2.3k images, 7 diagnoses) | `smartphone photo of` | `smartphone photo of basal cell carcinoma, malignant, face, Fitzpatrick skin type 2` |
| **Clinical Photo** | **Fitzpatrick17k** (~16k images, 114 conditions) | `clinical photo of` | `clinical photo of psoriasis, non-neoplastic, Fitzpatrick skin type 5` |

---

## Project Structure

```
medsynth/
├── main.py                     <- Streamlit clinical studio application
├── train.py                    <- Dataset preparation, LoRA fine-tuning, and model export
├── requirements.txt            <- Dependencies (PyTorch CUDA 12.1 via --extra-index-url)
├── README.md
├── .streamlit/
│   └── config.toml             <- Light clinical theme (teal accent) + server settings
├── utils/
│   ├── prompt_enhancer.py      <- Caption grammar, modality tails, CLIP token budget
│   └── chat_history.py         <- SQLite threads, message log, settings, Markdown export
├── medsynth-model/             <- Created by train.py (gitignored) — training is COMPLETE
│   ├── pipeline/               <- Standalone diffusion pipeline loaded by main.py ✅
│   ├── lora/                   <- Exported LoRA safetensors ✅
│   ├── samples/                <- Final validation samples (step 15000) ✅
│   ├── training_complete.json  <- Marker: 15,000 steps, finished 2026-10-03 ✅
│   ├── model_info.json         <- Base model, LoRA rank, steps, datasets
│   ├── train_config.json       <- Full config snapshot of the finished run
│   ├── train_log.csv           <- step / loss / lr history
│   ├── prompt_examples.txt     <- One example caption per dataset + class
│   ├── checkpoints/            <- lora-step-* snapshots (created during training only)
│   └── state/                  <- Accelerate resume state (created during training only)
├── data/                       <- Created by train.py on first run (gitignored; absent now)
│   ├── raw/
│   │   ├── ham10000/           <- HAM10000 archives and metadata
│   │   ├── pad_ufes_20/        <- PAD-UFES-20 archives and metadata
│   │   └── fitzpatrick17k/     <- Downloaded automatically by train.py
│   └── prepared/               <- Processed 512x512 images, captions, 2% held-out set
├── model/                      <- Legacy training output (gitignored; pipeline fallback location)
├── results/                    <- Runtime data (gitignored)
│   ├── medsynth_history.sqlite3   <- Threads, messages, per-thread settings ✅
│   └── exports/<thread>-<stamp>/ <- Created on first export: Markdown summary + PNGs
├── main_gradio_backup.py       <- Preserved Gradio interface (gitignored, local only)
└── video-assets/               <- Demo recording scripts (gitignored, local only)
```

---

## Installation

Medsynth requires **Python 3.10–3.12** (PyTorch 2.2.2 ships no wheels for 3.13+).

* **For using the studio:** an NVIDIA GPU gives near-instant diffusion sampling; CPU is supported automatically if CUDA is absent (slower).
* **For training:** an NVIDIA GPU with CUDA is **required** — auto-batching works down to roughly 7 GB VRAM, with 10 GB+ comfortable and 16 GB+ recommended.

### 1. Create and activate a virtual environment

```bash
python -m venv env
```

```powershell
# Windows PowerShell
.\env\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source env/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
# Alternatively, using uv:
# uv pip install -r requirements.txt
```

### 3. Verify the install

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected: `CUDA available: True` plus your GPU name (or `CPU` if you only plan to run the studio on CPU).

> **Note on `requirements.txt`:** it uses `--index-url https://pypi.org/simple` together with `--extra-index-url https://download.pytorch.org/whl/cu121`, so both standard `pip` and fast `uv pip` resolve PyTorch CUDA 12.1 binaries (`torch==2.2.2+cu121`) alongside regular PyPI packages (`streamlit`, `requests`, etc.).

---

## Using the Studio

The trained pipeline already ships in `medsynth-model/pipeline/`, so no datasets and no training run are needed.

### Start the app

```bash
streamlit run main.py
```

or equivalently:

```bash
python -m streamlit run main.py
```

The studio opens automatically at `http://localhost:8501`.

### Generate your first image

1. **Wait for the header pill** in the sidebar to show `GPU Ready · <your GPU> · FP16` (or `CPU Mode · FP32 (slower)`). The pipeline loads once, on first generation, and is cached afterwards.
2. **Pick the capture view** with the **Image type** selector at the top-right of the studio: `Dermoscopy`, `Smartphone`, or `Clinical photo`. The choice is saved per conversation.
3. **Describe the lesion** in the composer bar at the bottom — free text is fine, e.g. *"mole on the back of a fitzpatrick 3 patient"*. The enhancer maps it onto the clinical caption grammar the model was trained on.
4. **Click `✨ Generate`** (or press Enter). A spinner shows while diffusion sampling runs.
5. **Read the result card**: it shows the enhanced caption, steps, CFG scale, seed, and CLIP token count. Use **`📥 Download PNG`** to save the untouched full-resolution file.
6. **Reuse a seed** by typing it into `Seed` under `🎛️ Parameters & Modifiers` — the same seed + prompt reproduces the identical image.

Everything (prompts, images, settings) is persisted per conversation in `results/medsynth_history.sqlite3`, so closing the browser loses nothing.

### Custom pipeline location

If your trained model lives somewhere else, set `MEDSYNTH_MODEL_DIR` to the folder containing `model_index.json`:

```powershell
# Windows PowerShell
$env:MEDSYNTH_MODEL_DIR = "D:\models\medsynth\pipeline"
streamlit run main.py
```

```bash
# Linux / macOS
MEDSYNTH_MODEL_DIR=/path/to/medsynth/pipeline streamlit run main.py
```

Without the variable, the studio resolves the pipeline in order: `medsynth-model/pipeline/` → `model/pipeline/`.

### How the studio loads the model

* The pipeline is loaded **once** and cached across reruns (`st.cache_resource`), with `HF_HUB_OFFLINE=1` so inference never contacts the Hugging Face Hub.
* CUDA runs in `float16` with the DPM++ Karras scheduler, channels-last memory format, and TF32 enabled; the safety checker is disabled (outputs are synthetic research artifacts — see [Responsible & Ethical Use](#responsible--ethical-use)).
* If moving the pipeline to the GPU fails (e.g. VRAM exhausted), the studio degrades gracefully: attention slicing → model CPU offload → plain CPU execution.
* Generations are serialized behind a lock, so only one diffusion run happens at a time.

---

## Studio Features & UI Guide

### Sidebar: Threads, Search & Settings

* **`＋ New Conversation`**: Creates a fresh thread with the default generation parameters.
* **Search conversations**: A live filter that matches thread titles **and** any text inside their messages, so old prompts stay findable. Pinned threads sort to the top; each thread button shows its relative last-active time on hover.
* **`⚙️ Manage & Rename Thread`** expander:
  * **Inline renaming**: edit the title and click **Save** — persisted to SQLite immediately.
  * **`📌 Pin` / `📍 Unpin`**: keep priority experiments at the top of the list.
  * **`🧹 Clear Chat`**: erases all messages and resets the title, while preserving the thread and its generation settings.
  * **`📋 Duplicate`**: clones the thread — prompt history, images, and settings — to branch parameter variations.
  * **`📥 Export`**: writes a self-contained folder to `results/exports/<thread>-<stamp>/` with a Markdown summary and all synthesized PNGs.
  * **`🗑️ Delete Thread` → `⚠️ Confirm Delete`**: two-step deletion prevents accidental loss of clinical synthesis records.

### Appearance & Hardware Status

* **Dark mode** toggle at the top of the sidebar switches the whole studio between the light clinical theme and a dark theme.
* A **GPU / CPU status pill** shows at a glance which device diffusion sampling will use.

### Clinical Modalities & Grammar

Pick the capture view with the **Image type** selector at the top-right of the studio (`Dermoscopy`, `Smartphone`, `Clinical photo`). The selection is stored per thread and prefixes every generated caption accordingly:

* **Dermoscopy**: polarized illumination, 10x magnification, surface pigment networks.
* **Smartphone**: handheld close-up framing, natural daylight depth of field.
* **Clinical Photo**: standard clinical distance and even lighting.

**Clinical Prompt Enhancement (always on)**:
The model was fine-tuned on short captions in a structured clinical grammar:
```
<view> of <diagnosis>, <category>, <body site>, <Fitzpatrick skin type>
```
Enhancement parses natural free text (e.g. `"mole on left forearm of a fitzpatrick 3 patient"`), maps synonyms onto canonical dermatological terms (`melanocytic nevus, benign, arm, Fitzpatrick skin type 3`), and appends view-specific modality tokens (`polarized illumination`, `10x magnification`) — all inside CLIP's token budget.

### Token-Safe Prompting

CLIP's text encoder accepts at most 77 tokens (75 usable). Instead of letting long prompts crash or silently truncate, `utils/prompt_enhancer.py` builds a **token-safe plan** before every generation:

* The enhanced caption is guaranteed to fit the budget; the modality tail is dropped first when space is tight.
* Each generated image's metadata records the final **token count** and whether the plan was **`trimmed`** or sent **verbatim**, so you can see exactly what reached the model.

### Parameters & Modifiers

Expand **`🎛️ Parameters & Modifiers`** under *Generation Settings* in the sidebar. Changes auto-save to the active thread:

* **Inference Steps** (1–50, default `30`): higher counts refine lesion borders and micro-textures.
* **Guidance Scale / CFG** (1.0–15.0, step 0.5, default `6.5`): adherence to clinical prompt keywords.
* **Seed** (`-1` = random, default) with a **`🎲`** shuffle button: fix a seed to reproduce identical lesions.
* **Negative Prompt**: pre-loaded with a dermatological negative prompt suppressing CGI, plastic/waxy skin, blur, distorted anatomy, watermarks, and non-skin imagery.
* **`🔄 Reset Defaults`**: reverts the thread's settings to the baseline.

### Chat Cards, Metadata & Download

* User prompts appear as **`Prompt:`** bubbles on the right; each assistant reply renders in a bordered card.
* Every generated image shows a **compact, framed preview** (rounded corners, capped display width — the full-resolution PNG is never shown inline) in a narrow left column.
* The right column — vertically centered against the image — holds:
  * the **metadata caption**: modality, enhanced caption, step count, CFG scale, seed, and token count;
  * the **`📥 Download PNG`** button, which always serves the untouched original file;
  * the attribution line.
* A standing disclaimer sits below the composer: *"Synthetic images are for research and education, not clinical diagnosis."*

---

## Training the Model

> **You do not need to train to use this project.** A finished 15,000-step model already exists in `medsynth-model/`. Train only if you want to retrain from scratch, change hyperparameters, or reproduce the run.

Training is a three-step process: **prerequisites → datasets → one command**.

### Step 1 — Prerequisites

* Python 3.10–3.12 in an activated virtual environment with `requirements.txt` installed ([Installation](#installation)).
* An NVIDIA GPU with working CUDA. `train.py` checks this *first* and exits with a fix-it message if CUDA is missing — it will never silently train on CPU.
* Internet access: the base model (`stable-diffusion-v1-5/stable-diffusion-v1-5`) is downloaded from the Hugging Face Hub on the first run, and Fitzpatrick17k images are fetched automatically.

Sanity check before you commit to a long run:

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA missing'; print(torch.cuda.get_device_name(0))"
```

### Step 2 — Get the Datasets

Training data is **not included** in the repository. Create the folders and drop the downloads in:

```
data/
└── raw/
    ├── ham10000/
    │   ├── HAM10000_metadata.csv
    │   ├── HAM10000_images_part_1.zip     (or extracted images)
    │   └── HAM10000_images_part_2.zip     (or extracted images)
    ├── pad_ufes_20/
    │   ├── metadata.csv
    │   └── imgs_part_1.zip                (or extracted images)
    └── fitzpatrick17k/
        ├── fitzpatrick17k.csv             (downloaded automatically by train.py)
        └── images/                        (downloaded automatically by train.py)
```

Create the folders on Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path "data\raw\ham10000", "data\raw\pad_ufes_20", "data\raw\fitzpatrick17k"
```

or on Linux / macOS:

```bash
mkdir -p data/raw/ham10000 data/raw/pad_ufes_20 data/raw/fitzpatrick17k
```

| Dataset | What to download | Where to put it |
|---|---|---|
| **HAM10000** | [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T) or [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) — both zip parts + metadata CSV | `data/raw/ham10000/` |
| **PAD-UFES-20** | [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) or [Kaggle](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer) — images + `metadata.csv` | `data/raw/pad_ufes_20/` |
| **Fitzpatrick17k** | Nothing — CSV and images are downloaded automatically on the first run | created by `train.py` |

Loose files dropped directly into `data/` are adopted and moved into the right sub-folder automatically. Missing datasets stop the run with per-dataset instructions (set `require_all_datasets = False` in `train.py`'s `Config` to train on a subset instead).

### Step 3 — Run the Trainer

```bash
python train.py
```

That single command does everything, in order — each stage is skipped automatically if it was already completed:

1. **CUDA check** — fails fast with a fix if no usable GPU is found.
2. **Tidy files** — moves loose dataset files into `data/raw/<dataset>/`.
3. **Download Fitzpatrick17k** — CSV first, then images (dead links are logged, not retried).
4. **Prepare data** — builds captions in the clinical grammar, balances classes, resizes to 512×512, writes `metadata.jsonl` and a 2% held-out split (`heldout.jsonl`).
5. **Train** — LoRA fine-tuning with auto-batched gradient accumulation; resumes automatically after a crash or `Ctrl+C`.
6. **Export** — merges the LoRA into a standalone fp16 pipeline at `medsynth-model/pipeline/` and writes `training_complete.json`.

On a healthy run you will see lines like:

```
CUDA GPU: <name> | 12.0 GB VRAM | torch 2.2.2 (CUDA 12.1)
Auto batch settings for 12 GB: batch 2 x accumulation 8 = 16 images/step, gradient checkpointing on
Prepared 26xxx training images (5xx held out) in .../data/prepared
Resuming from lora-step-1000 (step 1000 of 15000)
training:  13%|...| 2000/15000 ...
```

When it finishes:

```
Training finished at step 15000. LoRA saved to .../medsynth-model/lora
Merging .../medsynth-model/lora into stable-diffusion-v1-5/stable-diffusion-v1-5 ...
Done. Standalone model saved to: .../medsynth-model/pipeline
```

**Leave the terminal open for the whole run.** Training writes a checkpoint every 1,000 steps, so even a crash or `Ctrl+C` loses at most ~1,000 steps — just re-run `python train.py` to continue.

### What One Run Does

| Phase | Details |
|---|---|
| Data preparation | Archives extracted once (marker files prevent re-extraction), view-safe augmentations applied (dermoscopy gets flips + rotations; real photos keep gravity) |
| Balancing | Per-class caps (1,200 for HAM10000/PAD-UFES-20, 600 for Fitzpatrick17k) and weighted sampling so small datasets aren't drowned out (shares: 35% / 25% / 40%) |
| Training | Effective batch 16 images/step, LoRA rank 32 on the UNet attention layers, cosine LR 1e-4 with 500 warmup steps, Min-SNR γ=5 loss weighting, caption dropout 0.1 |
| Monitoring | Loss logged to `medsynth-model/train_log.csv` every 50 steps; a validation sample per view written to `medsynth-model/samples/` every 1,000 steps |
| Checkpointing | Full resume state + LoRA snapshot every 1,000 steps into `medsynth-model/state/` and `medsynth-model/checkpoints/` (last 2 states kept) |
| Export | LoRA fused into the base model, saved as fp16 safetensors to `medsynth-model/pipeline/`, plus `prompt_examples.txt` copied alongside |

### Outputs You Get

After a successful run, `medsynth-model/` contains:

| Path | What it is |
|---|---|
| `pipeline/` | **The model the studio loads** — standalone, no network needed |
| `lora/pytorch_lora_weights.safetensors` | The exported LoRA weights alone (reusable on other SD 1.5 pipelines) |
| `samples/step-*.png` | Validation renders per view — visual progress checkpoints |
| `training_complete.json` | Marker file; its presence makes `python train.py` a no-op |
| `model_info.json` | Base model, pipeline path, steps, LoRA rank, datasets |
| `train_config.json` | Full config snapshot of the run (for reproducibility) |
| `train_log.csv` | `step,loss,lr` history for plotting convergence |
| `checkpoints/lora-step-*` | Intermediate LoRA snapshots (during training) |
| `state/step-*` | Accelerate resume state (during training) |

### Resuming, Retraining & Smoke Test

| Goal | What to do |
|---|---|
| **Continue an interrupted run** | Just run `python train.py` again — it finds the latest checkpoint and resumes from that step. |
| **Do nothing (model finished)** | Run `python train.py` — it prints the completion message and exits. You can never retrain by accident. |
| **Start over from scratch** | Set `retrain = True` in the `Config` class at the top of `train.py` (deletes checkpoints and the completion marker, then trains fresh). |
| **Quick end-to-end check (~10 min)** | Set `smoke_test = True` in `Config`. Runs 40 steps on tiny subsets into `data/prepared_smoketest/` and `medsynth-smoketest/` — your real data and model output are untouched. Set it back to `False` afterwards. |
| **Change hyperparameters** | Edit the `Config` class at the top of `train.py` (steps, LR, rank, batch…), see [Training Configuration](#training-configuration). |
| **Use the model afterwards** | `streamlit run main.py` — no further conversion needed. |

### GPU Memory Requirements

`auto_batch = True` (default) keeps the effective batch fixed at **16 images per optimizer step** and splits it to fit your GPU:

| VRAM | Batch Size × Accumulation | Gradient Checkpointing |
|---|---|---|
| 20 GB+ | 4 × 4 | Off |
| 14–20 GB | 2 × 8 | Off |
| 10–14 GB | 2 × 8 | On |
| Under 10 GB | 1 × 16 | On |

Below ~7 GB the trainer prints a warning; expect out-of-memory errors on very small GPUs.

---

## Training Configuration

Defaults live in the `Config` class at the top of `train.py`:

| Setting | Default | Notes |
|---|---|---|
| Resolution | `512` | Matches SD 1.5's native training resolution |
| Max train steps | `15000` | Resumes automatically from checkpoints |
| Effective batch | `16` | Split into batch × accumulation by VRAM |
| Learning rate | `1e-4` | Cosine schedule, 500 warmup steps |
| LoRA | rank `32`, alpha `32` | Caption dropout `0.1` |
| Loss weighting | Min-SNR `γ = 5.0` | `0` disables it |
| Mixed precision | `auto` | bf16/fp16 selected from the GPU |
| Checkpoints | every `1000` steps | Keeps the last 2 |
| Validation samples | every `1000` steps | 25 inference steps per view |
| Holdout | `2%` | `data/prepared/heldout.jsonl` |
| Seed | `42` | Reproducible data ordering |

---

## Quality Checks

* Evaluate synthetic lesions against real images using the held-out dataset at `data/prepared/heldout.jsonl`.
* Compute FID (Fréchet Inception Distance) or train downstream diagnostic classifiers on synthetic sets to measure transfer accuracy.
* Inspect preview progressions across checkpoints in `medsynth-model/samples/` to monitor convergence.

---

## Troubleshooting

| Issue | Cause & Solution |
|---|---|
| `uv pip install` fails on `requests` version | Ensure `requirements.txt` keeps `--index-url https://pypi.org/simple` **and** `--extra-index-url https://download.pytorch.org/whl/cu121` — mixing them up makes uv resolve `requests` against the PyTorch index. |
| `CUDA is not available ... CPU-only build` | Reinstall PyTorch with CUDA 12.1 support matching your graphics driver: `pip install torch==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121`. |
| `CUDA out of memory` during training | In `train.py`, set `auto_batch = False`, `batch_size = 1`, `grad_accum = 16`, and `gradient_checkpointing = True`. |
| `Cannot start training: some datasets are not ready` | Follow the per-dataset instructions printed in the error — check the folder layout in [Step 2](#step-2--get-the-datasets). |
| Trainer exits immediately with "A trained model already exists" | Expected: training is complete. Use `streamlit run main.py`, or set `retrain = True` to train again. |
| Studio shows `Trained model not found` | Ensure `medsynth-model/pipeline/model_index.json` exists, or set `$env:MEDSYNTH_MODEL_DIR`. |
| **`Connection error / Is Streamlit still running?` during generation** | The Streamlit server process died — usually RAM/VRAM exhaustion while loading the pipeline (this shows up as hard crashes in Windows Event Log, not a Python traceback). Close memory-heavy apps (browsers, IDEs) and retry; on CUDA OOM the studio automatically falls back to attention slicing → model CPU offload → CPU. |
| Generation is very slow | No NVIDIA GPU is available, so sampling runs on CPU. A CUDA GPU makes each image near-instant. |
| Port 8501 is already in use | Run `streamlit run main.py --server.port 8502`. |

---

## Responsible & Ethical Use

* **Synthetic Data Only**: Outputs generated by Medsynth are artificial and must **never** be used as primary diagnostic criteria or represented as authentic patient photography.
* **Licensing & Research Use**: HAM10000 (CC BY-NC 4.0) and Fitzpatrick17k sources contain non-commercial restrictions. Treat exported pipeline weights as research and educational artifacts.
* **Demographic Diversity**: Datasets have historical representation skews. Always evaluate multi-tone representation across Fitzpatrick skin types I through VI when building downstream models.
