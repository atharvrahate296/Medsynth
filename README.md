# Medsynth: Synthetic Dermatological Image Studio

Medsynth fine-tunes Stable Diffusion 1.5 with LoRA across three benchmark skin lesion datasets and provides a dedicated, high-contrast **Streamlit** clinical studio for generating synthetic dermatological images. Training and inference are fully decoupled: `train.py` prepares the datasets, trains, and exports the standalone pipeline; `main.py` runs the Streamlit UI to synthesize images and manage clinical conversation histories.

> **Already have a trained model?** Keep `medsynth-model/pipeline/`, install the dependencies, and run `streamlit run main.py`. You do not need the source training datasets for image generation.

---

## Contents

- [At a Glance](#at-a-glance)
- [The Three Modalities & Datasets](#the-three-modalities--datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Streamlit Studio](#running-the-streamlit-studio)
- [Studio Features & UI Guide](#studio-features--ui-guide)
  - [Chat & Thread Management](#chat--thread-management)
  - [Clinical Modalities & Grammar](#clinical-modalities--grammar)
  - [CLIP Token Budget Meter](#clip-token-budget-meter)
  - [Fine-Tuning Parameter Controls](#fine-tuning-parameter-controls)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Training Configuration](#training-configuration)
- [Quality Checks](#quality-checks)
- [Troubleshooting](#troubleshooting)
- [Responsible & Ethical Use](#responsible--ethical-use)

---

## At a Glance

| Task | Entry Point | Requirements |
|---|---|---|
| **Generate Images (Studio)** | `streamlit run main.py` | Exported `medsynth-model/pipeline/`, Python 3.10–3.12 |
| **Train or Resume Training** | `python train.py` | Source datasets, CUDA-capable NVIDIA GPU, internet access for base SD 1.5 |
| **Prepare Datasets** | Built into `train.py` | Place raw dataset archives into `data/raw/` |

```bash
# 1. Install dependencies
uv pip install -r requirements.txt
# or: pip install -r requirements.txt

# 2. Launch the Streamlit Studio
streamlit run main.py
```

---

## The Three Modalities & Datasets

A single unified model learns all three dermatological views. Each view is conditioned with its trained caption prefix:

| View Modality | Source Dataset | Caption Prefix | Example Clinical Prompt |
|---|---|---|---|
| **Dermoscopy** | **HAM10000** (~10k images, 7 diagnoses) | `dermoscopy image of` | `dermoscopy image of melanoma, malignant, back` |
| **Smartphone Close-Up** | **PAD-UFES-20** (~2.3k images, 7 diagnoses) | `smartphone photo of` | `smartphone photo of basal cell carcinoma, malignant, face, Fitzpatrick skin type 2` |
| **Clinical Photo** | **Fitzpatrick17k** (~16k images, 114 conditions) | `clinical photo of` | `clinical photo of psoriasis, non-neoplastic, Fitzpatrick skin type 5` |

---

## Project Structure

```
medsynth/
├── main.py                     <- Modern Streamlit clinical studio application
├── main_gradio_backup.py       <- Preserved backup of previous Gradio interface
├── train.py                    <- Dataset preparation, LoRA fine-tuning, and model export
├── requirements.txt            <- Dependencies with PyTorch CUDA --find-links
├── .streamlit/
│   └── config.toml             <- Clinical light theme configuration
├── utils/
│   ├── prompt_enhancer.py      <- Maps clinical free text onto trained caption grammar
│   └── chat_history.py         <- SQLite thread persistence, message log, and settings
├── data/                       <- Local dataset files (ignored by git; not needed for inference)
│   ├── raw/
│   │   ├── ham10000/           <- HAM10000 archives and metadata
│   │   ├── pad_ufes_20/        <- PAD-UFES-20 archives and metadata
│   │   └── fitzpatrick17k/     <- Fetched automatically by train.py
│   └── prepared/               <- Processed 512x512 images, captions, and 2% held-out set
├── medsynth-model/             <- Created by train.py
│   ├── pipeline/               <- Standalone diffusion pipeline loaded by main.py
│   ├── lora/                   <- Exported LoRA safetensors
│   ├── checkpoints/            <- Step-wise checkpoints (saved every 1,000 steps)
│   └── samples/                <- Multi-view sample images generated during training
└── results/                    <- Runtime conversation history and user exports
    ├── medsynth_history.sqlite3
    └── exports/                <- Markdown + PNG exports
```

---

## Installation

Medsynth requires Python 3.10–3.12. For training, an **NVIDIA GPU with CUDA** is required (8 GB VRAM minimum, 16 GB+ recommended). For inference in the Streamlit studio, an NVIDIA GPU provides near-instant diffusion sampling; CPU fallback is supported automatically if CUDA is absent.

```bash
# Create and activate virtual environment
python -m venv env

# Windows PowerShell:
.\env\Scripts\Activate.ps1
# Linux / macOS:
# source env/bin/activate

# Upgrade installer and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
# Alternatively, using uv:
# uv pip install -r requirements.txt

# Verify CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> **Note on `requirements.txt`**: The requirements specify `--find-links https://download.pytorch.org/whl/cu121` so that both standard `pip` and fast `uv pip` resolve PyTorch CUDA 12.1 binaries without index confusion or version mismatches against PyPI packages (`requests`, `streamlit`, etc.).

---

## Running the Streamlit Studio

To launch the Medsynth Studio, run:

```bash
streamlit run main.py
```

Or via Python module execution:
```bash
python -m streamlit run main.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### Custom Pipeline Location

If your trained model is located in a non-default directory, set `MEDSYNTH_MODEL_DIR`:

```powershell
# Windows PowerShell
$env:MEDSYNTH_MODEL_DIR = "D:\models\medsynth\pipeline"
streamlit run main.py
```

```bash
# Linux / macOS
MEDSYNTH_MODEL_DIR=/path/to/medsynth/pipeline streamlit run main.py
```

---

## Studio Features & UI Guide

The Medsynth Studio is engineered with a high-contrast **Clinical Light Theme** tailored for medical visualization:

### 1. Chat & Thread Management
* **`＋ New Conversation`**: Creates a fresh conversation thread with default generation parameters.
* **Search Filter**: A live search bar at the top of the sidebar filters threads in real time by title or text inside prior prompt messages.
* **Inline Renaming**: Edit the current thread's title directly in the sidebar under *Active Thread Management* and click **Save** (or press Enter). Changes persist immediately to SQLite and update the sidebar and header.
* **📌 Pin / Unpin**: Pin priority experiments so they remain at the top of your sidebar list.
* **📋 Copy / Duplicate**: Duplicate an entire thread—including prompt history, generated images, and generation settings—to branch off alternative parameter variations.
* **🧹 Clear Chat**: Erases all messages in the thread while preserving custom generation settings and the conversation slot.
* **🗑️ Safe Delete**: Two-step deletion (`🗑️ Delete` -> `⚠️ Confirm`) prevents accidental loss of clinical synthesis records.
* **📥 Export Thread**: Generates a self-contained folder under `results/exports/` containing a formatted Markdown summary and all synthesized PNG images.

### 2. Clinical Modalities & Grammar
Choose between three capture views using the top selector:
* **Dermoscopy**: Polarized illumination, 10x magnification, surface pigment networks.
* **Smartphone**: Handheld close-up framing, natural daylight depth of field.
* **Clinical Photo**: Standard clinical distance and even lighting.

**Clinical Prompt Enhancement (`Always On`)**:
The model was fine-tuned on short captions in a structured clinical grammar:
```
<view> of <diagnosis>, <category>, <body site>, <Fitzpatrick skin type>
```
Enhancement automatically parses natural free text (e.g. `"mole on left forearm of a fitzpatrick 3 patient"`), maps synonyms onto canonical dermatological terms (`melanocytic nevus, benign, arm, Fitzpatrick skin type 3`), and appends view-specific modality tokens (`polarized illumination`, `10x magnification`) while respecting the 75-token CLIP budget.

### 3. CLIP Token Budget Meter
CLIP text encoders enforce a strict 77-token ceiling (75 usable tokens). The studio includes a real-time **CLIP Budget Meter** with visual progress tracking:
* Safe zone: Green bar (< 70 tokens).
* Near limit: Orange warning bar (≥ 70 tokens). Long prompts are safely normalized and truncated without crashing the pipeline.

### 4. Fine-Tuning Parameter Controls
Expand the **Fine-Tune Parameters** drawer in the sidebar to adjust:
* **Inference Steps** (1–50, default: `30`): Higher step counts refine lesion borders and micro-textures.
* **Guidance Scale / CFG** (1.0–15.0, default: `6.5`): Adjusts adherence to clinical prompt keywords.
* **Seed & 🎲 Randomize**: Set a fixed seed to reproduce identical lesion patterns, or click the shuffle button to generate a new random seed.
* **Negative Prompt**: Pre-configured with a dermatological negative prompt suppressing CGI, plastic/waxy skin, blur, borders, and anatomical distortions.
* **🔄 Reset Defaults**: Reverts all sliders and negative prompts back to recommended baseline settings.

### 5. Synthesized Results & Download
Every generated image includes:
* High-resolution synthetic skin lesion rendering.
* Metadata caption detailing the modality, diagnosis, category, step count, CFG scale, seed used, and token count.
* **`📥 Download PNG`** button for instant saving.
* Medical research disclaimer note.

---

## Dataset Preparation

Training data is **not included** in the repository. Place dataset archives in the following folders:

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

Create folders on Windows PowerShell:
```powershell
New-Item -ItemType Directory -Force -Path "data\raw\ham10000", "data\raw\pad_ufes_20", "data\raw\fitzpatrick17k"
```

* **HAM10000**: Download from [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T) or [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). Place zip files into `data/raw/ham10000/`.
* **PAD-UFES-20**: Download from [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) or [Kaggle](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer). Place into `data/raw/pad_ufes_20/`.
* **Fitzpatrick17k**: Handled automatically when starting `train.py`.

---

## Model Training

To begin training:
```bash
python train.py
```

`train.py` automatically handles:
1. CUDA verification and mixed-precision configuration (bf16/fp16).
2. Archive extraction and image preprocessing (512x512 with safe augmentations).
3. 2% held-out test split creation under `data/prepared/heldout.jsonl`.
4. LoRA fine-tuning of Stable Diffusion 1.5 with auto-batched gradient accumulation.
5. Periodic sampling previews under `medsynth-model/samples/`.
6. Full standalone pipeline export into `medsynth-model/pipeline/`.

**Resuming**: If training is interrupted, re-running `python train.py` automatically detects and resumes from the latest checkpoint under `medsynth-model/checkpoints/`.

### GPU Memory Requirements

| VRAM | Batch Size × Accumulation | Gradient Checkpointing |
|---|---|---|
| 20 GB+ | 4 × 4 | Off |
| 14–20 GB | 2 × 8 | Off |
| 10–14 GB | 2 × 8 | On |
| Under 10 GB | 1 × 16 | On |

---

## Quality Checks

* Evaluate synthetic lesions against real images using the held-out dataset at `data/prepared/heldout.jsonl`.
* Compute FID (Fréchet Inception Distance) or train downstream diagnostic classifiers on synthetic sets to measure transfer accuracy.
* Inspect preview progressions across checkpoints in `medsynth-model/samples/` to monitor convergence.

---

## Troubleshooting

| Issue | Cause & Solution |
|---|---|
| `uv pip install` fails on `requests` version | Ensure `requirements.txt` uses `--find-links https://download.pytorch.org/whl/cu121` instead of `--extra-index-url`. |
| `CUDA is not available ... CPU-only build` | Reinstall PyTorch with CUDA 12.1 support matching your graphics driver. |
| `CUDA out of memory` during training | In `train.py`, set `auto_batch = False`, `batch_size = 1`, `grad_accum = 16`, and `gradient_checkpointing = True`. |
| Studio shows `Trained model not found` | Ensure `medsynth-model/pipeline/model_index.json` exists, or set `$env:MEDSYNTH_MODEL_DIR`. |
| Port 8501 is already in use | Run `streamlit run main.py --server.port 8502`. |

---

## Responsible & Ethical Use

* **Synthetic Data Only**: Outputs generated by Medsynth are artificial and must **never** be used as primary diagnostic criteria or represented as authentic patient photography.
* **Licensing & Research Use**: HAM10000 (CC BY-NC 4.0) and Fitzpatrick17k sources contain non-commercial restrictions. Treat exported pipeline weights as research and educational artifacts.
* **Demographic Diversity**: Datasets have historical representation skews. Always evaluate multi-tone representation across Fitzpatrick skin types I through VI when building downstream models.