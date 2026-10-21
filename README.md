# Medsynth: Synthetic medical image generator

Medsynth fine-tunes Stable Diffusion 1.5 with LoRA on three public skin-image datasets and
provides a local Gradio app for generating synthetic images. Training and inference are
separate: `train.py` prepares the datasets, trains, and exports the model; `main.py` only loads
the exported local pipeline and generates images.

> **Already have a trained model?** Keep `medsynth-model/pipeline/`, install the dependencies,
> and run `python main.py`. You do not need the training datasets for image generation.

## Contents

- [Project structure](#project-structure)
- [Install](#install)
- [Prepare the datasets](#prepare-the-datasets)
- [Train the model](#train-the-model)
- [Run the app](#run-the-app)
- [Writing prompts](#writing-prompts)
- [Clinical prompt enhancement](#clinical-prompt-enhancement)
- [Conversations](#conversations)
- [Quality checks](#quality-checks)
- [Troubleshooting](#troubleshooting)
- [Responsible use](#responsible-use)

## At a glance

| Task | Entry point | Needs |
|---|---|---|
| Generate images | `python main.py` | Exported `medsynth-model/pipeline/` |
| Train or resume training | `python train.py` | Dataset files, CUDA-capable NVIDIA GPU, internet for the base model |
| Prepare datasets | Included in `train.py` | Place source datasets as described below |

```bash
# Train once (requires an NVIDIA GPU with CUDA and the training datasets):
python train.py

# Run the app after training:
python main.py
```

## The three datasets = three imaging views

One model learns all three views. Each view has its own caption prefix, so you can ask for
the one you want.

| View | Dataset | Caption prefix | Example caption |
|---|---|---|---|
| Dermoscopy | **HAM10000** (~10k images, 7 diagnoses) | `dermoscopy image of` | `dermoscopy image of melanoma, malignant, back` |
| Smartphone close-up | **PAD-UFES-20** (~2.3k images, 7 diagnoses) | `smartphone photo of` | `smartphone photo of basal cell carcinoma, malignant, face, Fitzpatrick skin type 2` |
| Clinical photo | **Fitzpatrick17k** (~16k images, 114 conditions) | `clinical photo of` | `clinical photo of psoriasis, non-neoplastic, Fitzpatrick skin type 5` |

HAM10000 is the curated subset of the ISIC archive. The full ISIC archive is much larger and
needs its own downloader.

---

## Project structure

```
medsynth/
├── main.py
├── train.py                  <- dataset preparation, training, and model export
├── requirements.txt
├── utils/
│   ├── prompt_enhancer.py    <- maps free text to the training caption grammar
│   └── chat_history.py       <- conversations, images and settings (SQLite)
├── data/                     <- local training data; not needed for inference
│   ├── raw/                  <- source datasets go here
│   │   ├── ham10000/
│   │   ├── pad_ufes_20/
│   │   └── fitzpatrick17k/    <- populated automatically for Fitzpatrick17k
│   └── prepared/             <- generated images, captions, and holdout metadata
├── medsynth-model/           <- created by train.py
│   ├── pipeline/             <- standalone model loaded by main.py
│   ├── lora/ checkpoints/ state/ samples/  <- training artifacts
│   └── model_info.json train_config.json train_log.csv prompt_examples.txt
└── results/                  <- created by the app for conversation history
```

`train.py` is the supported setup and training entry point. It includes dataset organization,
archive extraction, image processing, and caption generation; there is no separate preprocessing
script to run. Dataset files and generated model/data directories are local working files, not
application source code.

## Install

Use Python 3.10–3.12. Training requires an **NVIDIA GPU with CUDA** (8 GB VRAM minimum,
16 GB+ comfortable) and about 30 GB of free disk for the base model, datasets, and training
outputs. Inference uses the exported pipeline and can run without the base-model cache; a GPU
is recommended for generation speed.

```bash
python -m venv .venv
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Check whether CUDA is available; training requires this to print True.
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

The requirements pin a CUDA 12.1 PyTorch build. If you need a different PyTorch/CUDA build,
install the PyTorch version matching your driver and adjust the PyTorch pin and package index
in `requirements.txt`. `train.py` checks CUDA itself and refuses to train on CPU. On the first
training run, the base model
`stable-diffusion-v1-5/stable-diffusion-v1-5` downloads from Hugging Face (no token required).

## Prepare the datasets

Training data is **not included** in the repository. Download the datasets from their official
sources and place them in the matching folders below. From the project root, the expected layout
is:

```
data/
└── raw/
    ├── ham10000/
    │   ├── HAM10000_metadata.csv
    │   ├── HAM10000_images_part_1.zip     (or extracted image files)
    │   └── HAM10000_images_part_2.zip     (or extracted image files)
    ├── pad_ufes_20/
    │   ├── metadata.csv
    │   ├── imgs_part_1.zip                (or extracted image files)
    │   └── ...                            (other image archives/files)
    └── fitzpatrick17k/
        ├── fitzpatrick17k.csv             (downloaded automatically)
        ├── download_failures.tsv          (created by the downloader)
        └── images/                        (downloaded automatically)
```

Create the dataset directories before copying files. For example, on Windows PowerShell, run
these commands from the Medsynth project folder:

```powershell
New-Item -ItemType Directory -Force -Path `
  "data\raw\ham10000", `
  "data\raw\pad_ufes_20", `
  "data\raw\fitzpatrick17k"
```

On macOS or Linux:

```bash
mkdir -p data/raw/ham10000 data/raw/pad_ufes_20 data/raw/fitzpatrick17k
```

### HAM10000 — dermoscopy

1. Download **HAM10000_images_part_1.zip**, **HAM10000_images_part_2.zip**, and the metadata
   file from the [Harvard Dataverse dataset](https://doi.org/10.7910/DVN/DBW86T) or its
   [Kaggle mirror](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
2. Put both image archives and the metadata file inside `data/raw/ham10000/`. The metadata may
   be named `HAM10000_metadata.csv` or `HAM10000_metadata`.
3. Leave the archives zipped or extract them there. The trainer searches recursively for supported
   image files and extracts zip archives during preparation.

This dataset is licensed CC BY-NC 4.0; check its terms before use or redistribution.

### PAD-UFES-20 — smartphone photos

1. Download the image archives and `metadata.csv` from
   [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) or the
   [Kaggle mirror](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer).
2. Put `metadata.csv` and all the image archives/files inside `data/raw/pad_ufes_20/`.
3. Leave archives zipped or extract them into this folder. The trainer finds image files inside
   nested folders as well.

### Fitzpatrick17k — clinical photos

No manual image download is normally needed. When you run `python train.py`, the trainer creates
`data/raw/fitzpatrick17k/`, downloads `fitzpatrick17k.csv`, and attempts to fetch the linked
images into its `images/` subfolder. Internet access is required. Some source links are no longer
available; failed links are listed in `download_failures.tsv` and are skipped on subsequent runs.
Set `retry_failed_downloads = True` in `train.py` if you want to try failed links again. Image
sources have their own terms; review the dataset's source and licence information before use.

### Before you start training

- Keep each dataset's metadata file beside its images (any nested subfolder is fine).
- Do not rename image files independently of their metadata IDs; the trainer uses those IDs to
  match rows to images.
- Supported image formats are `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp`.
- The default `require_all_datasets = True` setting expects at least 50 usable images from each
  of the three datasets. If you intentionally have only a subset, set
  `require_all_datasets = False` in the `Config` section of `train.py`; the remaining dataset(s)
  must still contain at least 50 usable images.
- The trainer creates `data/prepared/` itself. Do not download or manually create the prepared
  images, `metadata.jsonl`, or `heldout.jsonl`.
- `data/` is ignored by Git, so downloaded datasets and generated prepared images are not added
  to commits by default.

If a dataset is missing or its metadata/images cannot be matched, training stops and prints the
dataset folder and expected files. Correct the folder contents and run `python train.py` again.

## Train the model

```bash
python train.py
```

The training script performs the full setup and training workflow:

1. Verifies CUDA availability.
2. Organizes recognized loose dataset files into `data/raw/<dataset>/`.
3. Downloads the Fitzpatrick17k CSV and available images.
4. Reads metadata, matches images, builds captions, and creates resized images and split metadata
   under `data/prepared/` (2% held out for quality checks).
5. Fine-tunes the LoRA on the GPU, writing checkpoints and training state under
   `medsynth-model/`.
6. Exports the standalone pipeline to `medsynth-model/pipeline/`.

**Stopped or crashed?** Just run `python train.py` again. It resumes from the newest
checkpoint (saved every 1,000 steps). **Already finished?** Running it again prints where the
model is and exits, so you can never retrain by accident.

**First-time check:** set `smoke_test = True` in the config (below) and run it once. It uses a
small slice of each dataset, 40 steps, and writes to `medsynth-smoketest/` and
`data/prepared_smoketest/`. Set it back to `False` before full training.

### What one epoch looks like
Every epoch draws from the three views in fixed proportions (35 % dermoscopy, 25 % smartphone,
40 % clinical) so the small PAD-UFES-20 set is not drowned out. Each view also gets only the
augmentations that make sense for it: dermoscopy is flipped and rotated freely; smartphone and
clinical photos are only mirrored, so gravity and anatomy stay realistic.

### GPU speed
Training runs on CUDA with mixed precision (bf16 or fp16), TF32, cuDNN autotuning and a fused
optimizer. The batch settings are chosen from your VRAM automatically while keeping 16 images
per optimizer step. These thresholds are estimates:

| VRAM | Batch × accumulation | Gradient checkpointing |
|---|---|---|
| 20 GB+ | 4 × 4 | off |
| 14–20 GB | 2 × 8 | off |
| 10–14 GB | 2 × 8 | on |
| under 10 GB | 1 × 16 | on |

Rough time for the default 15,000 steps is on the order of half a day on a 24 GB card and
longer on slower ones. Check the `it/s` in the progress bar, and look at `medsynth-model/samples/`
(4 preview images, one per view, every 1,000 steps).

## Configure training

There are no command-line arguments. Open `train.py` and edit the **CONFIG** section near the top.

### Per dataset (`DATASETS`)

| Setting | HAM10000 | PAD-UFES-20 | Fitzpatrick17k | Meaning |
|---|---|---|---|---|
| `caption_prefix` | dermoscopy image of | smartphone photo of | clinical photo of | how the view is named in prompts |
| `sample_share` | 0.35 | 0.25 | 0.40 | share of each epoch |
| `max_per_class` | 1200 | 1200 | 600 | cap per diagnosis so common ones don't dominate |
| `min_side` | 256 | 128 | 128 | drop images smaller than this (avoids blurry upscaling) |
| `hflip / vflip / rot90` | yes / yes / yes | yes / no / no | yes / no / no | safe augmentations |
| `use_category` | yes | yes | yes | add benign / malignant / pre-malignant |
| `use_body_site` | yes | yes | no (not in data) | add e.g. "back", "face" |
| `use_skin_type` | no (not in data) | yes | yes | add "Fitzpatrick skin type N" |

### General (`Config`)

| Setting | Default | Notes |
|---|---|---|
| `max_train_steps` | 15000 | try 5000 for a quicker first model |
| `effective_batch` / `auto_batch` | 16 / True | set `auto_batch = False` to use `batch_size`, `grad_accum`, `gradient_checkpointing` yourself |
| `learning_rate`, `rank` | 1e-4, 32 | LoRA rank 64 gives more capacity |
| `resolution` | 512 | |
| `require_all_datasets` | True | set False to train on whichever datasets you have |
| `retrain` | False | True discards checkpoints and trains from scratch (set it back afterwards) |
| `smoke_test` | False | quick end-to-end check |
| `num_workers` | 2 (0 on Windows) | |

## Run the app

```bash
python main.py
```

`main.py` loads `medsynth-model/pipeline` from disk in offline mode and reuses it for every
prompt. Training data, checkpoints, and the Hugging Face base-model cache are not needed for
inference once the pipeline is exported. Keep the complete `medsynth-model/pipeline/` directory.

### Can I delete the `data/` folder?

**Yes—after training has completed and you have confirmed image generation works.** Check that
`medsynth-model/pipeline/model_index.json` exists, start the app with `python main.py`, and
generate an image before removing training data.

For inference-only use, you can then delete the entire project `data/` folder. The trained
pipeline is self-contained, and the app does not read training data. Deleting `data/` removes the
raw datasets, prepared images/captions, and held-out evaluation split; you will need to download
and prepare datasets again before retraining or running those quality checks. It does **not**
remove the trained pipeline.

If you may want to retrain or evaluate later, keep at least `data/raw/` and `data/prepared/`.
Training checkpoints and the separate LoRA weights are under `medsynth-model/`; retain them too
if you need to resume or continue training. The app creates `results/` for chat history; keep it
only if you want to preserve conversations.

To load a pipeline stored elsewhere, set `MEDSYNTH_MODEL_DIR` to the directory containing
`model_index.json` before starting the app:

```powershell
# Windows PowerShell
$env:MEDSYNTH_MODEL_DIR = "C:\path\to\pipeline"
python main.py
```

```bash
# macOS/Linux
MEDSYNTH_MODEL_DIR=/path/to/pipeline python main.py
```

### Writing prompts
Prompts work best in the same grammar as the training captions. Choose **Dermoscopy /
Smartphone / Clinical photo** in the app; with *Clinical prompt enhancement* on, the matching
prefix is added for you, so you type the rest:

```
melanoma, malignant, back
basal cell carcinoma, malignant, face, Fitzpatrick skin type 2
psoriasis, non-neoplastic, Fitzpatrick skin type 5
```

Free text works too — enhancement maps what you wrote onto that grammar:

```
a mole on the left arm of a fitzpatrick 3 patient   ->  smartphone photo of melanocytic nevus,
                                                         benign, arm, Fitzpatrick skin type 3, …
irregular lesion, pigment network, upper extremity  ->  dermoscopy image of … , upper extremity, …
```

Expand **How the prompt is built** under the composer to see the exact caption that will be sent,
the token count, and which parts were recognised.

`medsynth-model/prompt_examples.txt` lists one example caption per class (HAM10000 has 7,
PAD-UFES-20 has 7, Fitzpatrick17k has 114). Use the skin-type phrase (1 to 6) to steer skin
tone. Good settings: 30 steps, guidance 6 to 8.

### Clinical prompt enhancement
The fine-tune only ever saw short captions shaped
`<view> of <diagnosis>, <category>, <body site>, <Fitzpatrick type>`. Enhancement keeps you
inside that distribution instead of layering prose on top of it. It:

1. picks the caption prefix for the view you selected,
2. normalises your wording onto the dataset vocabulary (`bcc` → `basal cell carcinoma`,
   `left arm` → `arm`, `fitzpatrick 3` → `Fitzpatrick skin type 3`),
3. infers what you left out — a matched diagnosis supplies its category,
4. appends the imaging terms that view needs (`10x magnification`, `polarized illumination`,
   …) while budget allows.

Turn it off to send your text verbatim. Either way the caption is capped at **75 CLIP tokens**,
which is what keeps the tokenizer from rejecting long prompts.

### Conversations
Every thread is stored in `results/medsynth_history.sqlite3` together with its images.

- **New conversation** starts a fresh thread; the sidebar lists all of them with message and
  image counts and a relative timestamp.
- **Search** filters by title *and* by anything said inside a thread.
- **Clear chat** empties a thread but keeps it; **Delete conversation** needs two clicks so a
  mis-click cannot lose history.
- Generation settings (view, steps, guidance, seed, negative prompt) are saved **per
  conversation** — switching threads restores the setup that thread was last used with.
- **Shuffle** rolls a new seed, and every generated image records the seed that produced it, so
  a good result can be reproduced by pasting that number back into the seed field.

### Using the model in other projects

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "medsynth-model/pipeline", torch_dtype=torch.float16, local_files_only=True
).to("cuda")
image = pipe("dermoscopy image of melanoma, malignant, back",
             num_inference_steps=30, guidance_scale=6.5).images[0]
```
Back up `medsynth-model/pipeline`. For a small file to share, use
`medsynth-model/lora/pytorch_lora_weights.safetensors` with `pipe.load_lora_weights(...)` on the
same SD 1.5 base.

## Quality checks
`data/prepared/heldout.jsonl` lists 2 % of images kept out of training. Generate from their
captions and compare with those real images (FID/KID), or train a classifier on synthetic images
and test it on real ones. If the previews stop improving, or start copying training images, use
an earlier `medsynth-model/checkpoints/lora-step-*`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA is not available ... CPU-only build` | Reinstall a PyTorch build that matches your CUDA driver (see [Install](#install)). |
| `CUDA out of memory` | Set `auto_batch = False`, `batch_size = 1`, `grad_accum = 16`, `gradient_checkpointing = True`. |
| Loss becomes `nan` with fp16 | Set `mixed_precision = "bf16"` (RTX 30-series and newer) or `learning_rate = 5e-5`. |
| Cannot start: some datasets are not ready | Check the `data/raw/ham10000/`, `data/raw/pad_ufes_20/`, and `data/raw/fitzpatrick17k/` folders, metadata filenames, and image files. |
| App says "Trained model not found" | Training hasn't finished, or the model isn't at `medsynth-model/pipeline`. |
| Many Fitzpatrick downloads fail | Expected for dead links. Set `retry_failed_downloads = True` to try them again. |

## Responsible use

- The output is **synthetic data**. It must not be used for diagnosis or presented as real
  patient images.
- HAM10000 (CC BY-NC) and the Fitzpatrick17k image sources restrict commercial use. Treat the
  trained weights as **research/non-commercial** unless you have cleared every licence.
- The datasets are skewed (HAM10000 toward lighter skin, PAD-UFES-20 from one Brazilian region).
  Fitzpatrick17k plus the skin-type captions helps, but check outputs across skin tones.
- The benign / malignant / pre-malignant words are a simplified mapping made for prompting, not
  clinical ground truth. Have a dermatologist review samples if the images feed research.