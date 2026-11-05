# Vocal intensity estimation and expressive TTS evaluation

Quentin Le Tellier, Albert Rilliard, Marc Evrard
Experiments of *Time-resolved vocal intensity estimation for the evaluation of expressive
speech synthesis*, submitted to Computer Speech & Language. *Paper link to come.*

## Overview

This repository holds the scripts and configurations that reproduce the experiments of the
paper:

1. **Intensity estimation.** Predictors of vocal intensity (dB SPL, every 20 ms) are trained
   on calibrated recordings and evaluated over frames, sliding windows and utterances,
   against a regression on spectral tilt.
2. **TTS generation.** Three expressive TTS systems (CosyVoice 3, IndexTTS2, Qwen3-TTS)
   synthesize speech for eight emotions and four vocal effort levels.
3. **TTS evaluation.** The vocal intensity of the synthesized and natural speech is
   estimated with one of these predictors and compared per condition.

The predictor itself — model, training, inference, and the spectral-tilt baseline — is in
**[vocal-intensity-prediction](https://github.com/vers-project/vocal-intensity-prediction)**,
a general-purpose toolkit. Use it directly to train a predictor or to estimate the vocal
intensity of your own recordings. This repository only adds the paper's configurations and
the TTS experiments around it.

## Repository layout

```
intensity_estimation/   predictor grid and tilt baseline configs, results table (run with vocal-intensity-prediction)
tts_generation/
  generation_design/    shared package: generation config -> manifest of samples, generation loop
  cosyvoice3/           CosyVoice 3 (git submodule ./CosyVoice), own environment
  indextts2/            IndexTTS2, own environment
  qwen3tts/             Qwen3-TTS VoiceDesign, own environment
  generate.sh           every system × every experiment
tts_evaluation/         manifests, attribute table, distributions, figure, paper numbers
```

Every configuration is a YAML file run through
[slurm_launcher](https://github.com/vers-project/slurm_launcher)
(`launch_experiment --script <script> --config <config>`), locally or on a SLURM cluster
(set `launcher.device_launcher`). Paths starting with `/path/to/`, and `<date>/<time>`
run folders, are placeholders: point them at your own copies of the data and at your own
runs. Each run writes its outputs and the configuration it used to
`<output_dir>/<date>/<time>/<job_index>/`.

## Installation

[uv](https://docs.astral.sh/uv/) ≥ 0.9 is required. Clone with the submodules (CosyVoice and
its Matcha-TTS dependency):

```bash
git clone --recursive https://github.com/vers-project/csl-intensity-tts-eval.git
```

Each folder with a `pyproject.toml` is its own environment, because the TTS systems need
incompatible dependencies. Each `uv.lock` pins the versions the paper's experiments ran
with.

| folder | Python | setup |
|---|---|---|
| `tts_generation/cosyvoice3` | 3.12 | `uv sync` |
| `tts_generation/indextts2` | 3.11 | `uv sync` |
| `tts_generation/qwen3tts` | 3.12 | `uv sync --extra cu128` |
| `tts_evaluation` | 3.12 | `uv sync --extra cpu` |

The predictor steps run in a clone of
[vocal-intensity-prediction](https://github.com/vers-project/vocal-intensity-prediction)
at tag `v0.1.0`, and the RMS level in a clone of
[speech-eval](https://github.com/vers-project/speech-eval) at tag `v0.2.0`, each with its own
environment:

```bash
git clone --branch v0.1.0 https://github.com/vers-project/vocal-intensity-prediction.git
git clone --branch v0.2.0 https://github.com/vers-project/speech-eval.git
```

## Data

| corpus | used for | licence |
|---|---|---|
| [AVID](https://doi.org/10.5281/zenodo.10524873) | predictor training and test; effort prompts and natural reference | CC BY 4.0 |
| [FLombard](https://doi.org/10.5281/zenodo.17340497) | predictor test | CC BY-NC-SA 4.0 |
| [RAVDESS](https://doi.org/10.5281/zenodo.1188976) | emotion prompts and natural reference | CC BY-NC-SA 4.0 |
| VIC | predictor test | released once the corpus is complete |

The metadata table of the predictor's corpora is built with the steps of
vocal-intensity-prediction (*Training*, step 1). Two further tables come with the data
release of the paper (*link to come*): the RAVDESS metadata, and the AVID metadata with our
hand annotation of the AVID recording sessions (effort level, sentence and reading of each
take). The data release also holds the synthesized speech.

## 1. Intensity estimation

Run from the vocal-intensity-prediction clone, with the configurations of
`intensity_estimation/configs/`:

```bash
# the 9 predictors: log-mel, WavLM Base+ layer 3, wav2vec 2.0 XLS-R 300M layer 5; 3 seeds each
uv run --extra cu128 launch_experiment --script scripts/train_predictor.py \
    --config <this repo>/intensity_estimation/configs/train_predictor_grid.yaml

# spectral tilt of every evaluation window and the alpha-ratio baseline
uv run --extra cu128 launch_experiment --script scripts/extract_window_tilt.py \
    --config <this repo>/intensity_estimation/configs/extract_window_tilt.yaml
uv run --extra cpu python scripts/fit_tilt_baseline.py <window tilt run> \
    --feature alpha --degrees 1 3 --output-dir <window tilt run>/alpha
uv run --extra cpu python scripts/fit_tilt_baseline.py <window tilt run> \
    --feature hammarberg --degrees 1 3 --output-dir <window tilt run>/hammarberg
```

Then the results table (LaTeX), with any environment that has pandas and PyYAML:

```bash
python intensity_estimation/scripts/make_results_table.py \
    --experiments_dir <grid run> \
    --alpha_baseline <window tilt run>/alpha --hammarberg_baseline <window tilt run>/hammarberg \
    --output results_table.tex
```

Job 3 of the grid (WavLM Base+ layer 3, seed 0) is the predictor used in the TTS evaluation.

## 2. TTS generation

Download each system's weights (the scripts pin the Hugging Face revisions used in the
paper), set `model_dir` / `model.dir` and the data paths in the configs, then generate:

```bash
(cd tts_generation/cosyvoice3 && uv run launch_experiment --script download_model.py --config configs/download_model.yaml)
(cd tts_generation/indextts2  && uv run python download_models.py --config configs/emotion.yaml)
(cd tts_generation/qwen3tts   && uv run --extra cu128 launch_experiment --script download_model.py --config configs/download_model.yaml)

bash tts_generation/generate.sh                      # every system × experiment
bash tts_generation/generate.sh cosyvoice3/effort    # or a subset
```

| system | emotion control | effort control |
|---|---|---|
| CosyVoice 3 | RAVDESS clip of the emotion as prompt | AVID take at the level as prompt |
| IndexTTS2 | emotion vector, RAVDESS neutral clip as speaker prompt | AVID take at the level as emotion prompt |
| Qwen3-TTS VoiceDesign | text instruction | text instruction; `effort_scale`: a numbered four-level scale |

Every speaker × sentence × seed is generated under every condition of its experiment
(`generation_design`). Audio is written as 32-bit float WAV, without rescaling.

## 3. TTS evaluation

The configurations are in `tts_evaluation/configs/`, in this order:

| step | config | run from | script |
|---|---|---|---|
| VAD-trimmed manifest per source | `manifests` | `tts_evaluation/` | `scripts/build_eval_manifests.py` |
| vocal intensity (the predictor) | `predict` | vocal-intensity-prediction | `scripts/predict_intensity.py` |
| alpha-ratio baseline | `tilt_predict` | vocal-intensity-prediction | `scripts/predict_tilt_baseline.py` |
| RMS level | `metrics` | speech-eval | `scripts/run_eval.py` |
| attribute table, failed generations | `attributes` | `tts_evaluation/` | `scripts/build_attribute_table.py` |
| distributions per condition | `distributions` | `tts_evaluation/` | `scripts/attribute_distributions.py` |
| figure | `figures` | `tts_evaluation/` | `scripts/make_figures.py` |
| numbers quoted in the text | `numbers` | `tts_evaluation/` | `scripts/make_paper_numbers.py` |

For example, from `tts_evaluation/`:

```bash
uv run --extra cpu launch_experiment --script scripts/build_eval_manifests.py --config configs/manifests.yaml
```

The tests run with `uv run --extra cpu --with pytest pytest tests` in `tts_evaluation/` and
`uv run --with pytest pytest tests` in `tts_generation/generation_design/`.

## Third-party systems

CosyVoice, IndexTTS2 and Qwen3-TTS are used unmodified, at the versions pinned here, and
each comes with its own licence for code and model weights; see their repositories. The MIT
licence of this repository covers only its own code.

## Acknowledgements

This work was supported by the French National Research Agency (ANR) through the VERS
project (ANR-23-CE38-0010-01). It was granted access to the HPC resources of IDRIS under the
allocation 2026-AD011016051R1 made by GENCI.

## Licence

MIT. See [`LICENSE`](LICENSE).
