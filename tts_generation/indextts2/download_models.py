"""Download every model IndexTTS2 needs, so that generation can run offline.

The IndexTTS2 checkpoint goes to ``model.dir`` and the auxiliary models to
``model.dir/hf_cache``. Each is verified against the specific files IndexTTS2
reads at startup, so a partially downloaded directory is detected and re-fetched.

Usage
-----
  uv run python download_models.py --config configs/emotion.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from omegaconf import OmegaConf


# The commit of each Hugging Face repository used in the paper.
REVISIONS = {
    "IndexTeam/IndexTTS-2": "740dcaff396282ffb241903d150ac011cd4b1ede",
    "facebook/w2v-bert-2.0": "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
    "amphion/MaskGCT": "265c6cef07625665d0c28d2faafb1415562379dc",
    "funasr/campplus": "e4b6ede7ce16997aff4ae69fbca1f0175e2afede",
    "nvidia/bigvgan_v2_22khz_80band_256x": "633ff708ed5b74903e86ff1298cf4a98e921c513",
}


# ---------------------------------------------------------------------------
# Per-model download helpers
# ---------------------------------------------------------------------------

def _ensure_main_model(model_dir: Path, repo_id: str = "IndexTeam/IndexTTS-2") -> Path:
    """IndexTTS2 itself — config.yaml plus every checkpoint it names."""
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        print(f"  [dl] {repo_id} -> {model_dir}")
        snapshot_download(repo_id, revision=REVISIONS[repo_id], local_dir=str(model_dir))
    cfg = OmegaConf.load(config_path)
    required = [cfg.gpt_checkpoint, cfg.s2mel_checkpoint, cfg.dataset["bpe_model"],
                cfg.emo_matrix, cfg.spk_matrix, cfg.w2v_stat, cfg.qwen_emo_path]
    missing = [f for f in required if not (model_dir / f).exists()]
    if missing:
        print(f"  [dl] {repo_id} (missing {missing}) -> {model_dir}")
        snapshot_download(repo_id, revision=REVISIONS[repo_id], local_dir=str(model_dir))
    print(f"  [ok] IndexTTS2  ({model_dir})")
    return model_dir


def _ensure_w2v_bert(cache_dir: Path) -> Path:
    """facebook/w2v-bert-2.0 — full repo needed by SeamlessM4TFeatureExtractor."""
    dst = cache_dir / "w2v-bert-2.0"
    required = ["preprocessor_config.json", "config.json", "model.safetensors"]
    if all((dst / f).exists() for f in required):
        print(f"  [ok] w2v-bert-2.0  ({dst})")
        return dst
    print(f"  [dl] facebook/w2v-bert-2.0 -> {dst}")
    snapshot_download("facebook/w2v-bert-2.0", revision=REVISIONS["facebook/w2v-bert-2.0"], local_dir=str(dst))
    return dst


def _ensure_file(
    cache_dir: Path,
    repo_id: str,
    remote_file: str,
    local_name: str,
    label: str,
) -> Path:
    """Download a single file from a HF repo into cache_dir/local_name."""
    dst = cache_dir / local_name
    if dst.exists():
        print(f"  [ok] {label}  ({dst})")
        return dst
    print(f"  [dl] {repo_id}/{remote_file} -> {dst}")
    tmp = Path(hf_hub_download(repo_id=repo_id, filename=remote_file, revision=REVISIONS[repo_id], local_dir=str(cache_dir)))
    if tmp.resolve() != dst.resolve():
        shutil.copy2(tmp, dst)
    return dst


def _ensure_bigvgan(cache_dir: Path, repo_id: str = "nvidia/bigvgan_v2_22khz_80band_256x") -> Path:
    """BigVGAN vocoder — two files: config.json + bigvgan_generator.pt."""
    dst = cache_dir / "bigvgan"
    dst.mkdir(parents=True, exist_ok=True)
    required = ["config.json", "bigvgan_generator.pt"]
    for fname in required:
        if (dst / fname).exists():
            print(f"  [ok] bigvgan/{fname}  ({dst / fname})")
            continue
        print(f"  [dl] {repo_id}/{fname} -> {dst / fname}")
        tmp = Path(hf_hub_download(repo_id=repo_id, filename=fname, revision=REVISIONS[repo_id], local_dir=str(dst)))
        if tmp.resolve() != (dst / fname).resolve():
            shutil.copy2(tmp, dst / fname)
    return dst


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-download all IndexTTS2 auxiliary models")
    p.add_argument("--config", type=Path, required=True, help="Path to the experiment config yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    model_dir = Path(config["model"]["dir"])
    cache_dir = model_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model dir : {model_dir}")
    print(f"Cache dir : {cache_dir}\n")

    _ensure_main_model(model_dir)
    _ensure_w2v_bert(cache_dir)
    _ensure_file(cache_dir, "amphion/MaskGCT", "semantic_codec/model.safetensors", "semantic_codec_model.safetensors", "MaskGCT semantic_codec")
    _ensure_file(cache_dir, "funasr/campplus", "campplus_cn_common.bin", "campplus_cn_common.bin", "CAMPPlus")
    _ensure_bigvgan(cache_dir)

    print("\nDone — all auxiliary models are ready for offline use.")


if __name__ == "__main__":
    main()
