"""Download the Fun-CosyVoice3-0.5B weights into ``model_dir`` and check the expected files.

Usage
-----
  uv run launch_experiment --script download_model.py --config configs/download_model.yaml
"""
from pathlib import Path

from experiment_launcher import parse_args

# The repository (renamed from Fun-CosyVoice3-0.5B) and the commit used in the paper.
REPO_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
REVISION = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
REQUIRED_FILES = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "llm.pt",
                  "flow.pt", "hift.pt"]


@parse_args
def main(config: dict, output_dir: Path):
    from huggingface_hub import snapshot_download

    model_dir = Path(config["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID}@{REVISION} -> {model_dir}")
    snapshot_download(repo_id=REPO_ID, revision=REVISION, local_dir=str(model_dir),
                      ignore_patterns=["*.git*", "figures/*"])

    missing = [f for f in REQUIRED_FILES if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {model_dir}: {missing}")
    print("Done.")


if __name__ == "__main__":
    main()
