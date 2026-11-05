"""Download Qwen3-TTS VoiceDesign into a local directory, then load it once on CPU.

The load check makes a missing file (e.g. the ``speech_tokenizer/`` subfolder) fail here
rather than inside a GPU job.  Loading from a local directory never contacts the Hub.

Usage
-----
  uv run --extra cpu launch_experiment --script download_model.py --config configs/download_model.yaml
"""
from pathlib import Path
from experiment_launcher import parse_args


@parse_args
def main(config: dict, output_dir: Path):
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{config['model_id']}'@{config['revision']} into {config['model_dir']} ...")
    snapshot_download(repo_id=config["model_id"], revision=config["revision"],
                      local_dir=config["model_dir"])

    if config.get("check_load", True):
        import torch
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(config["model_dir"], device_map="cpu",
                                              dtype=torch.bfloat16, attn_implementation="sdpa")
        print(f"Loaded on CPU: model type {model.model.tts_model_type}, "
              f"languages {model.get_supported_languages()}")
    print("Done.")


if __name__ == "__main__":
    main()
