"""The generation loop: manifest -> WAV files + generation_log.csv.

Every row is generated with the random state re-seeded to the row's seed, so the conditions
of a pair share the same random draws as far as the system allows.

A row whose WAV already exists is skipped, so a job that hits its time limit is simply
relaunched.  A row that raises is logged as ``failed`` and the loop goes on: a failed
generation is a result (failure rate per condition), not a reason to lose the job.  The log
is appended one line per row, so it survives a killed job; a relaunch appends a new run.

Audio is written as 32-bit float WAV through soundfile (``torchaudio.save`` silently writes
16-bit), without any rescaling or clipping.
"""
from __future__ import annotations

import csv
import random
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

LOG_COLUMNS = ["run_started", "pair_id", "condition", "output_path", "status",
               "duration_s", "elapsed_s", "error"]
CLIP_COLUMNS = ("condition_clip", "anchor_clip")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_row(row: pd.Series, output_dir: Path) -> dict:
    """The row with clip and output paths made absolute, as handed to ``generate``."""
    resolved = row.to_dict()
    if pd.notna(row["clip_root"]):
        root = Path(row["clip_root"])
        root = root if root.is_absolute() else output_dir / root
        for column in CLIP_COLUMNS:
            resolved[column] = str(root / row[column])
    resolved["output_path"] = str(output_dir / row["output_path"])
    return resolved


def run_manifest(
    manifest: pd.DataFrame,
    generate: Callable[[dict], tuple[np.ndarray, int]],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Generate every missing WAV; return this run's log rows.

    ``generate(row) -> (audio, sample_rate)`` receives the resolved row (``resolve_row``)
    and returns a mono waveform.
    """
    output_dir = Path(output_dir)
    log_path = output_dir / "generation_log.csv"
    run_started = datetime.now().isoformat(timespec="seconds")
    new_log = not log_path.exists()
    entries = []
    counts = {"ok": 0, "exists": 0, "failed": 0}

    with open(log_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if new_log:
            writer.writeheader()
        for _, row in manifest.iterrows():
            target = output_dir / row["output_path"]
            entry = {"run_started": run_started, "pair_id": row["pair_id"],
                     "condition": row["condition"], "output_path": row["output_path"],
                     "duration_s": "", "elapsed_s": "", "error": ""}
            if target.exists():
                entry["status"] = "exists"
            else:
                start = time.perf_counter()
                try:
                    seed_everything(int(row["seed"]))
                    audio, sample_rate = generate(resolve_row(row, output_dir))
                    audio = _as_mono_float32(audio)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(target), audio, int(sample_rate), subtype="FLOAT")
                    entry.update(status="ok", duration_s=f"{len(audio) / sample_rate:.3f}")
                except Exception as error:  # a failed row is logged, the loop goes on
                    traceback.print_exc()
                    entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                entry["elapsed_s"] = f"{time.perf_counter() - start:.3f}"
            counts[entry["status"]] += 1
            writer.writerow(entry)
            handle.flush()
            entries.append(entry)

    print(f"Generation: {counts['ok']} written, {counts['exists']} already there, "
          f"{counts['failed']} failed.  Log: {log_path}")
    return pd.DataFrame(entries, columns=LOG_COLUMNS)


def _as_mono_float32(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.ndim != 1:
        raise ValueError(f"Expected a mono waveform, got shape {audio.shape}.")
    if audio.size == 0:
        raise ValueError("Empty waveform.")
    if not np.isfinite(audio).all():
        raise ValueError("Waveform contains NaN or inf.")
    return audio
