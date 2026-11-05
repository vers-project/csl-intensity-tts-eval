#!/usr/bin/env bash
# Generate every TTS sample set: each system × each experiment config it has.
#
#   bash tts_generation/generate.sh                         # every (system, experiment)
#   bash tts_generation/generate.sh cosyvoice3/effort ...   # only these
#
# Experiments: emotion, effort, and effort_scale (the numbered four-level scale, Qwen3-TTS
# only).  Seeds, output directory and launcher come from each config
# (<system>/configs/<experiment>.yaml).
#
# Each system has its own environment, so each config is launched from its folder with
# `uv run launch_experiment`.  Re-running is safe: rows whose WAV exists are skipped.
set -uo pipefail

generation_dir=$(cd "$(dirname "$0")" && pwd)

# system : uv extras for a GPU machine
systems=("cosyvoice3:" "indextts2:" "qwen3tts:--extra cu128")
experiments=(emotion effort effort_scale)

selected=("$@")
is_selected() {
    [[ ${#selected[@]} -eq 0 ]] && return 0
    local s
    for s in "${selected[@]}"; do [[ $s == "$1" ]] && return 0; done
    return 1
}

declare -a summary
for entry in "${systems[@]}"; do
    IFS=: read -r system extras <<< "$entry"
    for experiment in "${experiments[@]}"; do
        is_selected "$system/$experiment" || continue
        config="configs/$experiment.yaml"
        [[ -f "$generation_dir/$system/$config" ]] || continue

        echo "=== $system / $experiment"
        # shellcheck disable=SC2086  # $extras is deliberately split into flags
        if (cd "$generation_dir/$system" && uv run $extras launch_experiment \
                --script generate_parallel.py --config "$config"); then
            summary+=("ok      $system/$experiment")
        else
            summary+=("FAILED  $system/$experiment")
        fi
    done
done

echo
echo "=== Summary"
printf '%s\n' "${summary[@]}"
