#!/usr/bin/env bash
# Batch-run the talking-head pipeline over several voice clips on ONE image.
# Each voice gets its own pipeline/outputs/run_<timestamp>/ folder
# (4 model sub-folders + summary.json inside each).
#
# Usage:   bash run_batch.sh
# Edit IMAGE and the AUDIOS array below to change inputs.
#
# Note: NO `set -e` on purpose - run_pipeline.py exits 1 when any single model
# fails, and we still want the remaining voices to run.
set -u

cd "$(dirname "$0")"
# IMAGE="/home/porch/Desktop/code/project/voice/pipeline/experimental_source/women.png"
IMAGE="/home/porch/Desktop/code/project/voice/pipeline/experimental_source/boy.jpeg"
# IMAGE="/home/porch/Pictures/Screenshots/Screenshot from 2026-05-20 17-33-42.png"
# --- inputs ---------------------------------------------------------------
# IMAGE="/home/porch/Desktop/code/project/voice/portrait_asian_woman.png"
# IMAGE="/home/porch/Downloads/360_F_500522027_QQFrNZH8jd2ieiraJ4cuRLlDM37Jt4Kw.jpg"
# AUDIOS=(
#   "/home/porch/Downloads/output/output/female_teacher/baifern/03_encourage.wav"
#   "/home/porch/Downloads/output/output/female_teacher/baifern/04_qa.wav"
#   "/home/porch/Downloads/output/output/female_teacher/bantita/01_opening.wav"
# )
AUDIOS=("/home/porch/Desktop/code/project/voice/pipeline/recording_22052026.wav")

# AUDIOS=("/home/porch/Desktop/code/project/synthetic_voice/inference_out/9arm_ajarn_jem_v3.wav")

# Output folder: each voice still gets its own run_<timestamp>/ subfolder
# INSIDE this directory. Default is pipeline/outputs/. Override per-invocation
# with: OUTDIR=/path/to/folder bash run_batch.sh
# OUTDIR="/home/porch/Desktop/code/project/voice/pipeline/outputs_v2"
OUTDIR="/home/porch/Desktop/code/project/voice/pipeline/experimental_source/outputs_v4_original_source"

# python that runs run_pipeline.py (only needs pyyaml; the 4 model conda envs
# are selected per-model inside config.yaml). Override with: RUNNER_PY=... bash run_batch.sh
RUNNER_PY="${RUNNER_PY:-python}"

# --- run ------------------------------------------------------------------
total=${#AUDIOS[@]}
i=0
for audio in "${AUDIOS[@]}"; do
  i=$((i + 1))
  echo
  echo "############################################################"
  echo "# [$i/$total] voice: $audio"
  echo "############################################################"
  "$RUNNER_PY" run_pipeline.py --source-image "$IMAGE" --driven-audio "$audio" \
                               --outputs-dir "$OUTDIR"
  echo "# [$i/$total] finished (run_pipeline exit $?)"
done

echo
echo "All $total voice runs done. Outputs: $OUTDIR/run_<timestamp>/"
