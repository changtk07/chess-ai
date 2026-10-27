#!/bin/bash
# Self-play throughput for each network size and number of games at once, to compare a machine
# with the tables in docs/performance.md. Builds first, then plays each combination for DURATION
# seconds and averages its 5-second reports, skipping the first two: every run compiles its
# kernels again, which takes up to about a minute.
#
# Usage: scripts/bench-sizes.sh [BLOCKSxWIDTH ...]    default: 6x256 8x256 10x256 12x256 8x384
# Environment: PARALLEL (default "256 512 1024"), DURATION (default 150)
#
# Check nothing else is using the GPU first: other load skews every number.

cd "$(dirname "$0")/.." || exit 1
export CUDA_PATH=${CUDA_PATH:-/opt/cuda}

sizes=("$@")
[ ${#sizes[@]} -eq 0 ] && sizes=(6x256 8x256 10x256 12x256 8x384)
read -ra parallels <<< "${PARALLEL:-256 512 1024}"
duration=${DURATION:-150}

cargo build --release --quiet || exit 1
nvidia-smi --query-gpu=name,enforced.power.limit --format=csv,noheader

for size in "${sizes[@]}"; do
  blocks=${size%x*}
  width=${size#*x}
  for parallel in "${parallels[@]}"; do
    rate=$(timeout "$duration" target/release/cinder self-play --backend cuda-bf16 --games 1000000 \
        --parallel "$parallel" --blocks "$blocks" --width "$width" 2>&1 \
      | awk '
          /sims\/s/ && ++seen > 2 { sum += $4; n++ }
          END { if (n >= 3) printf "%6.0f simulations/s (%d reports)", sum / n, n
                else print "too few reports, raise DURATION" }')
    printf '%3s x %-4s %4s games: %s\n' "$blocks" "$width" "$parallel" "$rate"
  done
done
