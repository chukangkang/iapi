#!/usr/bin/env bash
set -euo pipefail

repo_url="https://github.com/ByteDance-Seed/SeedVR.git"
target_path="$(pwd)/SeedVR"

if [ -d "$target_path" ]; then
    if [ -d "$target_path/.git" ]; then
        echo "SeedVR already exists, updating: $target_path"
        git -C "$target_path" pull --ff-only
    else
        echo "SeedVR exists but is not a git repository: $target_path" >&2
        exit 1
    fi
else
    echo "Cloning SeedVR to: $target_path"
    git clone "$repo_url" "$target_path"
fi

required_paths=(
    "projects/inference_seedvr2_3b.py"
    "configs_3b/main.yaml"
    "common"
    "models"
    "data"
    "projects/video_diffusion_sr"
    "pos_emb.pt"
    "neg_emb.pt"
)

for relative_path in "${required_paths[@]}"; do
    if [ ! -e "$target_path/$relative_path" ]; then
        echo "SeedVR checkout is incomplete, missing: $relative_path" >&2
        exit 1
    fi
done

echo "SeedVR is ready: $target_path"