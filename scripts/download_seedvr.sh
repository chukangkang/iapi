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
    "data/image/transforms/divisible_crop.py"
    "data/image/transforms/na_resize.py"
    "data/video/transforms/rearrange.py"
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

package_dirs=(
    "common"
    "configs_3b"
    "data"
    "data/image"
    "data/image/transforms"
    "data/video"
    "data/video/transforms"
    "models"
    "projects"
    "projects/video_diffusion_sr"
)

for relative_dir in "${package_dirs[@]}"; do
    if [ -d "$target_path/$relative_dir" ]; then
        touch "$target_path/$relative_dir/__init__.py"
    fi
done

echo "SeedVR is ready: $target_path"