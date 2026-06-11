$ErrorActionPreference = "Stop"

$repoUrl = "https://github.com/ByteDance-Seed/SeedVR.git"
$targetPath = Join-Path (Get-Location) "SeedVR"

if (Test-Path $targetPath) {
    if (Test-Path (Join-Path $targetPath ".git")) {
        Write-Host "SeedVR already exists, updating: $targetPath"
        git -C $targetPath pull --ff-only
    }
    else {
        throw "SeedVR exists but is not a git repository: $targetPath"
    }
}
else {
    Write-Host "Cloning SeedVR to: $targetPath"
    git clone $repoUrl $targetPath
}

$requiredPaths = @(
    "projects/inference_seedvr2_3b.py",
    "configs_3b/main.yaml",
    "common",
    "models",
    "data",
    "projects/video_diffusion_sr",
    "pos_emb.pt",
    "neg_emb.pt"
)

foreach ($relativePath in $requiredPaths) {
    $path = Join-Path $targetPath $relativePath
    if (-not (Test-Path $path)) {
        throw "SeedVR checkout is incomplete, missing: $relativePath"
    }
}

Write-Host "SeedVR is ready: $targetPath"