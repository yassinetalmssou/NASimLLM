################################################################################
# hpc_upload.ps1 — Upload NASimLLM project to VUB Hydra HPC (PowerShell/scp)
#
# Run from the project root in PowerShell:
#   .\hpc_upload.ps1
#
# Uses built-in Windows scp — no extra tools needed.
################################################################################

$HPC_USER = "vsc11800"
$HPC_HOST = "login.hpc.vub.be"
$HPC_DEST = "/data/leuven/118/vsc11800/NASimLLM"   # $VSC_DATA/NASimLLM on Hydra
$PROJECT  = Split-Path -Parent $MyInvocation.MyCommand.Path

# Folders/files to skip (won't be uploaded)
$EXCLUDE = @(
    "llm_weights"
    ".venv"
    "runs"
    "__pycache__"
    "nasim.egg-info"
    ".ipynb_checkpoints"
    "*.safetensors"
    "*.bin"
    "*.pt"
    "*.pth"
    "logs"
    ".git"
    "dist"
    "build"
)

Write-Host ""
Write-Host "NASimLLM -> VUB Hydra HPC Upload" -ForegroundColor Cyan
Write-Host "  Local : $PROJECT"
Write-Host "  Remote: ${HPC_USER}@${HPC_HOST}:${HPC_DEST}"
Write-Host ""

# Create remote directory
Write-Host "Creating remote directory..." -ForegroundColor Yellow
ssh "${HPC_USER}@${HPC_HOST}" "mkdir -p $HPC_DEST"

# Collect files to upload (respecting exclusions)
$allItems = Get-ChildItem -Path $PROJECT -Force | Where-Object {
    $name = $_.Name
    $excluded = $false
    foreach ($ex in $EXCLUDE) {
        if ($name -like $ex) { $excluded = $true; break }
    }
    -not $excluded
}

Write-Host "Uploading files..." -ForegroundColor Yellow
foreach ($item in $allItems) {
    if ($item.PSIsContainer) {
        Write-Host "  [dir]  $($item.Name)"
        scp -r "$($item.FullName)" "${HPC_USER}@${HPC_HOST}:${HPC_DEST}/"
    } else {
        Write-Host "  [file] $($item.Name)"
        scp "$($item.FullName)" "${HPC_USER}@${HPC_HOST}:${HPC_DEST}/"
    }
}

Write-Host ""
Write-Host "Upload complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps on the HPC:"
Write-Host "  ssh ${HPC_USER}@${HPC_HOST}"
Write-Host "  cd `$VSC_DATA/NASimLLM"
Write-Host "  bash hpc_setup.sh"
