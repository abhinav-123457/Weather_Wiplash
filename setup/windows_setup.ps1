<#
    windows_setup.ps1 - Stage 0 environment setup for the demo machine
    (i7-1355U / 16 GB / Windows 11)

    Creates the app venv, installs CPU-only PyTorch plus the rest of the
    backend stack, and runs the smoke test.

    RUN FROM THE PROJECT ROOT:

        powershell -ExecutionPolicy Bypass -File setup\windows_setup.ps1

    The -ExecutionPolicy Bypass matters. Windows blocks unsigned scripts by
    default and the venv activation script is unsigned, so without it you get
    a confusing "cannot be loaded because running scripts is disabled" error.
#>

$ErrorActionPreference = "Stop"

function Say($msg)  { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Good($msg) { Write-Host "  OK  $msg"    -ForegroundColor Green }
function Warn($msg) { Write-Host "  !   $msg"    -ForegroundColor Yellow }
function Die($msg)  { Write-Host "  X   $msg"    -ForegroundColor Red; exit 1 }


# ---------------------------------------------------------------------------
Say "Checking Python"
# ---------------------------------------------------------------------------
# On Ubuntu we were stuck with whatever the distro shipped. Here you choose -
# so choose the version with the widest wheel coverage, not the newest.
# 3.12 and 3.13 are the safe picks. 3.14 may work but Windows wheels are built
# separately from Linux ones, so do not assume our Ubuntu result carries over.

$py = $null
foreach ($v in @("3.12", "3.13", "3.11")) {
    try {
        $out = & py "-$v" --version 2>&1
        if ($LASTEXITCODE -eq 0) { $py = "-$v"; Good "found Python $out"; break }
    } catch { }
}

if (-not $py) {
    try {
        $out = & python --version 2>&1
        Warn "falling back to default python: $out"
        Warn "If this is 3.14+, consider installing 3.12 from python.org -"
        Warn "PyTorch Windows wheels lag new Python releases."
        $py = $null
    } catch {
        Die "No Python found. Install 3.12 from python.org and tick 'Add to PATH'."
    }
}


# ---------------------------------------------------------------------------
Say "Creating virtual environment (.venv)"
# ---------------------------------------------------------------------------
if (Test-Path ".venv") {
    Warn ".venv already exists - reusing it. Delete it for a clean start."
} else {
    if ($py) { & py $py -m venv .venv } else { & python -m venv .venv }
    Good "created .venv"
}

$vpy = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { Die "venv python missing at $vpy" }

& $vpy --version
& $vpy -m pip install --quiet --upgrade pip


# ---------------------------------------------------------------------------
Say "Installing PyTorch (CPU-only)"
# ---------------------------------------------------------------------------
# The CPU index serves only CPU builds, so this cannot accidentally pull the
# ~2.5 GB of CUDA packages this machine has no GPU for.
# Deliberately unpinned: Windows wheel availability differs from Linux, and
# any recent 2.x CPU build works for us.

& $vpy -m pip install --no-cache-dir torch torchvision `
    --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { Die "PyTorch install failed - see output above." }
Good "PyTorch installed"


# ---------------------------------------------------------------------------
Say "Installing the rest of the backend stack"
# ---------------------------------------------------------------------------
# transformers is pinned to 5.x deliberately. Almost every CLIP and SegFormer
# tutorial online targets v4 and the APIs differ - pinning stops the version
# shifting under you mid-build and turning working code into import errors.

& $vpy -m pip install --no-cache-dir `
    "transformers>=5.0,<6.0" `
    pillow numpy `
    opencv-python-headless `
    fastapi "uvicorn[standard]" python-multipart `
    huggingface_hub `
    psutil
if ($LASTEXITCODE -ne 0) { Die "Dependency install failed - see output above." }
Good "backend stack installed"


# ---------------------------------------------------------------------------
Say "Running smoke test"
# ---------------------------------------------------------------------------
Write-Host "Downloads about 615 MB of model weights on first run.`n"

& $vpy setup\smoke_test.py
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Say "Stage 0 complete"
    Write-Host "Activate the venv with:  .\.venv\Scripts\Activate.ps1"
    Write-Host "Send the smoke test output back before starting Stage 1."
} else {
    Say "Smoke test FAILED"
    Write-Host "Send the full output above - the failure tells us what to fix."
}
exit $code
