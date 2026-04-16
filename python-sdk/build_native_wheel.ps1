$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$projectRoot = Split-Path -Parent $root
$dllSource = Join-Path $projectRoot "target\release\sim_core.dll"
$dllTargetDir = Join-Path $root "robot_sim\_native"

if (-not (Test-Path $dllSource)) {
    Push-Location $projectRoot
    try {
        cargo build -p sim_core --release
    }
    finally {
        Pop-Location
    }
}

New-Item -ItemType Directory -Force -Path $dllTargetDir | Out-Null
Copy-Item -LiteralPath $dllSource -Destination (Join-Path $dllTargetDir "sim_core.dll") -Force

Push-Location $root
try {
    & "C:\Users\root\Miniconda3\python.exe" -m build --wheel --no-isolation
}
finally {
    Pop-Location
}
