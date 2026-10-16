$ErrorActionPreference = 'Stop'
Set-Location "$PSScriptRoot\.."
$package = 'dist/SimRoki-windows-x64'
$binaryDir = 'target/x86_64-pc-windows-msvc/release'
if (Test-Path $package) { throw "Package directory already exists: $package" }
New-Item -ItemType Directory -Path $package -Force | Out-Null
Copy-Item "$binaryDir/native_app.exe", "$binaryDir/sim_core.dll" $package
Copy-Item robot_config.toml, motion_sequence.json, README.md $package
Copy-Item docs, motions, planner $package -Recurse
New-Item -ItemType Directory -Path "$package/python-sdk" -Force | Out-Null
Copy-Item python-sdk/robot_sim, python-sdk/examples "$package/python-sdk" -Recurse
Copy-Item python-sdk/README.md, python-sdk/PROTOCOL.md, python-sdk/pyproject.toml "$package/python-sdk"
New-Item -ItemType Directory -Path "$package/python-sdk/robot_sim/_native" -Force | Out-Null
Copy-Item "$binaryDir/sim_core.dll" "$package/python-sdk/robot_sim/_native/sim_core.dll" -Force
@'
@echo off
cd /d "%~dp0"
native_app.exe
if errorlevel 1 pause
'@ | Set-Content "$package/Start-SimRoki.cmd" -Encoding ascii
@'
SimRoki - Windows x64 portable build

Extract the entire ZIP before starting. Double-click Start-SimRoki.cmd.
No Rust or Python installation is required for the desktop app.
The launcher sets the working directory so configuration and motions load correctly.
Keep robot_config.toml and motion_sequence.json next to native_app.exe.
The local control API listens at http://127.0.0.1:8080.

Controls: Space = pause/resume, R = reset, mouse wheel = zoom.
Optional Python SDK (Python 3.11+): python -m pip install ./python-sdk
See README.md and python-sdk/README.md for API examples.

This build is unsigned. Interactive graphics require a compatible graphics driver.
'@ | Set-Content "$package/START-HERE.txt" -Encoding utf8
git rev-parse HEAD | Set-Content "$package/SOURCE-COMMIT.txt" -Encoding ascii
Compress-Archive -Path $package -DestinationPath 'dist/SimRoki-windows-x64.zip'
$hash = (Get-FileHash 'dist/SimRoki-windows-x64.zip' -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  SimRoki-windows-x64.zip" | Set-Content 'dist/SimRoki-windows-x64.zip.sha256' -Encoding ascii
