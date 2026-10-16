"""Package an Apple Silicon native app and SDK without developer dependencies."""
import argparse,hashlib,plistlib,shutil,subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--version',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
package=root/'dist/SimRoki-macos-arm64'
if package.exists():raise SystemExit(f'Package already exists: {package}')
contents=package/'SimRoki.app/Contents';macos=contents/'MacOS';resources=contents/'Resources'
macos.mkdir(parents=True);resources.mkdir()
shutil.copy2(root/'target/release/native_app',macos/'native_app')
launcher=macos/'SimRoki';launcher.write_text('#!/bin/sh\ncd "$(dirname "$0")/../Resources" || exit 1\nexec ../MacOS/native_app\n');launcher.chmod(0o755)
parts=a.version.lstrip('v').split('.')
short='.'.join(parts[:3]) if len(parts)>=3 and all(x.isdigit() for x in parts) else '0.1.0'
build='.'.join([parts[0],parts[1],str(int(parts[2])*10+int(parts[3]))]) if len(parts)==4 and all(x.isdigit() for x in parts) else short
with (contents/'Info.plist').open('wb') as f:
    plistlib.dump(dict(CFBundleIdentifier='io.github.dimaystinov.simroki',CFBundleName='SimRoki',CFBundleDisplayName='SimRoki',CFBundleExecutable='SimRoki',CFBundlePackageType='APPL',CFBundleShortVersionString=short,CFBundleVersion=build,LSMinimumSystemVersion='14.0',NSHighResolutionCapable=True),f)
for name in ['robot_config.toml','motion_sequence.json']:
    shutil.copy2(root/name,resources/name)
for name in ['README.md']:
    shutil.copy2(root/name,package/name)
for name in ['docs','motions','planner']:
    shutil.copytree(root/name,package/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
sdk=package/'python-sdk';sdk.mkdir()
for name in ['robot_sim','examples']:
    shutil.copytree(root/'python-sdk'/name,sdk/name,ignore=shutil.ignore_patterns('_native','__pycache__','*.pyc'))
for name in ['README.md','PROTOCOL.md','pyproject.toml']:
    shutil.copy2(root/'python-sdk'/name,sdk/name)
(sdk/'robot_sim/_native').mkdir();shutil.copy2(root/'target/release/libsim_core.dylib',sdk/'robot_sim/_native/libsim_core.dylib')
(package/'START-HERE.txt').write_text('SimRoki for macOS 14+ / Apple Silicon\n\nExtract the ZIP and open SimRoki.app. No Rust or Python installation is needed.\nRABBIT planned walk starts the planned gait; S stops and R resets.\nThe app and library are not Developer-ID signed or notarized.\nConfiguration is in SimRoki.app/Contents/Resources.\nAPI: http://127.0.0.1:8080\nSee docs/walking-controller.md for speed controls and measured results.\n')
(package/'SOURCE-COMMIT.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True))
archive=package.with_suffix('.zip')
subprocess.run(['ditto','-c','-k','--sequesterRsrc','--keepParent',str(package),str(archive)],check=True)
archive.with_suffix('.zip.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
print(archive)
