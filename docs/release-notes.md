RABBIT-based walking, higher steps and faster forward movement.

- Wide standing pose; both knees bend in the same anatomical direction.
- Five-link dynamics, impact/relabeling and constrained offline gait optimization using SimRoki's own parameters.
- State-phased motor control: the reference advances faster as the body moves faster.
- Forward torso lean, contact-driven leg changes, startup from rest and double-support stop.
- New higher-step profile: native tests measured approximately 0.23 m/s, 20 cm steps and 3 cm foot lift after settling. Nominal planner speed is not actual speed.
- Native RABBIT button and API with nominal profiles from 0.3 to 0.6 m/s.
- Primary RABBIT references, model derivation, reproducible validation and numerical return-map analysis included.

Downloads:
- **Windows x64:** extract `SimRoki-windows-x64.zip` and run `Start-SimRoki.cmd`.
- **macOS 14+ / Apple Silicon:** extract `SimRoki-macos-arm64.zip` and open `SimRoki.app`.
- SHA-256 checksum files accompany both archives.

Neither Rust nor Python is required for the desktop app. Python is optional for the SDK and offline planner. Press **S** to stop and **R** to reset; use **RABBIT planned walk** to start the planned controller. See `docs/walking-controller.md` for controls and speed API examples.

Validation includes release-mode Rust tests on both platforms and a smoke test of each packaged native library. The macOS native window was also exercised locally. Windows interactive graphics were not manually tested. Builds are not Developer-ID/code-signing certified or notarized.

Known limits: flat-ground forward walking at 120 Hz; actual step length and speed depend on the contact model and startup state. Immediate startup can converge to a shorter-step gait. Numerical point-foot stability is not a hardware or Rapier stability guarantee.
