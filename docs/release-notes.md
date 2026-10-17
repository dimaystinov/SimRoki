# Automatic walking restart and music

- Start/stop automatic walking from the top of the right control panel.
- Ten consecutive start/stop cycles at approximately 1 m/s verified without falls.
- Restart preparation preserves physical position and applies joint torques only.
- Bundled GHOSTFACE PLAYA — Why Not, with a separate looping music player.
- Music pause preserves playback position; resume continues from that point.
- Music is embedded, works offline, and starts paused.

RABBIT walking at approximately 1 m/s, higher steps and collapsible panels.

- Wide standing pose; both knees bend in the same anatomical direction.
- Five-link dynamics, impact/relabeling and constrained offline gait optimization using SimRoki's own parameters.
- State-phased motor control: the reference advances faster as the body moves faster.
- Forward torso lean, contact-driven leg changes and startup from rest. Fast-mode braking takes about 12–15 seconds before the double-support stop.
- New fast profile: native tests measured approximately 0.97–0.99 m/s, 27 cm steps, 5.5 cm foot lift and 14° forward lean. Nominal planner speed is not actual speed.
- Independent collapse/expand buttons for both panels; Robot mode has no solid background.
- Native RABBIT button and API with nominal profiles from 0.3 to 1.0 m/s.
- Primary RABBIT references, model derivation, reproducible validation and numerical return-map analysis included.
- Cross-platform contact math and IK use deterministic library functions; startup clears the standing servo integral, with gains recalibrated for this physics mode.

Downloads:
- **Windows x64:** extract `SimRoki-windows-x64.zip` and run `Start-SimRoki.cmd`.
- **macOS 14+ / Apple Silicon:** extract `SimRoki-macos-arm64.zip` and open `SimRoki.app`.
- SHA-256 checksum files accompany both archives.

Neither Rust nor Python is required for the desktop app. Python is optional for the SDK and offline planner. Press **S** to stop and **R** to reset; use **RABBIT planned walk** for the slow profile or **Start automatic walk ~1 m/s** for the fast profile. See `docs/walking-controller.md` for controls and speed API examples.

Validation includes release-mode Rust tests on both platforms and a smoke test of each packaged native library. The macOS native window was also exercised locally. Windows interactive graphics were not manually tested. Builds are not Developer-ID/code-signing certified or notarized.

Known limits: flat-ground forward walking at 120 Hz; actual step length and speed depend on the contact model and startup state. Immediate startup can converge to a shorter-step gait. Numerical point-foot stability is not a hardware or Rapier stability guarantee.
