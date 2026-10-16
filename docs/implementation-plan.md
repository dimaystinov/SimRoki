# RABBIT implementation status

Implemented and checked:

- [x] Wide initial stance and a working IK/PD reference with measured 22–23 cm steps.
- [x] Primary RABBIT papers, HZD theory, book and five-link executable example reviewed and linked.
- [x] Forward/inverse kinematics, independent mass/inertia and bias-force derivation from SimRoki parameters.
- [x] Unactuated dynamics, ground reaction/friction, inelastic impact and leg relabeling.
- [x] Offline constrained periodic motor-trajectory optimization and independent dense/integration checks.
- [x] Rust virtual-constraint controller with hip-progress phase, bounded internal torques and actual-contact leg switching.
- [x] Startup from rest, gradual reference changes at touchdown, and double-support stop.
- [x] Native button, API, phase/controller/contact/torque telemetry and explicit orbit-reset experiment.
- [x] Six symbolic-model tests, seven Rust controller tests, four long IK trials, four planned start/stop trials and speed-reference transitions.
- [x] Forty perturbed point-foot steps and a refined numerical Poincare calculation (spectral radius about 0.818).
- [x] macOS release build and visible native execution of the planned gait.

Remaining performance limits (not hidden by passing numerical-model tests):

- The ideal optimized step is 20 cm. Rapier's finite-width contacts produce
  approximately 17–18 cm steps in the default planned mode. Faster nominal
  settings may shorten them further. The IK reference achieves 22–23 cm.
- Nominal/requested speed is not yet a calibrated command for actual speed in
  either controller. The planned phase does advance faster when the body moves
  faster, and reference changes are implemented, but exact speed regulation is
  still unresolved.
- The return-map result is local numerical evidence for point feet. It is not
  a formal HZD invariance/stability proof for Rapier contacts.
- Validation covers forward flat-ground walking at 120 Hz with the checked-in
  parameters. Reverse motion, obstacles, arbitrary dynamics and hardware are
  not claimed as validated.

See [measured results and controls](walking-controller.md),
[model and sources](rabbit-model.md), and [planner commands](../planner/README.md).

The faster high-step profile now measures approximately 0.232–0.234 m/s, 19.9 cm steps and 3 cm lift after settling; see the latest section in walking-controller.md.
