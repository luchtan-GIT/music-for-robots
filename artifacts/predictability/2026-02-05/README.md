# Predictability Probe Artifacts (2026-02-05)

These artifacts accompany the "predictability probe" work in `toolchains/listening-protocols/`.

## What’s here

- `plots/`
  - `*_timeline.png`: surprise (next-step prediction error) vs time for a single pass.
  - `*_learning_curve.png`: per-loop mean error across 5 full listens (L1..L5).

- `corrupted_variable/` and `invalid_pointer/`
  - `*_surprise_mel.mp4`: a timeline video with a moving cursor; audio muxed.
  - `mirror_residue_pred_mel.mp4`: Mirror Residue protocol video with the flare driven primarily by online mel next-step surprise; audio muxed.
  - `predictability_L1_report.md`, `predictability_L5_report.md`: numerical summaries.

## Notes

- Errors are RMSE in online-standardized feature space.
- Two feature spaces are used:
  - embedding ("state")
  - mel-dB ("surface")

