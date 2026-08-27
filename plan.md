# KARN plan

## Status
- Git history is preserved; no destructive reset/rebase was used.
- No active rebase remains; stale rebase metadata was already cleared.
- Current repository is not ready for push because the working tree has local modifications and the generated training-window tensors do not yet align with the current processed label state.

## Immediate next steps
1. Review and decide whether the modified raw catalog and validation artifacts should be kept or discarded.
2. Rebuild/verify the generated split tensors in `data/processed/training_windows` from the current processed label generation logic.
3. Re-run `python ml/validate_pipeline.py` after the tensor rebuild.
4. Only push after the repo is clean and validation passes without scientific over-claiming.

## Safety guardrails
- Never modify raw FITS or telemetry files.
- Keep Git history intact.
- Do not claim scientific validity for prototype or synthetic labels.
- Keep training CPU-safe and lightweight.
