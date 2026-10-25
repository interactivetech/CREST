# CREST Stage 3: ReClor preference learning

## Summary

Run stage 3 DPO on the existing ReClor stage 1 preference data, starting from the selected stage 2 adapter at checkpoint 17,300. Select among four DPO rounds using second-pass validation accuracy, generate test predictions once from the selected adapter, and upload that adapter to a new private Hugging Face repo.

## Implementation Changes

- **Build preference data:** Read stage 1 high/low-quality rationales and consistency scores from scratch. Fix the reversed template arguments so each prompt contains the instruction, passage, question, and answer choices. Check that generated `only_z` and `only_tz` files are nonempty and contain valid prompt/chosen/rejected fields.
- **Update DPO for the installed stack:** Use the scratch environment's TRL 1.16 / Transformers 5.19 APIs: pass the tokenizer as `processing_class`, put `beta` in `DPOConfig`, and use TRL's PEFT reference-adapter handling. Retain the 512-token prompt budget and 1,024-token total limit.
- **Train and select:** Add a scratch-backed runner with W&B offline. Start from the local stage 2 checkpoint and keep the repository recipe: four rounds of 3,000 steps, lambda 0.6, learning rate `6e-6`, per-device batch size 8, and beta 0.1. Save each round's adapter and logs under `$CREST_SCRATCH_ROOT/models/ReClor/M_CREST/reclor_b300_dpo` and `$CREST_SCRATCH_ROOT/logs/reclor_b300_dpo`. Validate every round; select the highest second-pass validation accuracy, breaking ties in favor of the earlier round.
- **Finish and publish:** Run inference on the held-out test set once, using only the selected adapter. Save predictions without reporting test accuracy because the provided test labels are null. Upload the selected adapter, tokenizer, and model card to a new private repo named `mendeza/reclor-b300-crest-llama3-8b-dpo`, using a rotated HF token without saving it. Commit and push the stage 3 code changes to `origin/main`.

## Verification

- Check pair-data prompts and completions on representative examples, confirm both datasets have rows, and record their counts.
- Run a one-step DPO smoke using a capped sample from both preference datasets; confirm finite loss, a saved adapter, and successful adapter reload and one-example validation inference.
- Confirm four full-round adapters and validation metrics are saved; verify checkpoint selection follows the specified metric and tie-break.
- Confirm the selected adapter produces test predictions once, with no test accuracy claimed, and verify the private HF repo contains the adapter files and model card.

## Assumptions

- Stage 1 scratch outputs and the stage 2 checkpoint at step 17,300 remain available.
- “Repository defaults” means the stage 3 runner's four rounds, 3,000 steps per round, lambda 0.6, and learning rate `6e-6`; the installed TRL API changes will preserve the DPO reference behavior across rounds.
- The HF token will be rotated before upload; the repo remains private unless changed later.
