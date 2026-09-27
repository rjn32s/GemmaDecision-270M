# GemmaDecision-270M v4 source

Code and aggregate evidence for a fully tuned Gemma-270M candidate ranker. The modified model and inference helpers are distributed at [HF rajan2k/GemmaDecision-270M](https://huggingface.co/rajan2k/GemmaDecision-270M/tree/v0.4.0) under Gemma terms. This GitHub source repository uses Apache-2.0 for code and contains no model weights, datasets, credentials, caches or optimizer state.

See [RESULTS.md](RESULTS.md) for the actual fresh-prompt test, both baselines, uncertainty, failed targets and exposed illustrative examples. No official JevBench result or JEV equivalence is claimed.

The current implementation is in `outputs/gemma-local/v4`; v2/v3 sibling modules are preserved as historical dependencies. V4 learns a context-conditioned scalar score with the entire encoder trainable. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) describes exact initialization, required historical split inputs, Modal setup and source/data hashes. The repository alone does not recreate those historical inputs, so there is no claimed one-command exact retraining workflow.

Use the model repository's CPU quickstart for inference. Install `requirements.txt` here only when preparing the recorded training/evaluation environment. Review and adapt project-specific Modal volume names and budgets before any cloud execution. Historical benchmark scripts are source history, not an instruction to submit or a v4 benchmark claim.

Attribution and transformations are in [DATA_PROVENANCE.md](DATA_PROVENANCE.md), [DATA_PROTOCOL.md](DATA_PROTOCOL.md) and [RELEASE_ATTRIBUTION_AUDIT.md](RELEASE_ATTRIBUTION_AUDIT.md). The old v3 final is exposed history and must not be reused as an independent test for another tuned version.
