# Measured results

This is a private fresh-prompt project evaluation. It is not an official BANKING77, MultiNLI or JevBench score.

| Task | Original Gemma likelihood | Frozen v3 joint ranker | GemmaDecision v4 | Cases |
|---|---:|---:|---:|---:|
| Four-way banking topic selection | 47.17% | 64.83% | 96.50% | 600 |
| Three-way NLI evidence relation | 33.33% | 36.67% | 78.00% | 600 |
| Equal-family primary mean | 40.25% | 50.75% | 87.25% | — |

The final set contains 1,200 primary prompt groups. Rejections stay in accuracy denominators. All-population coverage for v4 is 100.00%; per-family coverage and calibrated NLL/Brier are in `evaluation.json`.

| Paired comparison against stronger baseline | Accuracy difference | 95% interval |
|---|---:|---:|
| Four-way banking topic selection | +31.67 pp | [+27.83 pp, +35.67 pp] |
| Three-way NLI evidence relation | +41.33 pp | [+36.00 pp, +46.00 pp] |
| Equal-family mean | +36.50 pp | [+33.21 pp, +39.83 pp] |

Intervals use 1,500 paired prompt-group bootstrap resamples within each primary family. The stronger of the original-Gemma and frozen-v3 baselines is recomputed within every resample. They quantify uncertainty for this constructed population, not arbitrary real-world decisions.

| Frozen release target | Accuracy target | Observed | All checks |
|---|---:|---:|---|
| Four-way banking topic selection | 85.00% | 96.50% | Met |
| Three-way NLI evidence relation | 70.00% | 78.00% | Met |

All checks combine the accuracy target, coverage requirement and positive lower paired interval against the stronger baseline. These thresholds were fixed before this final evaluation; they are not promises of deployment quality.

Capabilities satisfying those checks: Four-way banking topic selection, Three-way NLI evidence relation.
Release status from this protocol: **targets_met**. The repository remains a research model with the documented limits.

Synthetic challenge performance is separate from the primary mean:

| System | Challenge | Accuracy | Cases |
|---|---|---:|---:|
| GemmaDecision v4 | rule_compliance | 93.00% | 200 |
| Original Gemma likelihood | rule_compliance | 50.00% | 200 |
| Frozen v3 joint ranker | rule_compliance | 50.00% | 200 |

Synthetic cases come from a small number of policy templates and include correlated pairs. Their case count is not a count of independent workflows.

The selected run's best checkpoint is update 3,250; its completed run reached 4,250 updates. Selection used development accuracy with NLL tie-breaking. Calibration used a separate partition and does not change candidate ordering. Final outcomes did not select the checkpoint.

Development seeds recorded for final evaluation: [2704206]. Repeatability established: False. Different seed runs, when included in evidence, are development diagnostics rather than proof of replicated final gains.

Offline CPU verification passed software checks for loading with no network, independent reload, candidate reversal/addition invariance and explicit overlength rejection. These checks do not assert correct decisions. The following examples were already exposed during development and are not fresh benchmarks:

| Illustrative example | Actual top choice | Expected choice | Match | CPU request seconds |
|---|---|---|---|---:|
| support_routing | payments | payments | Yes | 0.516 |
| evidence_relation | contradicted | contradicted | Yes | 0.391 |
| tool_choice | weather | weather | Yes | 0.440 |

Cold model load: 3.204 seconds on the recorded Modal CPU worker. Latencies describe two CPU threads and these short requests; they are not measurements from the user's Mac. MPS and CUDA-to-CPU ranking equivalence are not certified by this CPU test.
