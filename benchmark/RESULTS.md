# JevBench public measurement — GemmaDecision-270M v0.4.0

This is an author-run diagnostic on the **231 published JevBench cases**. It is
not an official score, rank, or a measurement on JevBench's sealed test.

The frozen model answered **110/231 correctly
(47.62%)**. Attempted: 231;
valid distributions: 195; unattempted: 0.

| Public tier | Correct / planned | Accuracy | Attempted | Valid |
|---|---:|---:|---:|---:|
| easy | 43/48 | 89.58% | 48 | 48 |
| original | 39/72 | 54.17% | 72 | 72 |
| hard | 28/111 | 25.23% | 111 | 75 |

All 36 invalid outputs were explicit HTTP 422 refusals for state/question inputs
over the published 2,048-token limit: 19 long-policy cases and 17 multi-hop cases.
These count as incorrect in the headline result. All 231 cases were attempted;
there were no infrastructure failures or retries. Even among the 195 supported
inputs, 85 predictions were wrong. The context limit is not the only weakness.

These results measure transfer to broader decision tasks. They must not be
substituted for the project's 87.25% equal-family banking/NLI result or read as
JEV parity. The detailed per-family breakdown remains in the summary, including
weak families and refusals.

## Protocol and evidence

- Immutable model: [rajan2k/GemmaDecision-270M@v0.4.0](https://huggingface.co/rajan2k/GemmaDecision-270M/tree/785d530221c990671f29976902540101bb9c7647).
- [Protocol commit published before this run](https://github.com/rjn32s/GemmaDecision-270M/tree/b78e07e04d5a77a1eb1f22b968e44ed4f13c6973/benchmark).
- JevBench revision: `d06ee95988da1350eb8ae5511daa0e06bfffe911`.
- [Reproduction instructions](README.md), [fixed protocol](PROTOCOL.json),
  [development-exposure disclosure](EXPOSURE.md).
- [Full public summary](results/jevbench-v4-public-01/public/summary.json),
  [per-item records](results/jevbench-v4-public-01/public/records.jsonl),
  [run environment](results/jevbench-v4-public-01/environment.json).

No model/temperature selection, retries or prompt changes followed this run.
The earlier v0.2 public JevBench evaluation was already exposed during project
development. V0.4.0 weights and its external temperature were fixed beforehand;
no sealed cases were accessed. The static lineage review found no intentional
JevBench data import into training, but does not prove zero incidental overlap.

The unchanged upstream Runner and TypeSafe adapter made one serial HTTP attempt
per item. HTTP 422 input refusals count as incorrect; upstream outage stops
remain enabled. Strict validity, renormalization counts, calibration denominators,
ordinal MAE and all unattempted IDs are retained in the full summary.

## Probability and timing disclosure

The server applies `softmax(raw_rank_scores / 4.136820402388508)`. This is a fixed,
adapter-derived distribution using the project's earlier 300-case calibration
set. Upstream records call native-protocol responses `probs_source=native`; the
underlying ranker has a scalar head and does not natively return probabilities.

Hardware: NVIDIA H100 80GB HBM3, two reserved CPU cores and 32 GiB RAM. BF16 encoder,
FP32 scalar head. Serial loopback HTTP, no inference warmup; first-call latency
is retained. Model download/load occurs outside decision latency. Full latency
statistics (seconds): `{"n": 231, "p50_s": 0.06307101599999854, "p95_s": 0.27852970500000396}`. These are measured
local-container timings, not a production service latency guarantee.

Multiclass Brier mean: 0.508187 over 195 valid
scorable distributions. ECE and bin counts are in the full summary. Calibration
on these public domains is a measurement, not an assumed property of the reused
temperature. Ordinal MAE uses continuous expected value; ordinal accuracy uses
the pinned scorer's argmax rule.

## Cost and attribution

Estimated benchmark function compute: **$0.0565**
for 47.28 seconds of GPU plus reserved CPU/RAM.
This excludes startup, storage, transfer and any taxes. The full conservative
pre-run reservation was $1.80; the workspace meter may reconcile separately.
GemmaDecision has no advertised hosted tariff, so API cost per decision is
unknown/null. The harness's $0.02-per-attempt accounting reservations are not
actual spending or a per-decision price.

Published raw records contain only the public JevBench tasks and responses from
this run. Original JevBench public task text and harness are attributed to
[fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench/tree/d06ee95988da1350eb8ae5511daa0e06bfffe911)
under its MIT license, included as `JEVBENCH_LICENSE.txt`. Private training data,
credentials, sealed cases and sealed predictions are not redistributed.
