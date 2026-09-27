# GPU choice: H100 for the next GemmaDecision training pilot

Measured on Modal on 27 September 2026. H100 is the preferred GPU for the next
full-backbone training run: it completed the fixed training workload 2.73x faster
than A100 80 GB. At current configured rates, its measured update compute cost
was 44.7% lower. This is a short workload profile, not a prediction of accuracy
or guaranteed end-to-end training speed.

| GPU | Full profile | Non-padding training tokens/second | Configured cost/hour | Peak allocated VRAM |
|---|---|---:|---:|---:|
| NVIDIA L40S | Out of memory on long batch | Not comparable | $2.30 | 42.62 GiB before failure |
| NVIDIA A100-SXM4-80GB | Complete | 29,255 | $2.85 | 44.22 GiB |
| NVIDIA H100 80GB HBM3 | Complete | **79,828** | $4.30 | 44.29 GiB |

Both completed runs used the same model initialization, head seed, source code,
training-only examples, candidate labels and padded batches. The workload hash
was `d462403ca3e29c9b7ad5fedb56e7cb324e52e0b9df54c8ad46ec3f9b21e3510a`.
Each update trained all 268,098,176 backbone parameters plus the 329,985-parameter
head, using FP32 parameters and AdamW moments, BF16 autocast, fused AdamW, and no
activation checkpointing. Actual encoder gradients and weight changes passed.

Each of three batches contained four prompt groups, with all their candidate
pairs processed together. The padded shapes were 16x192, 12x641, and 16x2318
tokens. Three warmup updates preceded nine measured updates, covering each batch
three times. CUDA synchronization bounded each forward/loss/backward/gradient
clipping/optimizer interval. The measured interval excluded model loading,
tokenization, batch staging and zero_grad. Total measured time was 1.611 seconds
on H100 and 4.396 seconds on A100, for 128,604 non-padding tokens on each device.
That small sample establishes a practical initial choice, not a precise sustained
throughput forecast; confirm it on the actual next-round length mix.

L40S completed the short and medium warmup updates but exhausted memory on the
long batch. Smaller microbatches or activation checkpointing could make it viable;
this result does not imply that a 270M model generally requires an 80 GB GPU.
We did not reduce its workload silently or invent a complete speed comparison.

Configured prices include two CPU cores and 32 GiB host RAM. H100 needs 1.51x
A100 throughput to break even; this test measured 2.73x. Estimated compute for
the measured GPU updates is $0.01496 versus $0.02705 per million non-padding
training tokens. Those figures omit preparation, loading, validation, checkpoint
I/O and other lifecycle costs. Prices: [Modal](https://modal.com/pricing).

All three function-body estimates total approximately $0.0975. The launcher
retained $0.9486 of worst-case reservations against the $1 comparison ceiling,
including startup/teardown margin. This remains inside the next-round $20 and
original-project $30 ceilings. Actual workspace billing can reconcile later.

No development, calibration or final examples were used. The disposable updated
weights were discarded; this was not the main quality-training run. Existing
releases and publication/submission holds remain unchanged. Per-device JSON
reports and `GPU_CHOICE.json` retain the measurements and checks.
