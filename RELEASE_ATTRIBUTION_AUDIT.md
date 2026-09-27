# V4 release attribution audit

Checked 2026-09-27 against the existing v2/v3 bundles, pinned preparation code, and primary source terms. Scope: publish original code and derived model weights; **no raw training datasets**. This audit does not certify model quality or independently settle whether trained weights are adaptations of particular training records.

## Required release corrections

1. **Replace the old NOTICE.** Both existing bundles say the Gemma backbone is unchanged. V4 tunes the backbone and head. Identify `model.safetensors`, the trained head and modified configuration prominently as project modifications; retain their provenance hashes. The package code already records `base_weights_modified=true`.
2. **Keep model and code terms distinct.** HF model metadata should remain `license: gemma`, with base model `google/gemma-3-270m` at `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`. Include the full existing Gemma `LICENSE`, incorporated `GEMMA_PROHIBITED_USE_POLICY.txt`, `NOTICE`, and source model card. Original code may retain Apache-2.0 in `LICENSE_CODE`; do not label the weights or all datasets Apache-2.0. Google's redistribution conditions require the agreement, restriction notice and prominent modification notices. [Gemma terms, Section 3](https://ai.google.dev/gemma/terms)
3. Retain this required NOTICE sentence exactly: “Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms”. State that the included agreement governs the derivative model and its use restrictions. [Gemma terms](https://ai.google.dev/gemma/terms)
4. Describe banking-support **topic routing**, not financial eligibility or other consequential financial decision automation. The incorporated policy restricts automated decisions affecting rights or well-being. [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)
5. Preserve attribution for included `clm_schema.py` and any `clm_heads.py` from [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) revision `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`, with Apache-2.0 text from the existing `CLM_LICENSE`. Do not claim a helper is unchanged if edited. Credit the CLM method separately from the new joint-scoring implementation; neither CLM nor Google endorsement is implied.

## Training attribution to carry forward

The model starts from official Gemma plus the v3 trained scalar head. Consequently, attribution must cover **v3 head-training lineage as well as direct v4 replay**, even when a source has no direct v4 rows. Publish actual source counts from the final preparation audit; quotas are not measured counts.

| Source and pinned revision | Use and source-declared terms |
|---|---|
| [PolyAI BANKING77](https://huggingface.co/datasets/PolyAI/banking77/blob/90d4e2ee5521c04fc1488f065b8b083658768c57/README.md); HF `90d4e2ee5521c04fc1488f065b8b083658768c57`, official CSV repository `57ec275d8078af65b7731c2a98be812d844a6d6b` | New training and reserved evaluation, CC-BY-4.0. Credit PolyAI; disclose four-way conversion, alternatives, filtering and new split construction. |
| [NYU MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli/blob/da70db2af9d09693783c3320c4249840212ee221/README.md); `da70db2af9d09693783c3320c4249840212ee221` | New evidence training/evaluation. Only government, slate, telephone and travel; OANC permissive terms, not a blanket Apache license. Credit Williams, Nangia and Bowman. Fiction is excluded. The [original paper](https://cims.nyu.edu/~sbowman/multinli/paper.pdf) explains licensing differences. |
| [NVIDIA HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2); `990b2711a36180dd19d9c94b8627844866f8982a` | Training replay and v3 head lineage; CC-BY-4.0. Preserve preference ties and disclose reformatted ranking task. |
| [NVIDIA When2Call](https://huggingface.co/datasets/nvidia/When2Call); `0582f7749df63a96fdc3070932e83e72396ace53` | Training replay and lineage; CC-BY-4.0. Distinguish training examples from official benchmark results. |
| [sentence-transformers/paq](https://huggingface.co/datasets/sentence-transformers/paq); `74601d8d731019bc9c627ffc4271cdd640e1e748` | Training replay and lineage. PAQ **data** is CC-BY-SA; the upstream repository's CC-BY-NC **code** license is separate. We use data, not upstream PAQ code. Credit Lewis et al. and the source data. [Upstream licensing distinction](https://github.com/facebookresearch/PAQ#license) |
| [neulab/agent-data-collection](https://huggingface.co/datasets/neulab/agent-data-collection); `31a76bfb0124d77ae7322eabbb0171bf11ee2c67` | Selected AgentInstruct environments: Apache-2.0 per the previously downloaded subset LICENSE files and v2 audit. Do not extend this statement to the entire collection. |
| [Lite-Coder/LiteCoder-Terminal-SFT](https://huggingface.co/datasets/Lite-Coder/LiteCoder-Terminal-SFT); `6acdbbdb29979e4b8ea717b12accc8214606d087` | Selected recorded actions in replay/head lineage; MIT. Credit Lite-Coder. |
| [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc); `210d026faf9955653af8916fad021475a3f00453` | V3 head lineage; CC-BY-SA-4.0 according to the existing v2 provenance audit. |
| [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa); `94630fe30dad47192a8546eb75f094926d47e155` | V3 head lineage; MIT according to the existing v2 provenance audit. |
| Original counterfactual-rule generator | Apache-2.0 code, project-generated labels; disclose generated status and changes rather than attributing it to a real-world dataset. |

## Publication contents and wording

Include `DATA_PROVENANCE.md` with this source lineage, source links, pinned revisions, terms, actual preparation counts and the transformations. Keep dataset terms attached to their respective sources; do not infer universal redistribution or relicensing rights from the public HF download.

Publish model/code, original examples, aggregate evaluation reports, hashes and label-free reproducibility manifests. Exclude raw source downloads, private split JSONL, credentials, training caches and optimizer states. If subsequently releasing a derived dataset, review and attach the corresponding source terms separately rather than assuming this code/model release covers it.

A code-only GitHub repository may use Apache-2.0 for original/vendored Apache code, with attribution and a clear link to the separately governed Gemma model. HF carries Gemma model metadata and terms. Use the authorized publisher identities **GitHub `rjn32s`** and **HF `rajan2k`**.

Existing copies of `LICENSE`, `LICENSE_CODE`, `GEMMA_PROHIBITED_USE_POLICY.txt` and `BASE_MODEL_CARD.md` are available in `outputs/GemmaDecision-270M-local`. Copying them does not resolve the stale NOTICE wording; write the v4 modification notice explicitly. No existing release files were changed by this audit.
