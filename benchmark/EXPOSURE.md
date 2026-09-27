# Development exposure and data lineage

This is a static provenance audit for the GemmaDecision-270M v0.4.0 JevBench submission, prepared on 2026-09-27. It reviewed the actual preparation/training code and the previously published audit, selection, and calibration records. It did not run a model or scan the raw training corpus against JevBench.

## Submission disclosure

The project had already evaluated an earlier, materially different v0.2 checkpoint on JevBench's public files. Those results and their limitations were viewed and helped motivate subsequent development. This v0.4.0 submission therefore does **not** claim blind project-level development with respect to the public benchmark. No sealed JevBench cases or answer keys were accessed.

The v0.4.0 encoder and scalar head were selected on the project's own development partition. The scalar temperature was fitted on its separate BANKING77/MultiNLI calibration partition. Both the selected weights and temperature were published before the v0.4.0 JevBench run. No JevBench examples or labels were deliberately imported into these optimization, checkpoint-selection, or calibration paths. A code-path review found no loader that feeds JevBench public files, fixtures, or expected labels into them. This is a provenance finding, **not proof of zero incidental overlap** between public source corpora and JevBench.

An earlier discarded pilot had inspected and used the separate Typed Decisions diagnostic. The previously published v2 disclosure also records inspection of one official When2Call test record for its field schema. The released v2 model restarted from official Gemma with newly initialized heads. V3's joint scalar head was separately initialized from scratch, and v4 starts with official Gemma plus that v3 head; the discarded pilot's weights are not inherited. These prior exposures remain part of the project's development history even though they are not an identified data path into v4.

The claim is limited to the inspected code, recorded inputs, and stated development history. Gemma pretraining overlap, incidental upstream-dataset overlap, semantic paraphrases, and all manual influences are not exhaustively ruled out.

## Direct v4 data

The prepared training pool contains 21,432 rows: 5,999 four-way BANKING77 topic selections, 12,000 three-way MultiNLI evidence relations, 2,000 generated rule decisions, and 1,433 training-only replay rows. The replay consists of 400 HelpSteer2 response preferences, 333 When2Call decisions, 300 PAQ decisions, and 400 recorded agent-action decisions across six environments. The actual run manifest records all 21,432 rows as eligible.

- BANKING77 training comes from the pinned official `train.csv`; the project's new development/calibration/final cases come from official `test.csv`. These become four-way topic tasks with deterministic lexical alternatives, not official 77-way BANKING77 evaluation.
- MultiNLI uses the pinned source training file and only government, slate, telephone, and travel genres. Telephone/travel supply v4 training; slate supplies development/calibration; government supplies final cases. Class balancing and premise-group separation are explicit.
- The original numerical-rule generator emits paired opposite-label decisions. Rule labels follow programmatic predicates; fixed template families differ by project partition. They are not independent real-world cases.
- Replay is allowlisted from the prior **v3 training partition only**. Old development/calibration/final records are read for input/group exclusions, not reused as v4 labeled training rows.

The primary development selection uses 300 banking and 300 NLI cases. Calibration has 150 of each primary family; 60 synthetic cases exist in that partition but do not fit the primary temperature. The final project evaluation was reserved separately. All banking intents and selected NLI genres had previously appeared during project development; these are fresh prompts within known domains, not unseen-domain evaluation.

Source revisions, transformations, and terms are available in the published [data provenance](https://huggingface.co/rajan2k/GemmaDecision-270M/blob/785d530221c990671f29976902540101bb9c7647/DATA_PROVENANCE.md) and [attribution audit](https://huggingface.co/rajan2k/GemmaDecision-270M/blob/785d530221c990671f29976902540101bb9c7647/RELEASE_ATTRIBUTION_AUDIT.md).

## Inherited v3 head lineage

V4 imports v3 scalar-head SHA256 `fd2733b7cf138dad5d5fdc1e2cd7b007dcbb5558c24cffe53c3cf80abcbd9d42`. V3 initialized this head with PyTorch defaults (seed 2704203); it could not load the differently shaped v2 CLM-style heads. V3 kept the official Gemma encoder frozen while learning the new joint scalar head.

V3's input pool combined BANKING77, filtered MultiNLI, HelpSteer2 preference/ordinal tasks, generated rule/priority decisions, and allowlisted v2 replay. That replay inherited PAQ, When2Call, ARC, CommonsenseQA, selected AgentInstruct environments, and LiteCoder terminal demonstrations. V2 in turn copied only PAQ/When2Call from its older preparation tree, plus its explicitly downloaded QA and agent sources. It did not copy the earlier discarded Typed Decisions diagnostic.

V3's joint trainer calls the shared loader for `train.jsonl` and `development.jsonl`; calibration/final are not training inputs. Development includes visibly marked legacy retention cases already exposed in v2, and excludes those cases from its fresh-family primary selection metric. V3 final results later became exposed project evidence; v4 excludes prior v2/v3 groups and normalized inputs from its newly reserved holdouts. This protects the project's new evaluation construction, not JevBench contamination.

## Findings from the code-path review

Reviewed v4 `data_v4.py`, `train.py`, `evaluate_v4.py`, and `modal_job.py`; v3 `data.py`, `train.py`, and `joint.py`; and the relevant v2/earlier replay preparation functions. The local v4 preparation/trainer and v3 utility/joint/trainer files are byte-identical to the published source tree at [087497f7bea63dbaf81fb105bc191a52a8f51712](https://github.com/rjn32s/GemmaDecision-270M/tree/087497f7bea63dbaf81fb105bc191a52a8f51712).

The v4 preparation function reads pinned public source files plus exact historical split filenames. It does not glob every workspace JSONL, import the benchmark runner, or consume benchmark result files. Its imports of v3 utilities and v2 schema helpers have no data-loading side effects. Benchmark scripts coexisting in the repository do not constitute an input to this preparation path.

The v4 trainer opens only audited training/development files and checks their hashes; it does not open calibration/final. Checkpoint 3,250 was selected by equal primary-family development accuracy with NLL as tie-breaker. Training ended at 4,250 after early stopping. The evaluator then freezes model/source/data hashes, fits temperature, and opens project final labels. None of those code paths imports JevBench.

This does not rule out a JevBench-like passage already existing in PAQ, NLI, source QA, agent histories, or another upstream dataset. It also does not establish every historical input's contents by independently reconstructing the full dataset. Raw split files are not included in the released source tree, so this static review cannot directly certify a raw-text overlap count.

## Evidence bound before the v4 JevBench run

| Artifact | Value |
|---|---|
| Released HF snapshot | `785d530221c990671f29976902540101bb9c7647` |
| Selected encoder SHA256 | `d7a3e291bfdfa7cd85b33a8a99ef81a4a7d3192e46c77253f3daf14dfd7d6b95` |
| Selected scalar head SHA256 | `72ec4e7d1f0908ad684eeae250f0f70128aa4342b46174bf92a3c9851c5dd965` |
| v4 preparation source SHA256 | `9dbd9c6a73fbcd85dd8bfeb6ca072ac3b5d087712dec903b7aeeab5facbe087a` |
| Training JSONL SHA256 | `89779f75c07987c73f8b5a0db876aa4e5aee687d29e0e4f5cd7e551e3ad45876` |
| Development JSONL SHA256 | `26c7628d09cd1f44d5a77a10130f9e2ede63abc88b286c33c6ad6a9b98436a51` |
| Calibration JSONL SHA256 | `9db35f07cc3176168ef90ce489a8e47b04f2ef2781caf0165e4d3d4f799f24f2` |
| Published calibration artifact SHA256 | `3988b0b74e33ba4ccde8b2761ce6c4c9b3fd5fedf78cf2c52666565a40180c67` |
| Selected temperature | `4.136820402388508` |

The temperature minimizes equal-primary-family NLL over the project's 300 primary calibration cases, from a 121-point log-spaced grid spanning 0.1–10. It changes confidence, not ranking. Its reuse for JevBench is a transfer of an already frozen calibrator; it does not establish calibration on new benchmark domains.

The historical recipe's `benchmark_issue_submission_authorized: false` records the hold in effect when the model was frozen. The later user request authorizes benchmark preparation/submission; the historical record should remain unchanged.

## Existing exclusions versus an additional overlap audit

The published data audit verifies exact project group/state separation and a bounded approximate lexical screen against prior project inputs. It reports zero prior-holdout training conflicts and zero prior-exposure conflicts in new holdouts. Its near-duplicate method uses state-only five-token shingles, bottom-512 sketches, 16 two-hash bands, and a Jaccard threshold of 0.85. Old v2 inputs receive exact group/normalized-state checks only; generated templates are intentionally correlated and excluded from lexical checks. **This is not a JevBench overlap audit.**

If an additional descriptive overlap check is performed, it can run without a model on the existing data volume: pin the 231 public benchmark records; normalize NFKC/case/whitespace; compare full state, full instruction, and state-plus-instruction hashes against actual v4 train/development/calibration and v3 head train/development inputs; then report shared 13-word spans separately. Cover candidate texts as separate fields, retain source/split counts and hashes, and distinguish generic shared labels/instructions from copied task content. Use only benchmark input fields, never expected labels, for that comparison. Publish aggregate findings and hashes, not copied source records. Such an audit should remain descriptive for this frozen release, not trigger answer-dependent edits or further selection.

No such full-corpus JevBench overlap scan was executed by this audit. Even zero exact/13-gram matches would not prove semantic independence or unknown base-pretraining absence.
