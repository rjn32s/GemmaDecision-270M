# V4 data protocol

This dataset targets **four-option banking intent routing and three-option evidence relations**. It is a new private prompt-group evaluation, not an official BANKING77, MultiNLI, or JevBench submission. Protocol: `gemmadecision-v4-fresh-prompts-1`; seed: `2704404`.

Preparation runs on Modal CPU through `data_v4.prepare()`, using the cached tokenizer, not model weights. The defaults are `/experiment/work/decision-v4/data`, prior v3 data at `/experiment/work/decision-v3/data`, and legacy data at `/experiment/work/decision-v2`. The launcher mounts v3 `data.py` as `data_v3.py` for pinned download and serialization helpers. Importing the module has no download or training side effects.

## Fixed splits

| Split | Banking prompts | NLI premises | Synthetic cases |
|---|---:|---:|---:|
| Training target | 6,000 | 12,000 | 2,000 |
| Development | 300 | 300 | 100 |
| Calibration | 150 | 150 | 60 |
| Final | 600 | 600 | 200 |

Up to 1,500 previous **training-only** examples are replayed: 400 HelpSteer2 preferences, 400 When2Call cases, 300 PAQ cases, and 400 recorded agent actions. Available groups and conservative duplicate filtering can reduce these counts; `audit.json` records actual counts. Training rows near previous holdouts are removed without replacement after selection. The final split must contain at least 1,000 distinct real prompt groups before preparation succeeds.

The two primary families are `intent_routing` and `evidence_relation`; report their equal-family mean, both individual accuracies, coverage and calibration. Synthetic `rule_compliance` and replay families are separate diagnostics. Synthetic pairs share a prompt group, and template counts must accompany claims about generalization.

## Sources and labels

**BANKING77:** [PolyAI's source](https://github.com/PolyAI-LDN/task-specific-datasets) at commit `57ec275d8078af65b7731c2a98be812d844a6d6b`, CC-BY-4.0. Source `train.csv` supplies training; the previously unused official `test.csv` supplies new development/calibration/final cases using a fixed prompt hash. All 77 known labels are eligible. Each query has its gold topic and three alternatives with greatest label-word overlap, with hash tie breaks. Candidates are independently shuffled. This constructed four-option task differs from official 77-way classification and does not establish arbitrary tool-routing quality.

**MultiNLI:** [nyu-mll/multi_nli](https://huggingface.co/datasets/nyu-mll/multi_nli) at `da70db2af9d09693783c3320c4249840212ee221`; cached training parquet SHA256 `1c1de03640b168e410aabfca19e7cc2f3dfcd7f0e126e935674e56fb102c4529`. Only government, slate, telephone and travel nonfiction are used under the permissive OANC terms described in the [source paper](https://cims.nyu.edu/~sbowman/multinli/paper.pdf). Fiction and ANLI are excluded. Telephone/travel supply training; slate supplies new development and calibration; government supplies new final cases. Normalize the premise to define a group; keep one hypothesis per selected premise. Cross-partition premises are removed before sampling. NLI is class balanced, with entailment/support, neutral/unresolved and contradiction kept distinct.

**Counterfactual rules:** original Apache-2.0 generator produces paired requests with opposite mechanically checked outcomes. Change one qualifying amount or one flag while retaining the policy. Both members stay in a single group and split. Training, development, calibration and final use disjoint policy templates. These are correlated synthetic diagnostics, not independent natural requests; template transfer claims remain limited by the small template count.

**Replay:** reuse only v3 `train.jsonl`; source provenance remains in its audit and the v2 audit. Never borrow options or gold labels from old development, calibration or final. Preference ties retain soft labels. Source actions are recorded alternatives, not live agent success.

## Exposure and duplicate controls

All prior v3 train/development/calibration/final groups and all available v2 train/validation/test/development/calibration/final groups are excluded from **new holdouts**. Their normalized input hashes are also excluded. Old nontraining groups and input hashes are excluded from **new training**. Loading prior holdouts retains only group/input metadata; their labels never supply training supervision. The old v3 final set is now exposed regression evidence only.

New holdouts additionally undergo a state-only lexical screen against all v3 inputs and previously selected new holdouts. New training is screened against all new holdouts and old v3 holdouts. The screen uses normalized five-token shingles, bottom-512 sketches, 16 two-hash retrieval bands, and a 0.85 Jaccard threshold. Oversized retrieval buckets are counted and skipped. All-v2 exclusion is exact group/state hashing, not a full lexical audit. Candidate retrieval is approximate, so a clean screen does not prove semantic independence.

These are **fresh prompts with known workflows**: the v3 process already exposed all banking intents and the chosen NLI genres. The corpus may also have appeared in foundation-model pretraining. Do not describe this as unseen-domain or guaranteed pretraining-independent evaluation.

## Saved files and runtime contract

`train.jsonl`, `development.jsonl`, `calibration.jsonl`, and `final.jsonl` preserve the v3 schema: state, question, candidate strings, IDs/groups, source/family/category, metric and target or target probabilities. Model inputs are **only** state, question and candidate strings. Gold and provenance metadata are never rendered.

`within_pilot_limits` checks 2,048 exact `common.schema_state(state, question)` tokens and 768 tokens per candidate. It also checks the complete `schema_state + "\n\nCandidate action:\n" + candidate` input against the encoder's configured context limit. This includes the serving renderer's JSON-state handling; previous raw and legacy rendered hash variants remain protected. Training drops overlength rows. Holdout selection retains them so rejected requests count against coverage and accuracy. There is no silent truncation.

`split-manifest.jsonl` contains group/provenance identities without input text or gold. `audit.json` records source/code/partition hashes, counts, filtering, limitations and gates: `passed`, `passed_structural`, `prior_holdout_training_conflicts=0`, and `new_holdout_prior_exposure_conflicts=0`. A completed partition set is reused only if its saved hashes match; preparation refuses to overwrite altered frozen files. Final data is constructed and validated before training but remains unopened for performance or model selection until a frozen evaluation recipe is committed.

Local checks cover synthetic label oracles, group-preserving pairs, schema, prior-exposure exclusions and the lexical index using tiny fabricated fixtures. Complete data preparation and tokenization run only on Modal.
