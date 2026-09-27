# Data provenance

The fully tuned model starts from official Gemma and the previously trained v3 joint scalar head. Attribution covers direct v4 data and that head's earlier training lineage. Raw datasets and private split files are not redistributed.

Actual prepared counts are below; optimization uses the eligible rows recorded in the training manifest. These are measured counts, not requested quotas.

| Split | Cases | Prompt groups | Real primary prompt groups |
|---|---:|---:|---:|
| calibration | 360 | 330 | 300 |
| development | 700 | 650 | 600 |
| final | 1,400 | 1,300 | 1,200 |
| train | 21,432 | 20,432 | 17,999 |

| Prepared training family | Cases |
|---|---:|
| evidence_relation | 12,000 |
| intent_routing | 5,999 |
| response_preference | 400 |
| retention_agent_alfworld | 70 |
| retention_agent_db | 85 |
| retention_agent_kg | 1 |
| retention_agent_os | 82 |
| retention_agent_terminal | 75 |
| retention_agent_webshop | 87 |
| retention_paq | 300 |
| retention_when2call | 333 |
| rule_compliance | 2,000 |

The selected training run records 21,432 eligible training rows.

BANKING77 examples become four-way topic choices with lexical hard alternatives and shuffled candidate order. MultiNLI examples become support/unresolved/contradiction choices, preserving premise groups and class balancing. Counterfactual policy pairs have mechanically checked labels and remain together by group; held-out templates are correlated diagnostics. Replay uses previous training partitions only, including soft preference labels.

New holdouts exclude prior v2/v3 prompt groups and normalized inputs. Approximate lexical screens provide additional protection, without proving semantic independence. All banking intents and the selected NLI genres were previously encountered during development; these are fresh prompts within known domains. Public pretraining overlap is unknown.

Source licenses, authors, revisions, transformations and inherited sources are listed in [RELEASE_ATTRIBUTION_AUDIT.md](RELEASE_ATTRIBUTION_AUDIT.md). The precise split protocol is in [DATA_PROTOCOL.md](DATA_PROTOCOL.md), with measured audit evidence in [evidence/data-audit.json](evidence/data-audit.json). Source terms remain separate from the model's Gemma terms and the code's Apache-2.0 license.
