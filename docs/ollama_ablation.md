# Ollama ablation log

All experiments used the organizer-provided 200-session public set and `nomic-embed-text` through Ollama. Product candidates came from the same field-weighted BM25 index.

| Version | Semantic weight | Candidate window | Correction gate | TechnicalScore | Hit@10 | MRR | MTTC |
|---|---:|---:|---|---:|---:|---:|---:|
| Local retrieval checkpoint | 0 | — | — | 0.789582 | 0.925 | 0.589605 | 3.490 |
| Ollama, light fusion | 0.10 | 16 | No | 0.789700 | 0.925 | 0.590000 | 3.490 |
| Ollama, balanced fusion | 0.18 | 16 | No | 0.791543 | 0.925 | 0.595810 | 3.485 |
| Ollama, strong fusion | 0.35 | 16 | No | 0.789191 | 0.925 | 0.586637 | 3.465 |
| Ollama, smaller window | 0.18 | 12 | No | 0.790197 | 0.925 | 0.590990 | 3.480 |
| **Final: correction-aware hybrid** | **0.18** | **16** | **Yes** | **0.792766** | **0.925** | **0.600220** | **3.490** |

## What I learned

More model influence was not automatically better. A weight of 0.35 moved relevant exact matches down and reduced MRR. A 12-product window was cheaper but lost some ranking gain.

The largest useful observation was scenario-specific: semantic reranking improved Boundary, Browsing, and Buying MRR, but weakened Intent Override. Disabling semantic reranking after a correction kept the benefits without accepting that regression.

## Question-planner ablation

| Policy | TechnicalScore | Hit@10 | MRR | MTTC | Boundary Hit@10 |
|---|---:|---:|---:|---:|---:|
| Guarded counterfactual trace | **0.792766** | **0.925** | **0.600220** | **3.490** | 0.60 |
| Unrestricted counterfactual selection | 0.729011 | 0.885 | 0.494702 | 4.095 | **0.70** |

The unrestricted planner found one more Boundary target, but it asked less answerable questions in the much larger Buying and Browsing groups. The default therefore keeps the counterfactual calculation visible while using the validated question sequence as a deployment guardrail.

## Runtime notes

- Cold cache, 16 candidates: about 4 minutes 22 seconds on the development Mac.
- Final warm-cache verification: about 40 seconds on the same computer.
- Generated product cache after the experiments: about 21 MB.

Runtime varies with hardware and how many candidate embeddings are already cached. These measurements include the complete 200-session evaluator, not a single request.
