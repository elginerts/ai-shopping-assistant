# Ollama ablation log

All experiments used the organizer-provided 200-session public set and `nomic-embed-text` through Ollama. Product candidates came from the same field-weighted BM25 index.

| Version | Semantic weight | Candidate window | Correction gate | TechnicalScore | Hit@10 | MRR | MTTC |
|---|---:|---:|---|---:|---:|---:|---:|
| Local retrieval checkpoint | 0 | — | — | 0.789582 | 0.925 | 0.589605 | 3.490 |
| Ollama, light fusion | 0.10 | 16 | No | 0.789700 | 0.925 | 0.590000 | 3.490 |
| Ollama, balanced fusion | 0.18 | 16 | No | 0.791543 | 0.925 | 0.595810 | 3.485 |
| Ollama, strong fusion | 0.35 | 16 | No | 0.789191 | 0.925 | 0.586637 | 3.465 |
| Ollama, smaller window | 0.18 | 12 | No | 0.790197 | 0.925 | 0.590990 | 3.480 |
| Lexical-only after correction | 0.18 | 16 | Lexical | 0.792766 | 0.925 | 0.600220 | 3.490 |
| **Final: clean-ledger semantic query** | **0.18** | **16** | **Clean query** | **0.793614** | **0.925** | **0.602381** | **3.480** |
| Wider semantic window (rejected) | 0.18 | 30 | Clean query | 0.792106 | 0.925 | 0.596688 | 3.470 |

## What I learned

More model influence was not automatically better. A weight of 0.35 moved relevant exact matches down and reduced MRR. A 12-product window was cheaper but lost some ranking gain.

The largest useful observation was scenario-specific: embedding the raw conversation weakened Intent Override because old and negated preferences remained semantically close. Compiling a query from active ledger revisions performed better than disabling semantic ranking after corrections. Increasing the window to 30 cost more model work and reduced MRR, so the bounded 16-candidate design remains the default.

## Question-planner ablation

| Policy | TechnicalScore | Hit@10 | MRR | MTTC | Boundary Hit@10 |
|---|---:|---:|---:|---:|---:|
| Guarded counterfactual trace + clean intent | **0.793614** | **0.925** | **0.602381** | **3.480** | 0.60 |
| Unrestricted counterfactual selection | 0.729011 | 0.885 | 0.494702 | 4.095 | **0.70** |

The unrestricted planner found one more Boundary target, but it asked less answerable questions in the much larger Buying and Browsing groups. The default therefore keeps the counterfactual calculation visible while using the validated question sequence as a deployment guardrail.

## Runtime notes

- Cold cache, 16 candidates: about 4 minutes 22 seconds on the development Mac.
- Final warm-cache verification: about 40 seconds on the same computer.
- Generated product cache after the experiments: about 21 MB.

Runtime varies with hardware and how many candidate embeddings are already cached. These measurements include the complete 200-session evaluator, not a single request.

## Full-catalogue dense challenger ablation

The complete 50,000-product Nomic index found five failed targets that BM25 missed. Admission was the limiting step: broad promotion changed 557 turn slates and reduced MRR, while correction-only promotion changed 22 slates but promoted no correct target.

| Policy | TechnicalScore | Hit@10 | MRR | MTTC | Promotion turns |
|---|---:|---:|---:|---:|---:|
| Verified default | **0.793614** | 0.925 | **0.602381** | 3.480 | 0 |
| Dense every turn | 0.786133 | 0.925 | 0.575776 | **3.455** | 557 |
| Dense on revisions | 0.792364 | 0.925 | 0.597881 | 3.475 | 22 |

The code, portable index builder, and diagnostics remain available through `THREADLINE_DENSE_MODE=challenger`. The measured default stays off because an interesting retrieval idea is not a deployment improvement until its final ranking evidence is stronger.
