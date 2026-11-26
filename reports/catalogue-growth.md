# FMA medium-to-large retrieval scaling

GPU: NVIDIA GeForce RTX 4090

| Catalogue | Windows | Tracks |
|---|---:|---:|
| Medium | 110,320 | 24,970 |
| Large | 464,464 | 105,705 |

The Large archive contains 80,735 added track IDs and preserves every matched Medium window. Its maximum `segment_idx` is 121; tracks above the common 0–4 excerpt range: 008747, 008748, 008749, 008750, 008751, 008752, 008753, 008754, 008755, 008756, 008757, 008758, 035724, 036728, 039179, 106611, 106613, 137686.

## Closed-set identification

| Catalogue | Family | Exact | Accuracy-first LSH | Spherical IVF |
|---|---|---:|---:|---:|
| Medium | standard | 7,315/7,458 (98.08%) | 7,315/7,458 (98.08%) | 7,274/7,458 (97.53%) |
| Medium | tiktok | 8,411/8,992 (93.54%) | 8,411/8,992 (93.54%) | 8,342/8,992 (92.77%) |
| Large | standard | 7,185/7,458 (96.34%) | 7,195/7,458 (96.47%) | 7,159/7,458 (95.99%) |
| Large | tiktok | 8,095/8,992 (90.02%) | 8,111/8,992 (90.20%) | 8,044/8,992 (89.46%) |

## Paired single-query retrieval time

| Catalogue | Method | Median (ms) | 95th percentile (ms) |
|---|---|---:|---:|
| Medium | EXACT | 0.1730 | 0.1976 |
| Medium | LSH | 0.0815 | 0.0979 |
| Medium | IVF | 0.0699 | 0.0805 |
| Large | EXACT | 0.5849 | 0.6672 |
| Large | LSH | 0.2533 | 0.3326 |
| Large | IVF | 0.1238 | 0.2392 |

## Batch throughput

| Catalogue | Method | Queries/s |
|---|---|---:|
| Medium | EXACT | 376,036 |
| Medium | LSH | 71,854 |
| Medium | IVF | 433,447 |
| Large | EXACT | 95,267 |
| Large | LSH | 7,057 |
| Large | IVF | 89,420 |

## Speedup and catalogue growth

| Catalogue | LSH median speedup vs exact | IVF median speedup vs exact |
|---|---:|---:|
| Medium | 2.12× | 2.47× |
| Large | 2.31× | 4.72× |

| Method | Median latency change, Large vs Medium | p95 latency change, Large vs Medium | Batch throughput change, Large vs Medium |
|---|---:|---:|---:|
| EXACT | +238.1% | +237.6% | -74.7% |
| LSH | +210.8% | +239.6% | -90.2% |
| IVF | +77.2% | +197.1% | -79.4% |

Timing used the same 192 resident GPU query vectors per catalogue, with methods warmed and interleaved; every single-query call was synchronized before and after. Measurements include lookup compute, kernel launches, and host call overhead through GPU completion. They exclude audio processing, query transfer, index construction, and the analysis-only LSH margin pass.
Batch throughput used Three synchronized passes over the same held-out query vectors in chunks of 128..

## Paired closed-set changes by family

| Family | Method | Lost correct | Gained correct | Accuracy change, pp |
|---|---|---:|---:|---:|
| standard | EXACT | 130 | 0 | -1.743 |
| standard | LSH | 120 | 0 | -1.609 |
| standard | IVF | 141 | 26 | -1.542 |
| tiktok | EXACT | 316 | 0 | -3.514 |
| tiktok | LSH | 300 | 0 | -3.336 |
| tiktok | IVF | 348 | 50 | -3.314 |

## Open-set fixed thresholds

Thresholds were selected on the earlier development set and transferred unchanged. The designated missing-song IDs were removed from both catalogues before scoring.

| Catalogue | Method | Rule | Known correct accepted | Unknown false accepts |
|---|---|---|---:|---:|
| Medium | EXACT | frozen maximum score | 536/1,024 | 4/1,024 |
| Medium | EXACT | frozen score and gap | 731/1,024 | 11/1,024 |
| Medium | IVF | frozen maximum score | 536/1,024 | 4/1,024 |
| Medium | IVF | frozen score and gap | 725/1,024 | 11/1,024 |
| Medium | LSH | frozen maximum score | 536/1,024 | 4/1,024 |
| Medium | LSH | frozen score and gap | 731/1,024 | 11/1,024 |
| Large | EXACT | frozen maximum score | 523/1,024 | 18/1,024 |
| Large | EXACT | frozen score and gap | 691/1,024 | 23/1,024 |
| Large | IVF | frozen maximum score | 523/1,024 | 18/1,024 |
| Large | IVF | frozen score and gap | 684/1,024 | 23/1,024 |
| Large | LSH | frozen maximum score | 526/1,024 | 18/1,024 |
| Large | LSH | frozen score and gap | 691/1,024 | 23/1,024 |

| Catalogue | Method | Rule | Family | Known correct rate | Unknown false-accept rate |
|---|---|---|---|---:|---:|
| Medium | EXACT | frozen maximum score | standard | 71.68% | 0.78% |
| Medium | EXACT | frozen maximum score | tiktok | 33.01% | 0.00% |
| Medium | EXACT | frozen score and gap | standard | 88.67% | 1.56% |
| Medium | EXACT | frozen score and gap | tiktok | 54.10% | 0.59% |
| Medium | IVF | frozen maximum score | standard | 71.68% | 0.78% |
| Medium | IVF | frozen maximum score | tiktok | 33.01% | 0.00% |
| Medium | IVF | frozen score and gap | standard | 88.28% | 1.56% |
| Medium | IVF | frozen score and gap | tiktok | 53.32% | 0.59% |
| Medium | LSH | frozen maximum score | standard | 71.68% | 0.78% |
| Medium | LSH | frozen maximum score | tiktok | 33.01% | 0.00% |
| Medium | LSH | frozen score and gap | standard | 88.67% | 1.56% |
| Medium | LSH | frozen score and gap | tiktok | 54.10% | 0.59% |
| Large | EXACT | frozen maximum score | standard | 70.51% | 2.73% |
| Large | EXACT | frozen maximum score | tiktok | 31.64% | 0.78% |
| Large | EXACT | frozen score and gap | standard | 83.98% | 3.12% |
| Large | EXACT | frozen score and gap | tiktok | 50.98% | 1.37% |
| Large | IVF | frozen maximum score | standard | 70.31% | 2.73% |
| Large | IVF | frozen maximum score | tiktok | 31.84% | 0.78% |
| Large | IVF | frozen score and gap | standard | 83.20% | 3.12% |
| Large | IVF | frozen score and gap | tiktok | 50.39% | 1.37% |
| Large | LSH | frozen maximum score | standard | 70.90% | 2.73% |
| Large | LSH | frozen maximum score | tiktok | 31.84% | 0.78% |
| Large | LSH | frozen score and gap | standard | 83.98% | 3.12% |
| Large | LSH | frozen score and gap | tiktok | 50.98% | 1.37% |

### PR6 score-only count reconciliation

PR6 records 538/1,024 known correct accepts; the pinned CuPy row-order exact result gives 536/1,024 strict top-1 accepts. Counting tied top scores as correct gives 538/1,024, matching PR6. The frozen threshold is unchanged.

| Test query | Family / distortion | Known track | CuPy winner | Equal PR6 winner / runner-up score | Score minus threshold |
|---:|---|---:|---:|---:|---:|
| 16 | standard / time_stretch | 117253 | 117089 | 0.887678623 | +0.053468704 |
| 960 | standard / high_pass | 122153 | 046066 | 0.893376529 | +0.059166610 |
PR6 uses PyTorch top-k tie selection; the pinned CuPy exact search picks the first matching reference row. The tied scores and frozen threshold are unchanged, but deterministic top-1 tie winners differ.

## Scope

- FMA Large is the 30-second excerpt collection, though the processed catalogue preserves longer segment sequences; see the pair-validation summary.
- Closed-set queries retain the original medium head and synthetic distortions.
- Frozen open-set thresholds are transferred without recalibration to the larger catalogue.
- Search timings include lookup only after reference and index data are resident on the GPU.
- LSH score-gap is an analysis-only metric from the exact-score slab restricted to the union of its pinned multprobe buckets; it is excluded from LSH lookup timings.
