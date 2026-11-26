# Extending the frozen medium catalogue to FMA Large

This experiment changes the reference catalogue. It retains the archived medium queries, the trained projection, the MuQ checkpoint, the held-out split, the index settings and the decision thresholds. FMA Large contains 30-second excerpts for 106,574 source tracks. The official download is listed in the [FMA inventory](https://github.com/mdeff/fma#data).

## Build inputs

Use the [medium experiment configuration and inputs](https://github.com/ExoHumann/02807_computational_tools_ds/blob/5e9427ee5f83cd32fdd9b5a1c8f40396c3772989/README.md), including the original medium ZIP, the medium reference/query NPZ files, trained head and cached MuQ snapshot. For the open-set comparison use the [fresh runner-up cohort](https://github.com/ExoHumann/02807_computational_tools_ds/blob/c897de3728d60c47fda5271449b8166aa09cd01b/reports/runner-up-margin.md) in `data/medium/runner-up-queries.npz`. The builder checks the medium reference and head against their recorded SHA-256 digests. An independently retrained head is a different experiment.

Run the following from the repository root in the medium CUDA PyTorch environment. On Windows PowerShell:

~~~powershell
New-Item -ItemType Directory -Force data/large | Out-Null
curl.exe --fail --location --retry 4 --continue-at - --output data/large/fma_large.zip https://os.unil.cloud.switch.ch/fma/fma_large.zip
$env:PYTHONPATH = (Resolve-Path src).Path
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python scripts/extend_fma_catalogue.py data/large/fma_large.zip
~~~

To overlap download and inference, run the Python command in a second terminal with `--follow` once the ZIP starts growing. Use the same archive pathname throughout both processes. Complete groups checkpoint to `data/large/shards`; rerunning with the same inputs resumes from them.

On the measured RTX 4090, ten CPU workers prepared the next 96 tracks while CUDA processed batches of 64 windows from the previous group. When the transfer finished, 54,528 additional tracks had already been fingerprinted. A three-group spot check during extraction rotated 6, 10 and 14 workers over the same cached inputs. Median preparation times were 2.13, 2.14 and 2.13 seconds per group respectively. This small probe gave us no reason to increase the worker count on this machine.

The builder checks the shared MP3s' uncompressed byte counts and CRCs against the medium ZIP, copies the existing medium fingerprints and embeds only the 81,574 new source tracks. It streams complete MP3 members, applies the medium 24 kHz mono int16 conversion, and forms 10-second windows on a 5-second hop. CPU workers prepare windows in advance. The frozen MuQ encoder uses CUDA BF16 with FP32 pooling and a non-finite fallback. As in the medium pipeline, pooled features pass through FP16 before the FP32 projection. The builder checks the official Large SHA-1 before writing the final catalogue. The run summary in `data/large/catalogue.json` includes searchable-track counts, failures, fingerprint counts and output hashes.

Of the 81,574 additional source tracks, 80,735 supplied 354,144 searchable windows. The decoder could not read 253 tracks, 515 produced less than ten seconds of audio, and 71 had no window above the RMS cutoff. With the 24,970 searchable medium tracks, the benchmark therefore contains 105,705 tracks and 464,464 windows.

The archive describes its audio as 30-second clips, but 18 searchable tracks contributed more than five windows. Track 039179 alone contributed 122, with the last window starting at 605 seconds. We kept all usable windows under the original segmentation rule. The [pipeline record](catalogue-growth-pipeline.json) contains the archive and output checksums, attrition counts and CPU measurements.

We also independently re-encoded 24 shared, searchable tracks drawn without replacement with NumPy seed 2807. All 108 window indices and RMS values matched the archived medium values exactly. The lowest cosine between a regenerated fingerprint and its archived counterpart was 0.9999957, and every regenerated window retrieved its source track from Medium.

## Reproduce the comparison

The checked-in notebook reads the measured JSON and regenerates its figure without the audio or GPU. Repeating the GPU measurements requires the group's frozen medium reference, projection head and query files. The public v0.2 release contains the small experiment. Its fingerprints have different checksums. The archived medium retrieval input bundle contains the reference and the two closed-set query families. The head and runner-up query pack are separate inputs.

Use the CUDA CuPy environment from the [IVF benchmark](https://github.com/ExoHumann/02807_computational_tools_ds/blob/b35e9525385f7855b7858af046f9540c4dcb5e87/notebooks/ivf_medium.ipynb). The original LSH, IVF and runner-up revisions must be available to Git:

~~~sh
git fetch https://github.com/nickiliak/02807_computational_tools_ds.git refs/pull/1/head refs/pull/4/head refs/pull/6/head
python scripts/benchmark_catalogue_growth.py
~~~

Run timing after extraction completes with the GPU available. Search latency starts with the indexes and query fingerprints resident on the device. Initial download, audio decoding and neural feature extraction are separate costs.
