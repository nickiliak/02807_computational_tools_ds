"""Paired ablation of raw MuQ vs the trained fingerprint under exact CUDA lookup.

Run in the project's torch environment: python check_projection.py PATH_TO_REPO
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize


def cached(folder, keys):
    chunks = {k: [] for k in keys}
    for path in sorted(folder.glob("shard_*.npz")):
        with np.load(path) as shard:
            for k in keys:
                chunks[k].append(shard[k])
    return {k: np.concatenate(parts) for k, parts in chunks.items()}


def unit(values):
    matrix = torch.as_tensor(values, device="cuda", dtype=torch.float32)
    assert torch.isfinite(matrix).all() and (matrix.norm(dim=1) > 0).all()
    return normalize(matrix)


def nearest(reference, queries, ids):
    predictions = []
    with torch.inference_mode():
        for batch in queries.split(96):
            predictions.append(ids[(batch @ reference.T).argmax(dim=1)].cpu().numpy())
    return np.concatenate(predictions)


def run(repo):
    torch.set_float32_matmul_precision("highest")
    root = repo / "data/medium/data/processed"
    raw = cached(root / "embeddings/clean", ["embeddings", "track_id", "segment_idx", "rms"])
    keep = raw["rms"] >= 1e-3
    raw_ref = unit(raw["embeddings"][keep])
    with np.load(root / "fingerprints/references.npz") as ref:
        for k in ("track_id", "segment_idx", "rms"):
            np.testing.assert_array_equal(raw[k][keep], ref[k])
        ids = torch.as_tensor(ref["track_id"], device="cuda")
        trained_ref = unit(ref["vectors"])
    with np.load(root / "fingerprints/queries.npz") as std:
        songs = np.unique(std["track_id"])
    dev = np.random.default_rng(2211).choice(songs, len(songs) // 4, replace=False)
    result = {"device": torch.cuda.get_device_name(), "reference_windows": len(ids),
              "raw_dimension": raw_ref.shape[1], "trained_dimension": trained_ref.shape[1], "families": {}}
    for family, expected in (("queries", 7315), ("queries_tiktok", 8411)):
        q = cached(root / "embeddings" / family, ["distorted", "track_id", "start_s", "augmentation"])
        with np.load(root / "fingerprints" / (family + ".npz")) as fp:
            for k in ("track_id", "start_s", "augmentation"):
                np.testing.assert_array_equal(q[k], fp[k])
            keep = ~np.isin(q["track_id"], dev)
            source = q["track_id"][keep]
            hit_head = nearest(trained_ref, unit(fp["vectors"][keep]), ids) == source
        assert hit_head.sum() == expected
        hit_raw = nearest(raw_ref, unit(q["distorted"][keep]), ids) == source
        delta = hit_head.astype(int) - hit_raw.astype(int)
        table = pd.DataFrame({"song": source, "delta": delta}).groupby("song").delta.agg(["sum", "size"])
        sums, sizes = table["sum"].to_numpy(), table["size"].to_numpy()
        rng = np.random.default_rng(2807)
        intervals = []
        for _ in range(100):
            draw = rng.integers(len(table), size=(100, len(table)))
            intervals.extend(100 * sums[draw].sum(axis=1) / sizes[draw].sum(axis=1))
        result["families"][family] = {"queries": len(source), "songs": len(table),
                                      "raw_correct": int(hit_raw.sum()), "trained_correct": int(hit_head.sum()),
                                      "gained": int((delta == 1).sum()), "lost": int((delta == -1).sum()),
                                      "delta_pp": float(delta.mean() * 100),
                                      "paired_song_bootstrap95_pp": np.quantile(intervals, [.025, .975]).tolist()}
    report = json.dumps(result, indent=2)
    Path(__file__).with_name("projection-ablation.json").write_text(report + "\n", encoding="utf8")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    run(parser.parse_args().repo)
