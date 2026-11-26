"""Does a 256-centre, 16-probe spherical IVF index preserve a song's runner-up?

Run in the IVF CuPy environment: python check_cluster_margins.py PATH_TO_REPO
Uses the committed IVF implementation and the previous 2,048-song open-set cohort.
"""
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd


def blob(repo, commit, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{path}"])


def distinct_two(cp, scores, tracks):
    rows = cp.arange(len(scores))
    first = scores.argmax(axis=1)
    first_track = tracks[first]
    first_score = scores[rows, first]
    others = cp.where(tracks[None, :] != first_track[:, None], scores, -cp.inf)
    second = others.argmax(axis=1)
    return cp.asnumpy(cp.stack((first_track, tracks[second], first_score, others[rows, second]), axis=1))


def threshold(values):
    # Highest accepted set under the empirical 1% cap, including ties conservatively.
    scores = np.sort(values)[::-1]
    return np.nextafter(scores[len(scores) // 100], np.inf)


def run(repo):
    root = repo / "data/medium"
    os.environ.setdefault("CUPY_CACHE_DIR", str(root / "cupy-cache"))
    import cupy as cp

    with np.load(root / "unknown-song-queries.npz") as q:
        queries = cp.asarray(q["vectors"])
        table = pd.DataFrame({k: q[k] for k in ("track_id", "known", "phase", "family")})
    with np.load(root / "data/processed/fingerprints/references.npz") as ref:
        keep = ~np.isin(ref["track_id"], table.loc[~table.known, "track_id"])
        ids = ref["track_id"][keep]
        refs, tracks = cp.asarray(ref["vectors"][keep]), cp.asarray(ids)
    assert np.array_equal(np.isin(table.track_id, ids), table.known)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        for name in ("ivf.py", "ivf.cu"):
            (Path(tmp) / name).write_bytes(blob(repo, "b35e952", "src/afp/" + name))
        spec = importlib.util.spec_from_file_location("frozen_ivf", Path(tmp) / "ivf.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        index = module.SphericalIVF(refs, tracks, n_clusters=256, seed=2807, max_iter=20)
        labels = cp.empty(len(refs), dtype=cp.int32)
        labels[index.order] = cp.repeat(cp.arange(256, dtype=cp.int32), cp.diff(index.offsets))
        selected_tracks, selected_scores, _ = index.search(queries, nprobe=16)
        results = {"exact": [], "ivf": []}
        for start in range(0, len(queries), 64):
            batch = queries[start:start + 64]
            scores = batch @ refs.T
            results["exact"].append(distinct_two(cp, scores, tracks))
            closest = cp.argpartition(batch @ index.centroids.T, -16, axis=1)[:, -16:]
            routed = cp.zeros((len(batch), 256), dtype=cp.bool_)
            routed[cp.arange(len(batch))[:, None], closest] = True
            results["ivf"].append(distinct_two(cp, cp.where(routed[:, labels], scores, -cp.inf), tracks))

    original = pd.read_csv(io.BytesIO(blob(repo, "d15e8e9", "reports/unknown-song-scores.csv.gz")), compression="gzip")
    assert table.equals(original[table.columns])
    for name, chunks in results.items():
        top, runner, best, second = np.concatenate(chunks).T
        assert np.all(top != runner) and np.all(best >= second) and np.isfinite(second).all()
        table[name + "_top"] = top.astype(int)
        table[name + "_runner"] = runner.astype(int)
        table[name + "_best"] = best
        table[name + "_gap"] = best - second
    assert np.array_equal(table.exact_top, original.exact_track)
    np.testing.assert_allclose(table.exact_best, original.exact_score, atol=2e-5, rtol=0)
    assert np.array_equal(table.ivf_top, cp.asnumpy(selected_tracks))
    np.testing.assert_allclose(table.ivf_best, cp.asnumpy(selected_scores), atol=2e-5, rtol=0)
    dev_unknown = (table.phase == "development") & ~table.known
    test = table[table.phase == "test"]
    assert len(test) == 2048 and test.track_id.nunique() == 1024
    out = {"cohort": "previous open-set validation, exploratory", "test_queries": len(test),
           "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
           "index": {"centres": 256, "probes": 16, "reference_windows": len(refs)},
           "same_top": int((test.ivf_top == test.exact_top).sum()),
           "same_ordered_top_two": int(((test.ivf_top == test.exact_top) & (test.ivf_runner == test.exact_runner)).sum()),
           "rows": []}
    decisions = {}
    for signal in ("best", "gap"):
        thresholds = {method: threshold(table.loc[dev_unknown, method + "_" + signal]) for method in results}
        for retrieval, calibrated_on in (("exact", "exact"), ("ivf", "exact"), ("ivf", "ivf")):
            cut = thresholds[calibrated_on]
            accepted = test[retrieval + "_" + signal] >= cut
            correct = test[retrieval + "_top"] == test.track_id
            if calibrated_on == retrieval:
                decisions[retrieval, signal] = accepted & (correct | ~test.known)
            out["rows"].append({"signal": signal, "retrieval": retrieval, "calibration": calibrated_on,
                                "threshold": float(cut), "correct_known": int((accepted & correct & test.known).sum()),
                                "false_unknown": int((accepted & ~test.known).sum()), "known_n": int(test.known.sum()),
                                "unknown_n": int((~test.known).sum())})
    out["paired_gap_minus_best_pp"] = {}
    for method in results:
        delta = decisions[method, "gap"].astype(int) - decisions[method, "best"].astype(int)
        for known, metric in ((True, "correct_known"), (False, "false_unknown")):
            group = test[test.known == known]
            by_song = delta.loc[group.index].groupby(group.track_id).mean().to_numpy()
            rng = np.random.default_rng(2807)
            draws = rng.choice(by_song, size=(10000, len(by_song)), replace=True).mean(axis=1) * 100
            out["paired_gap_minus_best_pp"][method + "_" + metric] = {
                "estimate": float(by_song.mean() * 100),
                "bootstrap95": np.quantile(draws, [0.025, 0.975]).tolist()}
    path = Path(__file__).with_name("cluster-margins.json")
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf8")
    print(path.read_text(encoding="utf8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    run(parser.parse_args().repo.resolve())
