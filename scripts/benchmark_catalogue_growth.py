"""Compare frozen FMA medium and large retrieval on identical query cohorts.

Run from the repository root with the CUDA CuPy environment:
    python scripts/benchmark_catalogue_growth.py

The large reference archive is produced by the project's fingerprint pipeline.
The script checks that it contains the unchanged medium windows before timing.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LSH_COMMIT = "5e9427ee5f83cd32fdd9b5a1c8f40396c3772989"
IVF_COMMIT = "b35e9525385f7855b7858af046f9540c4dcb5e87"
MARGIN_COMMIT = "c897de3728d60c47fda5271449b8166aa09cd01b"
MEDIUM_REFERENCE_SHA256 = "174234b82db42e42839b26b7d52372b908075d5724451037b26e4a865d33a9b9"
LSH_SETTINGS = {"bits": 10, "tables": 64, "probes": 3, "seed": 2807}
IVF_SETTINGS = {"centres": 256, "probes": 16, "seed": 2807, "max_iter": 20}
DEV_SEED = 2211
TIMING_SEED = 42007
TIMING_QUERIES = 192
BATCH_SIZE = 128
GPUAngularLSH = None


def git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(REPO), "show", f"{commit}:{path}"])


def load_ivf_module(temp_root: Path):
    package = temp_root / "afp"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for filename in ("ivf.py", "ivf.cu"):
        source_path = "src/afp/" + filename
        (package / filename).write_bytes(git_blob(IVF_COMMIT, source_path))
    spec = importlib.util.spec_from_file_location("catalogue_growth_ivf", package / "ivf.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SphericalIVF


def load_lsh_module(temp_root: Path):
    package = temp_root / "lsh"
    package.mkdir()
    for filename in ("gpu_retrieval.py", "gpu_lsh.cu"):
        source_path = "src/afp/" + filename
        (package / filename).write_bytes(git_blob(LSH_COMMIT, source_path))
    spec = importlib.util.spec_from_file_location("catalogue_growth_lsh", package / "gpu_retrieval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json_blob(commit: str, path: str):
    return json.loads(git_blob(commit, path))


def load_pr6_score_rows():
    raw = git_blob(MARGIN_COMMIT, "reports/runner-up-margin-scores.csv.gz")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
        return list(csv.DictReader(io.TextIOWrapper(compressed, encoding="utf-8")))


def reconcile_pr6_score_only_ties(current_exact_rows, threshold_selection):
    """Compare strict CuPy top-1 outcomes to PR6 and explain exact tied winners."""
    pr6_rows = [row for row in load_pr6_score_rows() if row["phase"] == "test"]
    track_ids = np.asarray(current_exact_rows["track_id"], dtype=np.int64)
    known = np.asarray(current_exact_rows["known"], dtype=bool)
    family = np.asarray(current_exact_rows["family"], dtype=str)
    current_track = np.asarray(current_exact_rows["top_track"], dtype=np.int64)
    current_score = np.asarray(current_exact_rows["top_score"], dtype=np.float64)
    pr6_track = np.asarray([int(row["track_id"]) for row in pr6_rows], dtype=np.int64)
    pr6_known = np.asarray([row["known"].lower() == "true" for row in pr6_rows], dtype=bool)
    pr6_family = np.asarray([row["family"] for row in pr6_rows], dtype=str)
    pr6_top_track = np.asarray([int(row["exact_track"]) for row in pr6_rows], dtype=np.int64)
    pr6_top_score = np.asarray([float(row["exact_score"]) for row in pr6_rows], dtype=np.float64)
    pr6_runner_track = np.asarray([int(row["runner_up_track"]) for row in pr6_rows], dtype=np.int64)
    pr6_runner_score = np.asarray([float(row["runner_up_score"]) for row in pr6_rows], dtype=np.float64)
    if not (
        np.array_equal(track_ids, pr6_track)
        and np.array_equal(known, pr6_known)
        and np.array_equal(family, pr6_family)
    ):
        raise ValueError("Current open-set query order differs from the pinned PR6 score cohort")

    score_threshold = float(threshold_selection["maximum_score"]["score_threshold"])
    pr6_accepted = known & (pr6_top_track == track_ids) & (pr6_top_score >= score_threshold)
    current_accepted = known & (current_track == track_ids) & (current_score >= score_threshold)
    missing = np.flatnonzero(pr6_accepted & ~current_accepted)
    extra = np.flatnonzero(current_accepted & ~pr6_accepted)
    tie_cases = []
    unexplained = []
    for index in missing:
        score_gap = float(pr6_top_score[index] - pr6_runner_score[index])
        score_delta = float(current_score[index] - pr6_top_score[index])
        candidate = {
            "test_query_index": int(index),
            "track_id": f"{track_ids[index]:06d}",
            "family": str(family[index]),
            "distortion": pr6_rows[index]["distortion"],
            "pr6_winner": f"{pr6_top_track[index]:06d}",
            "cupy_winner": f"{current_track[index]:06d}",
            "pr6_winner_score": float(pr6_top_score[index]),
            "pr6_runner_up_track": f"{pr6_runner_track[index]:06d}",
            "pr6_runner_up_score": float(pr6_runner_score[index]),
            "pr6_score_gap": score_gap,
            "cupy_top_score": float(current_score[index]),
            "cupy_minus_pr6_score": score_delta,
            "score_threshold": score_threshold,
            "winner_score_minus_threshold": float(pr6_top_score[index] - score_threshold),
        }
        is_exact_tie = (
            pr6_runner_track[index] == current_track[index]
            and abs(score_gap) <= 1e-6
            and abs(score_delta) <= 2e-5
        )
        if is_exact_tie:
            tie_cases.append(candidate)
        else:
            unexplained.append(candidate)
    return {
        "pr6_known_correct_score_only_accepts": int(pr6_accepted.sum()),
        "cupy_strict_known_correct_score_only_accepts": int(current_accepted.sum()),
        "tie_aware_known_correct_score_only_accepts": int(current_accepted.sum() + len(tie_cases)),
        "score_threshold": score_threshold,
        "explained_by_exact_tie_breaks": len(tie_cases),
        "other_pr6_acceptances_missing_from_cupy": unexplained,
        "cupy_acceptances_not_in_pr6": [int(index) for index in extra],
        "reconciled": len(unexplained) == 0 and len(extra) == 0 and len(tie_cases) == len(missing),
        "tie_cases": tie_cases,
        "interpretation": (
            "PR6 uses PyTorch top-k tie selection; the pinned CuPy exact search picks the first matching reference row. "
            "The tied scores and frozen threshold are unchanged, but deterministic top-1 tie winners differ."
        ),
    }


def load_npz(path: Path, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        missing = set(fields) - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields {sorted(missing)}")
        return {name: archive[name] for name in fields}


def load_catalogue(path: Path):
    data = load_npz(path, ("vectors", "track_id", "segment_idx", "start_s", "rms"))
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    tracks = np.asarray(data["track_id"], dtype=np.int32)
    if vectors.ndim != 2 or vectors.shape[1] != 256 or len(tracks) != len(vectors):
        raise ValueError(f"Unexpected catalogue shapes in {path}")
    keys = catalogue_keys(tracks, data["segment_idx"])
    if len(np.unique(keys)) != len(keys):
        raise ValueError(f"Duplicate (track_id, segment_idx) rows in {path}")
    return data, vectors, tracks, keys


def catalogue_keys(track_ids, segments):
    track_values = np.asarray(track_ids, dtype=np.int64)
    segment_values = np.asarray(segments, dtype=np.int64)
    if np.any(segment_values < 0):
        raise ValueError("Expected nonnegative segment_idx values")
    keys = np.empty(len(track_values), dtype=[("track_id", np.int64), ("segment_idx", np.int64)])
    keys["track_id"] = track_values
    keys["segment_idx"] = segment_values
    return keys


def verify_catalogue_pair(medium, large):
    medium_meta, medium_vectors, medium_tracks, medium_keys = medium
    _, large_vectors, large_tracks, large_keys = large
    large_order = np.argsort(large_keys)
    sorted_keys = large_keys[large_order]
    positions = np.searchsorted(sorted_keys, medium_keys)
    if np.any(positions == len(sorted_keys)) or not np.array_equal(sorted_keys[positions], medium_keys):
        raise ValueError("The large reference is missing medium (track_id, segment_idx) windows")
    large_rows = large_order[positions]
    if not np.allclose(large_vectors[large_rows], medium_vectors, atol=2e-6, rtol=0):
        max_delta = float(np.max(np.abs(large_vectors[large_rows] - medium_vectors)))
        raise ValueError(
            "Medium vectors differ inside the large catalogue. Confirm the same frozen head was used. "
            f"Maximum absolute difference: {max_delta:.3g}"
        )
    medium_hash = hashlib.sha256((REPO / "data/medium/data/processed/fingerprints/references.npz").read_bytes()).hexdigest()
    if medium_hash != MEDIUM_REFERENCE_SHA256:
        raise ValueError(f"Medium reference checksum differs from the frozen benchmark: {medium_hash}")
    medium_track_ids = np.unique(medium_tracks)
    large_track_ids = np.unique(large_tracks)
    if not np.isin(medium_track_ids, large_track_ids).all():
        raise ValueError("Large catalogue is missing one or more medium source tracks")
    added_track_ids = np.setdiff1d(large_track_ids, medium_track_ids, assume_unique=True)
    large_segment_values = np.asarray(large[0]["segment_idx"], dtype=np.int64)
    large_tracks_with_long_segments = np.unique(large_tracks[large_segment_values > 4])
    return {
        "medium_windows": len(medium_vectors),
        "large_windows": len(large_vectors),
        "medium_windows_found_in_large": len(medium_meta["track_id"]),
        "medium_tracks": len(medium_track_ids),
        "large_tracks": len(large_track_ids),
        "added_distinct_track_ids": len(added_track_ids),
        "large_max_segment_idx": int(large_segment_values.max()),
        "large_tracks_with_segment_idx_above_4": [
            f"{int(track_id):06d}" for track_id in large_tracks_with_long_segments
        ],
        "medium_track_ids_all_present_in_large": True,
        "same_medium_fingerprints_in_large": True,
        "maximum_vector_absolute_difference": float(np.max(np.abs(large_vectors[large_rows] - medium_vectors))),
    }


def load_closed_queries(fp_root: Path):
    standard = load_npz(fp_root / "queries.npz", ("vectors", "track_id", "augmentation"))
    tiktok = load_npz(fp_root / "queries_tiktok.npz", ("vectors", "track_id", "augmentation"))
    dev_tracks = np.random.default_rng(DEV_SEED).choice(
        np.unique(standard["track_id"]),
        size=len(np.unique(standard["track_id"])) // 4,
        replace=False,
    )
    output = {}
    for name, archive in (("standard", standard), ("tiktok", tiktok)):
        keep = ~np.isin(archive["track_id"], dev_tracks)
        selected = {key: value[keep] for key, value in archive.items()}
        if np.isin(selected["track_id"], dev_tracks).any():
            raise ValueError("Development tracks leaked into held-out query selection")
        output[name] = selected
    expected = {"standard": 7458, "tiktok": 8992}
    actual = {name: len(item["track_id"]) for name, item in output.items()}
    if actual != expected:
        raise ValueError(f"Frozen query split changed. Expected {expected}, got {actual}")
    return output


def medium_query_vectors(queries):
    return np.concatenate([row["vectors"] for row in queries.values()]), np.concatenate(
        [row["track_id"] for row in queries.values()]
    )


def sync(cp):
    cp.cuda.get_current_stream().synchronize()


def score_closed(cp, exact_fn, lsh, ivf, references, tracks, queries):
    outputs = {}
    for family, data in queries.items():
        query_vectors = cp.asarray(data["vectors"], dtype=cp.float32)
        query_tracks = data["track_id"]
        preds, scores = {name: [] for name in ("exact", "lsh", "ivf")}, {
            name: [] for name in ("exact", "lsh", "ivf")
        }
        for first in range(0, len(query_tracks), BATCH_SIZE):
            batch = query_vectors[first : first + BATCH_SIZE]
            prediction, score = exact_fn(references, tracks, batch)
            preds["exact"].append(cp.asnumpy(prediction))
            scores["exact"].append(cp.asnumpy(score))
            prediction, score, _ = lsh.search(batch, probes=LSH_SETTINGS["probes"])
            preds["lsh"].append(cp.asnumpy(prediction))
            scores["lsh"].append(cp.asnumpy(score))
            prediction, score, _ = ivf.search(batch, nprobe=IVF_SETTINGS["probes"], validate=False)
            preds["ivf"].append(cp.asnumpy(prediction))
            scores["ivf"].append(cp.asnumpy(score))
        outputs[family] = {
            name: {"track": np.concatenate(preds[name]), "score": np.concatenate(scores[name])}
            for name in preds
        }
        outputs[family]["truth"] = query_tracks
    return outputs


def accuracy_summary(outputs):
    result = {}
    for family, methods in outputs.items():
        truth = methods["truth"]
        result[family] = {}
        for method in ("exact", "lsh", "ivf"):
            predicted = methods[method]["track"]
            hit = predicted == truth
            result[family][method] = {
                "queries": len(truth),
                "correct": int(hit.sum()),
                "accuracy": float(hit.mean()),
            }
    return result


def paired_catalogue_changes(medium, large):
    changes = {}
    for family in medium:
        changes[family] = {}
        if not np.array_equal(medium[family]["truth"], large[family]["truth"]):
            raise ValueError("The medium and large runs did not use identical query ordering")
        truth = medium[family]["truth"]
        for method in ("exact", "lsh", "ivf"):
            medium_hit = medium[family][method]["track"] == truth
            large_hit = large[family][method]["track"] == truth
            changes[family][method] = {
                "lost_correct": int(np.count_nonzero(medium_hit & ~large_hit)),
                "gained_correct": int(np.count_nonzero(~medium_hit & large_hit)),
                "gained_correct_query_ids": [
                    int(track_id)
                    for track_id in truth[(~medium_hit & large_hit)]
                ],
                "medium_accuracy": float(medium_hit.mean()),
                "large_accuracy": float(large_hit.mean()),
                "accuracy_change_percentage_points": float((large_hit.mean() - medium_hit.mean()) * 100),
                "same_prediction": int(
                    np.count_nonzero(medium[family][method]["track"] == large[family][method]["track"])
                ),
            }
    return changes


def diagnose_closed_exact_gains(cp, medium_catalogue, large_catalogue, queries, medium, large):
    """Expose score margins for any impossible medium-error to large-hit transition."""
    diagnostics = {}
    gains_by_family = {}
    for family, data in queries.items():
        truth = data["track_id"]
        medium_hit = medium[family]["exact"]["track"] == truth
        large_hit = large[family]["exact"]["track"] == truth
        gained = np.flatnonzero(~medium_hit & large_hit)
        gains_by_family[family] = gained
    if not any(len(gained) for gained in gains_by_family.values()):
        return {
            family: {"gained_correct_queries": 0, "score_diagnostics": []}
            for family in gains_by_family
        }
    _, medium_vectors, medium_tracks, _ = medium_catalogue
    _, large_vectors, large_tracks, _ = large_catalogue
    medium_device = cp.asarray(medium_vectors, dtype=cp.float32)
    medium_track_device = cp.asarray(medium_tracks, dtype=cp.int32)
    large_device = cp.asarray(large_vectors, dtype=cp.float32)
    large_track_device = cp.asarray(large_tracks, dtype=cp.int32)
    for family, data in queries.items():
        truth = data["track_id"]
        gained = gains_by_family[family]
        entries = []
        for query_index in gained:
            track_id = int(truth[query_index])
            query = cp.asarray(data["vectors"][query_index : query_index + 1], dtype=cp.float32)
            medium_scores = query @ medium_device.T
            large_scores = query @ large_device.T
            medium_truth_score = float(
                cp.max(cp.where(medium_track_device == track_id, medium_scores[0], -cp.inf)).item()
            )
            large_truth_score = float(
                cp.max(cp.where(large_track_device == track_id, large_scores[0], -cp.inf)).item()
            )
            medium_best_recomputed = float(cp.max(medium_scores).item())
            large_best_recomputed = float(cp.max(large_scores).item())
            medium_best_score = float(medium[family]["exact"]["score"][query_index])
            large_best_score = float(large[family]["exact"]["score"][query_index])
            entries.append(
                {
                    "query_index": int(query_index),
                    "track_id": track_id,
                    "medium_wrong_winner": int(medium[family]["exact"]["track"][query_index]),
                    "medium_best_score": medium_best_score,
                    "medium_true_track_score_recomputed": medium_truth_score,
                    "medium_best_minus_truth_score": medium_best_score - medium_truth_score,
                    "medium_recomputed_best_minus_truth_score": medium_best_recomputed - medium_truth_score,
                    "large_winner_score": large_best_score,
                    "large_true_track_score_recomputed": large_truth_score,
                    "large_true_minus_medium_true_score": large_truth_score - medium_truth_score,
                    "large_recomputed_best_minus_truth_score": large_best_recomputed - large_truth_score,
                    "medium_margin_within_2e-5": (medium_best_recomputed - medium_truth_score) <= 2e-5,
                }
            )
        diagnostics[family] = {"gained_correct_queries": len(entries), "score_diagnostics": entries}
    return diagnostics


def benchmark_open_set(cp, exact_fn, lsh, ivf, references, tracks, query_archive, frozen_rules):
    test = query_archive["phase"] == "test"
    qids = query_archive["track_id"][test]
    known = query_archive["known"][test]
    family = query_archive["family"][test]
    query = cp.asarray(query_archive["vectors"][test], dtype=cp.float32)
    removed = np.unique(query_archive["track_id"][~query_archive["known"]])
    keep = ~np.isin(cp.asnumpy(tracks), removed)
    refs = references[cp.asarray(keep)]
    ref_tracks = tracks[cp.asarray(keep)]
    if not np.array_equal(np.isin(qids, cp.asnumpy(ref_tracks)), known):
        raise ValueError("Missing-song labels disagree with actual catalogue membership after removal")

    # Exact and IVF top-two candidates are reduced to the best window per distinct song.
    # The pinned LSH API returns one winner, so compute its analysis-only margin by
    # masking the exact-score slab to the union of the same four buckets it probes.
    open_lsh_start = time.perf_counter()
    open_lsh = GPUAngularLSH(
        refs,
        ref_tracks,
        bits=LSH_SETTINGS["bits"],
        tables=LSH_SETTINGS["tables"],
        seed=LSH_SETTINGS["seed"],
    )
    sync(cp)
    open_lsh_build_seconds = time.perf_counter() - open_lsh_start
    open_ivf_start = time.perf_counter()
    open_ivf = ivf.__class__(
        refs,
        ref_tracks,
        n_clusters=IVF_SETTINGS["centres"],
        seed=IVF_SETTINGS["seed"],
        max_iter=IVF_SETTINGS["max_iter"],
    )
    sync(cp)
    open_ivf_build_seconds = time.perf_counter() - open_ivf_start
    labels = open_ivf._assign(refs)
    rows = {"exact": [], "ivf": [], "lsh": []}
    ivf_top1, ivf_score, _ = open_ivf.search(query, nprobe=IVF_SETTINGS["probes"], validate=False)
    lsh_top1_device, lsh_score_device, _ = open_lsh.search(query, probes=LSH_SETTINGS["probes"])
    ivf_top1, ivf_score = cp.asnumpy(ivf_top1), cp.asnumpy(ivf_score)
    lsh_top_one_verification = {"matches": 0, "near_tie_or_float_differences": 0, "queries": len(qids)}

    n_tracks = int(cp.max(ref_tracks).item()) + 1
    for start in range(0, len(qids), 16):
        stop = min(start + 16, len(qids))
        batch = query[start:stop]
        similarities = batch @ refs.T
        exact_candidates = similarity_to_track_scores(cp, similarities, ref_tracks, n_tracks)
        exact_best = cp.argpartition(exact_candidates, -2, axis=1)[:, -2:]
        exact_best_scores = cp.take_along_axis(exact_candidates, exact_best, axis=1)
        exact_rank = cp.argsort(exact_best_scores, axis=1)[:, ::-1]
        exact_best = cp.take_along_axis(exact_best, exact_rank, axis=1)
        exact_best_scores = cp.take_along_axis(exact_best_scores, exact_rank, axis=1)

        centroid_scores = batch @ open_ivf.centroids.T
        if IVF_SETTINGS["centres"] <= 1024 and IVF_SETTINGS["probes"] <= 16:
            selected = cp.empty((len(batch), IVF_SETTINGS["probes"]), dtype=cp.int32)
            open_ivf._select(
                (len(batch),),
                (32,),
                (
                    centroid_scores,
                    np.int32(IVF_SETTINGS["centres"]),
                    np.int32(IVF_SETTINGS["probes"]),
                    selected,
                ),
                shared_mem=4 * IVF_SETTINGS["centres"],
            )
        else:
            selected = cp.argpartition(
                centroid_scores, -IVF_SETTINGS["probes"], axis=1
            )[:, -IVF_SETTINGS["probes"] :]
        selected_mask = cp.zeros((len(batch), IVF_SETTINGS["centres"]), dtype=cp.bool_)
        selected_mask[cp.arange(len(batch))[:, None], selected] = True
        candidates = cp.where(selected_mask[:, labels], similarities, -cp.inf)
        ivf_candidates = similarity_to_track_scores(cp, candidates, ref_tracks, n_tracks)
        ivf_best = cp.argpartition(ivf_candidates, -2, axis=1)[:, -2:]
        ivf_best_scores = cp.take_along_axis(ivf_candidates, ivf_best, axis=1)
        ivf_rank = cp.argsort(ivf_best_scores, axis=1)[:, ::-1]
        ivf_best = cp.take_along_axis(ivf_best, ivf_rank, axis=1)
        ivf_best_scores = cp.take_along_axis(ivf_best_scores, ivf_rank, axis=1)

        lsh_candidates = lsh_candidate_mask(cp, open_lsh, batch)
        lsh_candidate_scores = cp.where(lsh_candidates, similarities, -cp.inf)
        lsh_by_track = similarity_to_track_scores(cp, lsh_candidate_scores, ref_tracks, n_tracks)
        frozen_winner = lsh_top1_device[start:stop].astype(cp.int64)
        frozen_score = lsh_score_device[start:stop]
        query_rows = cp.arange(len(batch))
        candidate_score_for_frozen_winner = lsh_by_track[query_rows, frozen_winner]
        if cp.any(cp.abs(candidate_score_for_frozen_winner - frozen_score) > 2e-5).item():
            raise ValueError("Frozen LSH top-1 is missing from the reconstructed union of probed buckets")
        reconstructed_winner = lsh_by_track.argmax(axis=1)
        near_tie = cp.abs(lsh_by_track[query_rows, reconstructed_winner] - frozen_score) <= 2e-5
        exact_winner_agreement = reconstructed_winner == frozen_winner
        lsh_top_one_verification["matches"] += int(cp.count_nonzero(exact_winner_agreement).item())
        lsh_top_one_verification["near_tie_or_float_differences"] += int(
            cp.count_nonzero(~exact_winner_agreement & near_tie).item()
        )
        if cp.any(~exact_winner_agreement & ~near_tie).item():
            raise ValueError("LSH bucket-union top-1 differs materially from the pinned CUDA search")
        runner_scores = lsh_by_track.copy()
        runner_scores[query_rows, frozen_winner] = -cp.inf
        runner_track = runner_scores.argmax(axis=1)
        runner_score = runner_scores[query_rows, runner_track]
        lsh_best = cp.stack((frozen_winner, runner_track), axis=1)
        lsh_best_scores = cp.stack((frozen_score, runner_score), axis=1)

        for method, top_ids, top_scores in (
            ("exact", exact_best, exact_best_scores),
            ("ivf", ivf_best, ivf_best_scores),
            ("lsh", lsh_best, lsh_best_scores),
        ):
            host_ids, host_scores = cp.asnumpy(top_ids), cp.asnumpy(top_scores)
            rows[method].append(
                np.column_stack(
                    (host_ids[:, 0], host_scores[:, 0], host_ids[:, 1], host_scores[:, 1])
                )
            )
    result = {}
    exact_pairs = np.concatenate(rows["exact"])
    ivf_pairs = np.concatenate(rows["ivf"])
    lsh_pairs = np.concatenate(rows["lsh"])
    top_one_reconciliation = {}

    def reconcile_pinned_top_one(values, baseline_track, baseline_score, method):
        baseline_track = np.asarray(baseline_track, dtype=np.int64)
        baseline_score = np.asarray(baseline_score, dtype=np.float32)
        candidate_track = values[:, 0].astype(np.int64)
        candidate_runner = values[:, 2].astype(np.int64)
        candidate_score = values[:, 1].astype(np.float32)
        score_delta = np.abs(candidate_score - baseline_score)
        if np.any(score_delta > 2e-5):
            raise ValueError(
                f"{method} distinct-track top score differs from the pinned search by "
                f"{float(score_delta.max()):.3g}, beyond tolerance"
            )
        mismatched = candidate_track != baseline_track
        # The pinned implementations break exact score ties by reference-row order;
        # the per-track matrix reduction can choose a different tied track. Preserve
        # the pinned winner, and keep two distinct songs if it was the provisional runner.
        replace_runner = mismatched & (candidate_runner == baseline_track)
        values[replace_runner, 2] = candidate_track[replace_runner]
        values[replace_runner, 3] = values[replace_runner, 1]
        values[:, 0] = baseline_track
        values[:, 1] = baseline_score
        top_one_reconciliation[method] = {
            "pinned_winner_matches_distinct_track_reduction": int(np.count_nonzero(~mismatched)),
            "floating_point_or_tie_differences_within_2e-5": int(np.count_nonzero(mismatched)),
            "queries": len(values),
            "maximum_score_absolute_difference": float(score_delta.max()),
            "tie_winner_reconciled_to_pinned_search": True,
        }

    for method, values in (("exact", exact_pairs), ("ivf", ivf_pairs), ("lsh", lsh_pairs)):
        if method == "exact":
            check_track, check_score = exact_fn(refs, ref_tracks, query)
            reconcile_pinned_top_one(
                values, cp.asnumpy(check_track), cp.asnumpy(check_score), method
            )
        elif method == "ivf":
            reconcile_pinned_top_one(values, ivf_top1, ivf_score, method)
        top_track, runner_track = values[:, 0].astype(np.int64), values[:, 2].astype(np.int64)
        top_score, runner_score = values[:, 1], values[:, 3]
        if np.any(top_track == runner_track):
            raise ValueError(f"{method} open-set ranking does not contain two distinct songs")
        rank_inversion = runner_score > top_score
        maximum_rank_inversion = float(max(0.0, np.max(runner_score - top_score)))
        if maximum_rank_inversion > 2e-5:
            raise ValueError(
                f"{method} runner-up exceeds its pinned winner by {maximum_rank_inversion:.3g}, "
                "beyond the floating-point/tie tolerance"
            )
        gap = np.maximum(top_score - runner_score, 0)

        scores_rule = frozen_rules["maximum_score"]
        margin_rule = frozen_rules["score_and_margin"]
        summary = {}
        for name, rule in (("frozen_maximum_score", scores_rule), ("frozen_score_and_gap", margin_rule)):
            accepted = top_score >= rule["score_threshold"]
            if rule["gap_threshold"] is not None:
                if np.isnan(gap).any():
                    summary[name] = {"available": False, "reason": "No distinct-track runner-up was available"}
                    continue
                accepted &= gap >= rule["gap_threshold"]
            accepted_known_correct = accepted & known & (top_track == qids)
            accepted_unknown = accepted & ~known
            accepted_known_wrong = accepted & known & (top_track != qids)
            summary[name] = {
                "score_threshold": rule["score_threshold"],
                "gap_threshold": rule["gap_threshold"],
                "known_correct": int(accepted_known_correct.sum()),
                "known_queries": int(known.sum()),
                "known_correct_rate": float(accepted_known_correct.sum() / known.sum()),
                "known_wrong_accepts": int(accepted_known_wrong.sum()),
                "unknown_false_accepts": int(accepted_unknown.sum()),
                "unknown_queries": int((~known).sum()),
                "unknown_false_acceptance_rate": float(accepted_unknown.sum() / (~known).sum()),
                "by_family": {},
            }
            for label in np.unique(family):
                subset = family == label
                correct_known = accepted_known_correct & subset
                false_unknown = accepted_unknown & subset
                known_count = int((known & subset).sum())
                unknown_count = int((~known & subset).sum())
                summary[name]["by_family"][str(label)] = {
                    "known_correct": int(correct_known.sum()),
                    "known_queries": known_count,
                    "known_correct_rate": (
                        float(correct_known.sum() / known_count) if known_count else None
                    ),
                    "unknown_false_accepts": int(false_unknown.sum()),
                    "unknown_queries": unknown_count,
                    "unknown_false_acceptance_rate": (
                        float(false_unknown.sum() / unknown_count) if unknown_count else None
                    ),
                }
        if method == "lsh":
            summary["_top_one_verification"] = {
                "frozen_lsh_winner_matches_candidate_mask": lsh_top_one_verification["matches"],
                "floating_point_or_tie_differences_within_2e-5": (
                    lsh_top_one_verification["near_tie_or_float_differences"]
                ),
                "queries": lsh_top_one_verification["queries"],
                "score_gap_is_analysis_only": True,
                "gap_candidates_are_union_of_pinned_lsh_buckets": True,
            }
        else:
            summary["_top_one_reconciliation"] = top_one_reconciliation[method]
        summary["_top_two_score_order_check"] = {
            "runner_above_pinned_winner_within_2e-5": int(np.count_nonzero(rank_inversion)),
            "maximum_runner_minus_winner_score": maximum_rank_inversion,
            "negative_gap_clamped_to_zero": True,
        }
        result[method] = summary
    result["_index_build_seconds_after_song_removal"] = {
        "lsh": float(open_lsh_build_seconds),
        "ivf": float(open_ivf_build_seconds),
    }
    result["_exact_query_results"] = {
        "track_id": qids.tolist(),
        "known": known.tolist(),
        "family": family.astype(str).tolist(),
        "top_track": exact_pairs[:, 0].astype(np.int64).tolist(),
        "top_score": exact_pairs[:, 1].astype(float).tolist(),
        "top_two_distinct_track_gap": np.maximum(
            exact_pairs[:, 1] - exact_pairs[:, 3], 0
        ).astype(float).tolist(),
    }
    return result


def similarity_to_track_scores(cp, similarities, track_ids, n_tracks):
    batch_count = similarities.shape[0]
    by_track = cp.full((batch_count, n_tracks), -cp.inf, dtype=cp.float32)
    destination = cp.arange(batch_count, dtype=cp.int64)[:, None] * n_tracks + track_ids[None, :]
    cp.maximum.at(by_track.ravel(), destination.ravel(), similarities.ravel())
    return by_track


def lsh_candidate_mask(cp, index, queries):
    """Mark the union of base and three multi-probe buckets used by frozen LSH."""
    projections = (queries @ index.planes).reshape(
        len(queries), LSH_SETTINGS["tables"], LSH_SETTINGS["bits"]
    )
    absolute = cp.abs(projections)
    first = absolute.argmin(axis=2)
    second_projection = absolute.copy()
    cp.put_along_axis(second_projection, first[:, :, None], cp.inf, axis=2)
    second = second_projection.argmin(axis=2)
    signs = projections >= 0
    weights = 1 << cp.arange(LSH_SETTINGS["bits"], dtype=cp.int32)
    keys = (signs * weights).sum(axis=2, dtype=cp.int32)
    masks = cp.stack(
        (
            cp.zeros_like(first),
            1 << first,
            1 << second,
            (1 << first) | (1 << second),
        ),
        axis=2,
    )
    buckets = keys[:, :, None] ^ masks
    table_ids = cp.arange(LSH_SETTINGS["tables"], dtype=cp.int32)[None, :, None]
    starts = index.offsets[table_ids, buckets]
    ends = index.offsets[table_ids, buckets + 1]
    maximum_bucket_size = int(cp.max(ends - starts).item())
    positions = starts[..., None] + cp.arange(maximum_bucket_size, dtype=cp.int32)
    valid = positions < ends[..., None]
    safe_positions = cp.minimum(positions, len(index.refs) - 1)
    candidate_rows = index.order[table_ids[..., None], safe_positions]
    query_ids = cp.broadcast_to(
        cp.arange(len(queries), dtype=cp.int32)[:, None, None, None], candidate_rows.shape
    )
    mask = cp.zeros((len(queries), len(index.refs)), dtype=cp.bool_)
    mask[query_ids[valid], candidate_rows[valid]] = True
    return mask


def measure_single_query(cp, exact_fn, lsh, ivf, references, tracks, vectors):
    rng = np.random.default_rng(TIMING_SEED)
    selection = rng.choice(len(vectors), size=min(TIMING_QUERIES, len(vectors)), replace=False)
    queries = cp.asarray(vectors[selection], dtype=cp.float32)
    methods = {
        "exact": lambda query: exact_fn(references, tracks, query),
        "lsh": lambda query: lsh.search(query, probes=LSH_SETTINGS["probes"]),
        "ivf": lambda query: ivf.search(query, nprobe=IVF_SETTINGS["probes"], validate=False),
    }
    for method in methods.values():
        method(queries[:2])
        sync(cp)
    samples = {name: [] for name in methods}
    order = np.random.default_rng(TIMING_SEED + 1)
    for query_index in range(len(queries)):
        names = list(methods)
        order.shuffle(names)
        query = queries[query_index : query_index + 1]
        for name in names:
            sync(cp)
            start = time.perf_counter()
            methods[name](query)
            sync(cp)
            samples[name].append((time.perf_counter() - start) * 1000)
    return {
        name: {
            "queries": len(times),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.quantile(times, 0.95)),
            "p99_ms": float(np.quantile(times, 0.99)),
            "order": "warm, single-query, per-call GPU synchronisation, interleaved methods",
        }
        for name, times in samples.items()
    }


def measure_batch_throughput(cp, exact_fn, lsh, ivf, references, tracks, vectors, repetitions=3):
    query = cp.asarray(vectors, dtype=cp.float32)
    methods = {
        "exact": lambda batch: exact_fn(references, tracks, batch),
        "lsh": lambda batch: lsh.search(batch, probes=LSH_SETTINGS["probes"]),
        "ivf": lambda batch: ivf.search(batch, nprobe=IVF_SETTINGS["probes"], validate=False),
    }
    for method in methods.values():
        method(query[:4])
        sync(cp)
    observations = {name: [] for name in methods}
    for repeat in range(repetitions):
        names = list(methods)
        np.random.default_rng(TIMING_SEED + 100 + repeat).shuffle(names)
        for name in names:
            sync(cp)
            start = time.perf_counter()
            for first in range(0, len(query), BATCH_SIZE):
                methods[name](query[first : first + BATCH_SIZE])
            sync(cp)
            observations[name].append(time.perf_counter() - start)
    return {
        name: {
            "queries": len(vectors),
            "chunk_size": BATCH_SIZE,
            "repetitions": repetitions,
            "median_seconds": float(np.median(times)),
            "queries_per_second": float(len(vectors) / np.median(times)),
        }
        for name, times in observations.items()
    }


def index_build(cp, refs_np, tracks_np, ivf_class):
    references = cp.asarray(refs_np, dtype=cp.float32)
    tracks = cp.asarray(tracks_np, dtype=cp.int32)
    lsh_start = time.perf_counter()
    lsh = GPUAngularLSH(references, tracks, bits=LSH_SETTINGS["bits"], tables=LSH_SETTINGS["tables"], seed=LSH_SETTINGS["seed"])
    sync(cp)
    lsh_seconds = time.perf_counter() - lsh_start
    ivf_start = time.perf_counter()
    ivf = ivf_class(
        references,
        tracks,
        n_clusters=IVF_SETTINGS["centres"],
        seed=IVF_SETTINGS["seed"],
        max_iter=IVF_SETTINGS["max_iter"],
    )
    sync(cp)
    ivf_seconds = time.perf_counter() - ivf_start
    return references, tracks, lsh, ivf, {
        "lsh_build_seconds": float(lsh_seconds),
        "ivf_build_seconds": float(ivf_seconds),
        "lsh_index_bytes_excluding_references": int(
            lsh.planes.nbytes + lsh.order.nbytes + lsh.offsets.nbytes
        ),
        "ivf_index_bytes_excluding_references": int(ivf.index_bytes),
        "shared_reference_bytes": int(references.nbytes),
    }


def benchmark_catalogue(cp, label, catalogue, exact_fn, ivf_class, queries, open_queries, rules):
    _, vectors, tracks, _ = catalogue
    catalogue_name = Path(label).parts[1].title()
    stage_start = time.perf_counter()
    print(f"[{catalogue_name}] Building resident GPU indexes", flush=True)
    references, device_tracks, lsh, ivf, build = index_build(cp, vectors, tracks, ivf_class)
    print(
        f"[{catalogue_name}] Indexes ready in {time.perf_counter() - stage_start:.1f}s; "
        f"closed-set scoring starts",
        flush=True,
    )
    query_vectors, _ = medium_query_vectors(queries)
    closed = score_closed(cp, exact_fn, lsh, ivf, references, device_tracks, queries)
    print(f"[{catalogue_name}] Closed-set scoring complete; single-query timing starts", flush=True)
    timing_sample = query_vectors
    timing = measure_single_query(cp, exact_fn, lsh, ivf, references, device_tracks, timing_sample)
    print(f"[{catalogue_name}] Single-query timing complete; batch throughput starts", flush=True)
    batch = measure_batch_throughput(
        cp, exact_fn, lsh, ivf, references, device_tracks, query_vectors, repetitions=3
    )
    print(f"[{catalogue_name}] Batch timing complete; open-set threshold evaluation starts", flush=True)
    open_metrics = benchmark_open_set(
        cp, exact_fn, lsh, ivf, references, device_tracks, open_queries, rules
    )
    print(f"[{catalogue_name}] Open-set evaluation complete", flush=True)
    result = {
        "reference_windows": len(vectors),
        "reference_tracks": len(np.unique(tracks)),
        "reference_file_bytes": len((REPO / label).read_bytes()),
        "methods": accuracy_summary(closed),
        "paired_timing": timing,
        "batch_throughput": batch,
        "build_and_memory": build,
        "open_set": open_metrics,
    }
    del references, device_tracks, lsh, ivf
    cp.get_default_memory_pool().free_all_blocks()
    return result, closed


def check_open_set_monotonicity(medium, large, frozen_rules):
    exact_medium, exact_large = medium["exact"], large["exact"]
    max_medium = exact_medium["frozen_maximum_score"]
    max_large = exact_large["frozen_maximum_score"]
    gap_medium = exact_medium["frozen_score_and_gap"]
    gap_large = exact_large["frozen_score_and_gap"]
    medium_rows = medium["_exact_query_results"]
    large_rows = large["_exact_query_results"]
    for key in ("track_id", "known", "family"):
        if medium_rows[key] != large_rows[key]:
            raise ValueError(f"Exact open-set {key} ordering differs across catalogue sizes")
    qids = np.asarray(medium_rows["track_id"], dtype=np.int64)
    known = np.asarray(medium_rows["known"], dtype=bool)
    family = np.asarray(medium_rows["family"], dtype=str)
    medium_top_track = np.asarray(medium_rows["top_track"], dtype=np.int64)
    large_top_track = np.asarray(large_rows["top_track"], dtype=np.int64)
    medium_score = np.asarray(medium_rows["top_score"], dtype=np.float64)
    large_score = np.asarray(large_rows["top_score"], dtype=np.float64)
    medium_gap = np.asarray(medium_rows["top_two_distinct_track_gap"], dtype=np.float64)
    large_gap = np.asarray(large_rows["top_two_distinct_track_gap"], dtype=np.float64)
    max_score_threshold = float(frozen_rules["maximum_score"]["score_threshold"])
    gap_score_threshold = float(frozen_rules["score_and_margin"]["score_threshold"])
    gap_threshold = float(frozen_rules["score_and_margin"]["gap_threshold"])

    medium_unknown_accept = ~known & (medium_score >= max_score_threshold)
    large_unknown_accept = ~known & (large_score >= max_score_threshold)
    declined_unknown = np.flatnonzero(medium_unknown_accept & ~large_unknown_accept)
    medium_known_correct = (
        known
        & (medium_top_track == qids)
        & (medium_score >= gap_score_threshold)
        & (medium_gap >= gap_threshold)
    )
    large_known_correct = (
        known
        & (large_top_track == qids)
        & (large_score >= gap_score_threshold)
        & (large_gap >= gap_threshold)
    )
    gained_known_correct = np.flatnonzero(large_known_correct & ~medium_known_correct)

    def describe(indices, include_gap):
        threshold = gap_score_threshold if include_gap else max_score_threshold
        details = []
        for index in indices:
            detail = {
                "query_index": int(index),
                "track_id": int(qids[index]),
                "family": str(family[index]),
                "medium_top_track": int(medium_top_track[index]),
                "large_top_track": int(large_top_track[index]),
                "medium_score": float(medium_score[index]),
                "large_score": float(large_score[index]),
                "score_delta_large_minus_medium": float(large_score[index] - medium_score[index]),
                "score_threshold": threshold,
                "medium_score_distance_to_threshold": float(medium_score[index] - threshold),
                "large_score_distance_to_threshold": float(large_score[index] - threshold),
                "score_within_2e-5_of_threshold": bool(
                    min(abs(medium_score[index] - threshold), abs(large_score[index] - threshold))
                    <= 2e-5
                ),
            }
            if include_gap:
                detail.update(
                    {
                        "medium_gap": float(medium_gap[index]),
                        "large_gap": float(large_gap[index]),
                        "gap_threshold": gap_threshold,
                        "medium_gap_distance_to_threshold": float(medium_gap[index] - gap_threshold),
                        "large_gap_distance_to_threshold": float(large_gap[index] - gap_threshold),
                        "gap_within_2e-5_of_threshold": bool(
                            min(abs(medium_gap[index] - gap_threshold), abs(large_gap[index] - gap_threshold))
                            <= 2e-5
                        ),
                    }
                )
            details.append(detail)
        return details

    unknown_violations = describe(declined_unknown, include_gap=False)
    gap_violations = describe(gained_known_correct, include_gap=True)
    return {
        "score_only_unknown_false_accepts": {
            "medium": max_medium["unknown_false_accepts"],
            "large": max_large["unknown_false_accepts"],
            "non_decreasing": max_large["unknown_false_accepts"] >= max_medium["unknown_false_accepts"],
            "no_medium_accept_became_large_reject": len(unknown_violations) == 0,
            "unexpected_declines": unknown_violations,
        },
        "score_plus_gap_known_correct_accepts": {
            "medium": gap_medium["known_correct"],
            "large": gap_large["known_correct"],
            "non_increasing": gap_large["known_correct"] <= gap_medium["known_correct"],
            "no_new_correct_accepts": len(gap_violations) == 0,
            "unexpected_new_accepts": gap_violations,
        },
    }


def run(args):
    global GPUAngularLSH
    os.environ.setdefault("CUPY_CACHE_DIR", str(REPO / "data/medium/cupy-cache"))
    import cupy as cp

    with tempfile.TemporaryDirectory(prefix="catalogue-growth-") as temp:
        pinned_lsh = load_lsh_module(Path(temp))
        GPUAngularLSH = pinned_lsh.GPUAngularLSH
        gpu_exact = pinned_lsh.gpu_exact
        ivf_class = load_ivf_module(Path(temp))
        processed = REPO / "data/medium/data/processed"
        medium_path = processed / "fingerprints/references.npz"
        large_path = Path(args.large_reference)
        if not large_path.is_absolute():
            large_path = REPO / large_path
        medium = load_catalogue(medium_path)
        large = load_catalogue(large_path)
        pair_validation = verify_catalogue_pair(medium, large)
        queries = load_closed_queries(processed / "fingerprints")
        open_queries = load_npz(
            REPO / "data/medium/runner-up-queries.npz",
            ("vectors", "track_id", "known", "phase", "family"),
        )
        margin_report = load_json_blob(MARGIN_COMMIT, "reports/runner-up-margin.json")
        rules = margin_report["threshold_selection"]
        medium_result, medium_closed = benchmark_catalogue(
            cp, "data/medium/data/processed/fingerprints/references.npz", medium, gpu_exact, ivf_class, queries, open_queries, rules
        )
        medium_result["open_set"]["pr6_score_only_tie_reconciliation"] = reconcile_pr6_score_only_ties(
            medium_result["open_set"]["_exact_query_results"], rules
        )
        expected = {"standard": {"exact": 7315, "lsh": 7315}, "tiktok": {"exact": 8411, "lsh": 8411}}
        for family, methods in expected.items():
            for method, count in methods.items():
                observed = medium_result["methods"][family][method]["correct"]
                if observed != count:
                    raise ValueError(
                        f"Medium reproducibility check failed for {family}/{method}: {observed}, expected {count}"
                    )
        large_result, large_closed = benchmark_catalogue(
            cp, str(large_path.relative_to(REPO)), large, gpu_exact, ivf_class, queries, open_queries, rules
        )
        large_result["paired_changes_vs_medium"] = paired_catalogue_changes(medium_closed, large_closed)
        closed_exact_gain_diagnostics = diagnose_closed_exact_gains(
            cp, medium, large, queries, medium_closed, large_closed
        )
        open_set_nested_checks = check_open_set_monotonicity(
            medium_result["open_set"], large_result["open_set"], rules
        )
        medium_result["open_set"].pop("_exact_query_results")
        large_result["open_set"].pop("_exact_query_results")
        large_result["exact_search_nested_catalogue_checks"] = {
            "closed_set_exact_correct_gains": {
                family: large_result["paired_changes_vs_medium"][family]["exact"]["gained_correct"]
                for family in ("standard", "tiktok")
            },
            "open_set": open_set_nested_checks,
            "closed_set_exact_gain_score_diagnostics": closed_exact_gain_diagnostics,
            "interpretation": (
                "Adding distinct tracks cannot mathematically turn an exact medium error into a correct match. "
                "If a recorded gain appears, inspect score ties and GPU floating-point changes."
            ),
        }
        speed_comparison = {"speedup_vs_exact": {}, "large_vs_medium_change": {}}
        for catalogue_name, catalogue_result in (("medium", medium_result), ("large", large_result)):
            exact_ms = catalogue_result["paired_timing"]["exact"]["median_ms"]
            speed_comparison["speedup_vs_exact"][catalogue_name] = {
                method: float(exact_ms / catalogue_result["paired_timing"][method]["median_ms"])
                for method in ("lsh", "ivf")
            }
        for method in ("exact", "lsh", "ivf"):
            medium_timing = medium_result["paired_timing"][method]
            large_timing = large_result["paired_timing"][method]
            medium_batch = medium_result["batch_throughput"][method]
            large_batch = large_result["batch_throughput"][method]
            speed_comparison["large_vs_medium_change"][method] = {
                "median_latency_percent": float(
                    (large_timing["median_ms"] / medium_timing["median_ms"] - 1) * 100
                ),
                "p95_latency_percent": float(
                    (large_timing["p95_ms"] / medium_timing["p95_ms"] - 1) * 100
                ),
                "batch_throughput_percent": float(
                    (large_batch["queries_per_second"] / medium_batch["queries_per_second"] - 1) * 100
                ),
            }
        result = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "protocol": {
                "medium_queries_are_identical_for_both_catalogues": True,
                "closed_queries": {name: len(row["track_id"]) for name, row in queries.items()},
                "development_tracks_excluded": True,
                "lsh_settings_frozen": LSH_SETTINGS,
                "ivf_settings_frozen": IVF_SETTINGS,
                "source_commits": {"lsh_and_exact": LSH_COMMIT, "ivf": IVF_COMMIT},
                "timing_boundaries": {
                    "query_sample_size": TIMING_QUERIES,
                    "single_query_calls": "Warm, resident GPU query and index; GPU synchronized before and after each lookup; methods interleaved on the same fixed query sample.",
                    "single_query_includes": "GPU search compute, kernel launches and host call overhead through completion.",
                    "single_query_excludes": "Audio decoding, fingerprint extraction, query host-to-device transfer, index construction, and the analysis-only LSH top-two margin pass.",
                    "batch_throughput": f"three synchronized passes over the same held-out query vectors in chunks of {BATCH_SIZE}.",
                },
                "open_set": {
                    "source_revision": MARGIN_COMMIT,
                    "test_queries": int(np.count_nonzero(open_queries["phase"] == "test")),
                    "thresholds": rules,
                    "removed_songs_excluded_from_each_catalogue": True,
                },
                "medium_large_pair_validation": pair_validation,
            },
            "medium": medium_result,
            "large": large_result,
            "speed_comparison": speed_comparison,
            "limitations": [
                "FMA Large is the 30-second excerpt collection, though the processed catalogue preserves longer segment sequences; see the pair-validation summary.",
                "Closed-set queries retain the original medium head and synthetic distortions.",
                "Frozen open-set thresholds are transferred without recalibration to the larger catalogue.",
                "Search timings include lookup only after reference and index data are resident on the GPU.",
                "LSH score-gap is an analysis-only metric from the exact-score slab restricted to the union of its pinned multprobe buckets; it is excluded from LSH lookup timings.",
            ],
        }
        output = Path(args.output)
        if not output.is_absolute():
            output = REPO / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        report = output.with_suffix(".md")
        report.write_text(render_report(result), encoding="utf-8")
        print(json.dumps(result, indent=2))


def render_report(result):
    lines = [
        "# FMA medium-to-large retrieval scaling",
        "",
        f"GPU: {result['gpu']}",
        "",
        "| Catalogue | Windows | Tracks |",
        "|---|---:|---:|",
    ]
    for name in ("medium", "large"):
        row = result[name]
        lines.append(f"| {name.title()} | {row['reference_windows']:,} | {row['reference_tracks']:,} |")
    pair = result["protocol"]["medium_large_pair_validation"]
    long_tracks = pair["large_tracks_with_segment_idx_above_4"]
    long_track_text = ", ".join(long_tracks) if long_tracks else "none"
    lines.extend(
        [
            "",
            f"The Large archive contains {pair['added_distinct_track_ids']:,} added track IDs and preserves every matched Medium window. Its maximum `segment_idx` is {pair['large_max_segment_idx']}; tracks above the common 0–4 excerpt range: {long_track_text}.",
        ]
    )
    lines.extend(["", "## Closed-set identification", "", "| Catalogue | Family | Exact | Accuracy-first LSH | Spherical IVF |", "|---|---|---:|---:|---:|"])
    for name in ("medium", "large"):
        for family in ("standard", "tiktok"):
            methods = result[name]["methods"][family]
            values = []
            for method in ("exact", "lsh", "ivf"):
                cell = methods[method]
                values.append(f"{cell['correct']:,}/{cell['queries']:,} ({cell['accuracy']:.2%})")
            lines.append(f"| {name.title()} | {family} | " + " | ".join(values) + " |")
    lines.extend(["", "## Paired single-query retrieval time", "", "| Catalogue | Method | Median (ms) | 95th percentile (ms) |", "|---|---|---:|---:|"])
    for name in ("medium", "large"):
        for method in ("exact", "lsh", "ivf"):
            row = result[name]["paired_timing"][method]
            lines.append(f"| {name.title()} | {method.upper()} | {row['median_ms']:.4f} | {row['p95_ms']:.4f} |")
    lines.extend(["", "## Batch throughput", "", "| Catalogue | Method | Queries/s |", "|---|---|---:|"])
    for name in ("medium", "large"):
        for method in ("exact", "lsh", "ivf"):
            row = result[name]["batch_throughput"][method]
            lines.append(f"| {name.title()} | {method.upper()} | {row['queries_per_second']:,.0f} |")
    lines.extend(
        [
            "",
            "## Speedup and catalogue growth",
            "",
            "| Catalogue | LSH median speedup vs exact | IVF median speedup vs exact |",
            "|---|---:|---:|",
        ]
    )
    for catalogue in ("medium", "large"):
        ratios = result["speed_comparison"]["speedup_vs_exact"][catalogue]
        lines.append(f"| {catalogue.title()} | {ratios['lsh']:.2f}× | {ratios['ivf']:.2f}× |")
    lines.extend(
        [
            "",
            "| Method | Median latency change, Large vs Medium | p95 latency change, Large vs Medium | Batch throughput change, Large vs Medium |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, row in result["speed_comparison"]["large_vs_medium_change"].items():
        lines.append(
            f"| {method.upper()} | {row['median_latency_percent']:+.1f}% | "
            f"{row['p95_latency_percent']:+.1f}% | {row['batch_throughput_percent']:+.1f}% |"
        )
    boundaries = result["protocol"]["timing_boundaries"]
    lines.extend(
        [
            "",
            "Timing used the same 192 resident GPU query vectors per catalogue, with methods warmed and interleaved; every single-query call was synchronized before and after. Measurements include lookup compute, kernel launches, and host call overhead through GPU completion. They exclude audio processing, query transfer, index construction, and the analysis-only LSH margin pass.",
            f"Batch throughput used {boundaries['batch_throughput']}.",
        ]
    )
    lines.extend(
        [
            "",
            "## Paired closed-set changes by family",
            "",
            "| Family | Method | Lost correct | Gained correct | Accuracy change, pp |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for family in ("standard", "tiktok"):
        for method, row in result["large"]["paired_changes_vs_medium"][family].items():
            lines.append(
                f"| {family} | {method.upper()} | {row['lost_correct']} | {row['gained_correct']} | "
                f"{row['accuracy_change_percentage_points']:+.3f} |"
            )
    lines.extend(["", "## Open-set fixed thresholds", "", "Thresholds were selected on the earlier development set and transferred unchanged. The designated missing-song IDs were removed from both catalogues before scoring.", "", "| Catalogue | Method | Rule | Known correct accepted | Unknown false accepts |", "|---|---|---|---:|---:|"])
    for catalogue in ("medium", "large"):
        for method in ("exact", "ivf", "lsh"):
            for rule, row in result[catalogue]["open_set"][method].items():
                if row.get("available") is False or "known_correct" not in row:
                    continue
                lines.append(
                    f"| {catalogue.title()} | {method.upper()} | {rule.replace('_', ' ')} | "
                    f"{row['known_correct']:,}/{row['known_queries']:,} | "
                    f"{row['unknown_false_accepts']:,}/{row['unknown_queries']:,} |"
                )
    lines.extend(
        [
            "",
            "| Catalogue | Method | Rule | Family | Known correct rate | Unknown false-accept rate |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for catalogue in ("medium", "large"):
        for method in ("exact", "ivf", "lsh"):
            for rule, metrics in result[catalogue]["open_set"][method].items():
                if "known_correct" not in metrics:
                    continue
                for family, family_metrics in metrics["by_family"].items():
                    known_rate = family_metrics["known_correct_rate"]
                    unknown_rate = family_metrics["unknown_false_acceptance_rate"]
                    known_display = "n/a" if known_rate is None else f"{known_rate:.2%}"
                    unknown_display = "n/a" if unknown_rate is None else f"{unknown_rate:.2%}"
                    lines.append(
                        f"| {catalogue.title()} | {method.upper()} | {rule.replace('_', ' ')} | {family} | "
                        f"{known_display} | {unknown_display} |"
                    )
    tie_audit = result["medium"]["open_set"].get("pr6_score_only_tie_reconciliation")
    if tie_audit:
        lines.extend(
            [
                "",
                "### PR6 score-only count reconciliation",
                "",
                f"PR6 records {tie_audit['pr6_known_correct_score_only_accepts']}/1,024 known correct accepts; the pinned CuPy row-order exact result gives {tie_audit['cupy_strict_known_correct_score_only_accepts']}/1,024 strict top-1 accepts. Counting tied top scores as correct gives {tie_audit['tie_aware_known_correct_score_only_accepts']}/1,024, matching PR6. The frozen threshold is unchanged.",
                "",
                "| Test query | Family / distortion | Known track | CuPy winner | Equal PR6 winner / runner-up score | Score minus threshold |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for tie in tie_audit["tie_cases"]:
            lines.append(
                f"| {tie['test_query_index']} | {tie['family']} / {tie['distortion']} | {tie['track_id']} | "
                f"{tie['cupy_winner']} | {tie['pr6_winner_score']:.9f} | "
                f"+{tie['winner_score_minus_threshold']:.9f} |"
            )
        lines.append(tie_audit["interpretation"])
    lines.extend(["", "## Scope", ""])
    lines.extend(f"- {limitation}" for limitation in result["limitations"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-reference", default="data/large/references.npz", help="FMA Large fingerprint archive")
    parser.add_argument("--output", default="reports/catalogue-growth.json", help="JSON output path")
    run(parser.parse_args())
