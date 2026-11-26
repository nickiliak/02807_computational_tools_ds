"""Extend the frozen FMA medium search catalogue using the official FMA large ZIP.

The local ZIP may still be downloading. Only complete members are decoded. Final
publication requires the official archive checksum and a complete track inventory.
"""

import argparse
import bz2
import hashlib
import io
import itertools
import json
import struct
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from afp.data import load_audio, segment
from afp.embed import MuQBackbone
from afp.fingerprint import load_head

LARGE_SHA1 = "497109f4dd721066b5ce5e5f250ec604dc78939e"
REF_SHA256 = "174234b82db42e42839b26b7d52372b908075d5724451037b26e4a865d33a9b9"
HEAD_SHA256 = "4a5fb50d5dea344c0fdc803e9c825b6dfbd39b5dc86bafdce96c261e63801c6f"
SNAPSHOT = "0562a57814f6f8bbd9fdea0a25921a2fce1a841a"
HEADER = struct.Struct("<4s5H3I2H")
FIELDS = ("vectors", "track_id", "segment_idx", "start_s", "rms")


def digest(path, algorithm):
    with path.open("rb") as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def save_npz(path, **arrays):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as out:
        np.savez(out, **arrays)
    temp.replace(path)


def available(path, end, follow):
    last_size = -1
    while (size := path.stat().st_size) < end:
        if size != last_size:
            changed = time.monotonic()
            last_size = size
        if not follow or time.monotonic() - changed > 300:
            raise EOFError(f"Archive stopped at {size:,} bytes, need {end:,}. Resume the transfer and rerun.")
        time.sleep(0.5)


def completed_archive(path, follow):
    """The central directory may arrive after the last MP3 during streaming."""
    last_size = -1
    while True:
        try:
            with zipfile.ZipFile(path) as archive:
                return sum(name.endswith(".mp3") for name in archive.namelist())
        except zipfile.BadZipFile:
            size = path.stat().st_size
            if size != last_size:
                changed, last_size = time.monotonic(), size
            if not follow or time.monotonic() - changed > 300:
                raise
            time.sleep(0.5)


def additions(path, medium, follow, seen):
    """Read local member headers in download order, validating the shared MP3 identities."""
    with path.open("rb") as stream:
        while True:
            available(path, stream.tell() + HEADER.size, follow)
            signature, _, flags, method, _, _, crc, size, raw_size, nname, extra = HEADER.unpack(
                stream.read(HEADER.size)
            )
            if signature == b"PK\x01\x02":
                break
            if signature != b"PK\x03\x04" or flags & 9 or size == 0xFFFFFFFF:
                raise ValueError("Unexpected ZIP local header; this reader targets the official FMA Large archive.")
            available(path, stream.tell() + nname + extra + size, follow)
            name = stream.read(nname).decode()
            stream.seek(extra, 1)
            start = stream.tell()
            stream.seek(size, 1)
            if not name.endswith(".mp3"):
                continue
            track = int(Path(name).stem)
            if track in seen:
                raise ValueError(f"Duplicate track {track}")
            seen.add(track)
            if track in medium:
                if medium[track] != (crc, raw_size):
                    raise ValueError(f"Overlapping audio differs for track {track}")
            else:
                yield track, start, size, crc, method


def decode(path, entry):
    track, start, size, crc, method = entry
    with path.open("rb") as stream:
        stream.seek(start)
        payload = stream.read(size)
    if method == zipfile.ZIP_BZIP2:
        payload = bz2.decompress(payload)
    elif method != zipfile.ZIP_STORED:
        raise ValueError(f"Unexpected compression method {method}")
    if zlib.crc32(payload) != crc:
        raise ValueError(f"CRC mismatch for track {track}")
    try:
        audio = load_audio(io.BytesIO(payload))
    except (RuntimeError, ValueError, OSError) as exc:
        return track, None, str(exc)
    pcm = np.round(np.clip(audio, -1, 1) * 32767).astype(np.int16)
    windows, _ = segment(pcm.astype(np.float32) / 32768)
    return track, windows, "" if len(windows) else "shorter than one window"


def prepare(pool, path, entries):
    windows, ids, indices, levels, failures = [], [], [], [], []
    for track, audio, error in pool.map(lambda row: decode(path, row), entries):
        if error:
            failures.append([track, error])
            continue
        rms = np.sqrt(np.mean(audio**2, axis=1))
        rows = np.flatnonzero(rms >= 1e-3)
        windows.append(audio[rows])
        ids.extend([track] * len(rows))
        indices.extend(rows)
        levels.extend(rms[rows])
    return windows, np.array(ids, np.int32), np.array(indices, np.int16), np.array(levels, np.float32), failures


@torch.inference_mode()
def encode(backbone, head, windows, batch):
    features = []
    fallbacks = 0
    for first in range(0, len(windows), batch):
        x = torch.from_numpy(windows[first : first + batch]).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = backbone.model(x, output_hidden_states=True).last_hidden_state
        pooled = hidden.float().mean(dim=1)
        if not torch.isfinite(pooled).all():
            pooled = backbone.model(x, output_hidden_states=True).last_hidden_state.float().mean(dim=1)
            fallbacks += 1
        # The medium pipeline cached pooled MuQ features in FP16 before the FP32 head.
        features.append(pooled.half().float())
    vectors = head(torch.cat(features)).cpu().numpy()
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-4):
        raise ValueError("Invalid GPU projection")
    return vectors, fallbacks


def run(args):
    root, medium = args.output, args.medium
    root.mkdir(parents=True, exist_ok=True)
    shards = root / "shards"
    shards.mkdir(exist_ok=True)
    head_file = medium / "models/projection_head.pt"
    ref_file = medium / "data/processed/fingerprints/references.npz"
    if digest(head_file, "sha256") != HEAD_SHA256 or digest(ref_file, "sha256") != REF_SHA256:
        raise ValueError("Baseline inputs differ from the measured medium run")
    with zipfile.ZipFile(medium / "data/raw/fma_medium.zip") as archive:
        shared = {
            int(Path(i.filename).stem): (i.CRC, i.file_size) for i in archive.infolist() if i.filename.endswith(".mp3")
        }
    torch.set_num_threads(1)
    backbone = MuQBackbone(
        "cuda", str((medium / "hf-cache/hub/models--OpenMuQ--MuQ-large-msd-iter/snapshots" / SNAPSHOT).resolve())
    )
    head = load_head(head_file).cuda().eval()
    seen, paths = set(), []
    new = additions(args.archive, shared, args.follow, seen)
    processed = vectors_done = fallback_total = 0
    started = time.monotonic()

    def batches():
        while group := list(itertools.islice(new, 96)):
            yield group

    with ThreadPoolExecutor(args.workers) as pool, ThreadPoolExecutor(1) as ahead:

        def queue(group):
            path = shards / f"{group[0][0]:06}-{group[-1][0]:06}.npz"
            future = None if path.exists() else ahead.submit(prepare, pool, args.archive, group)
            return path, group, future

        iterator = iter(batches())
        current = next(iterator, None)
        pending = queue(current) if current else None
        while pending:
            path, group, future = pending
            following = next(iterator, None)
            pending = queue(following) if following else None
            track_ids = np.array([g[0] for g in group], dtype=np.int32)
            if future:
                waves, ids, indices, rms, errors = future.result()
                vec, fallback = (
                    encode(backbone, head, np.concatenate(waves), args.batch)
                    if len(ids)
                    else (np.empty((0, 256), np.float32), 0)
                )
                save_npz(
                    path,
                    vectors=vec,
                    track_id=ids,
                    segment_idx=indices,
                    start_s=indices.astype(np.float32) * 5,
                    rms=rms,
                    source_tracks=track_ids,
                    failures=json.dumps(errors),
                    fp32_fallbacks=fallback,
                )
            with np.load(path) as done:
                if not np.array_equal(done["source_tracks"], track_ids):
                    raise ValueError(f"Checkpoint disagrees with archive: {path}")
                vectors_done += len(done["track_id"])
                fallback_total += int(done["fp32_fallbacks"])
            processed += len(group)
            paths.append(path)
            if len(paths) % 4 == 0 or len(paths) == 1:
                print(
                    f"{processed}/81574 added tracks, {vectors_done} windows, {time.monotonic() - started:.1f}s",
                    flush=True,
                )

    if len(seen) != 106574 or len(shared) != 25000 or not set(shared).issubset(seen) or processed != 81574:
        raise ValueError(f"Incomplete catalogue: {len(seen)} source tracks, {processed} additions")
    print("Checking complete Large archive SHA-1 before publishing fingerprints", flush=True)
    if completed_archive(args.archive, args.follow) != len(seen) or digest(args.archive, "sha1") != LARGE_SHA1:
        raise ValueError("Official FMA Large SHA-1 mismatch")
    arrays = {}
    with np.load(ref_file) as baseline:
        for key in FIELDS:
            pieces = [baseline[key]]
            for path in paths:
                with np.load(path) as part:
                    pieces.append(part[key])
            arrays[key] = np.concatenate(pieces)
    with np.load(ref_file) as base:
        for k in FIELDS:
            np.testing.assert_array_equal(arrays[k][: len(base[k])], base[k])
    save_npz(root / "references.npz", **arrays)
    failures = []
    for path in paths:
        with np.load(path) as part:
            failures.extend(json.loads(str(part["failures"])))
    summary = {
        "source_tracks": len(seen),
        "added_source_tracks": processed,
        "searchable_tracks": len(np.unique(arrays["track_id"])),
        "windows": len(arrays["track_id"]),
        "added_windows": vectors_done,
        "fp32_fallbacks": fallback_total,
        "failed_new_tracks": failures,
        "archive_sha1": LARGE_SHA1,
        "medium_reference_sha256": REF_SHA256,
        "projection_sha256": HEAD_SHA256,
        "backbone_revision": SNAPSHOT,
        "references_sha256": digest(root / "references.npz", "sha256"),
        "gpu": torch.cuda.get_device_name(),
        "workers": args.workers,
        "batch": args.batch,
    }
    (root / "catalogue.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print({k: v for k, v in summary.items() if k != "failed_new_tracks"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--medium", type=Path, default=Path("data/medium"))
    parser.add_argument("--output", type=Path, default=Path("data/large"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--follow", action="store_true", help="Read complete ZIP members while the archive downloads")
    run(parser.parse_args())
