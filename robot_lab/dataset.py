"""Read public demonstrations. Never synthesize training labels."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request
import numpy as np

REPO = "jalal06/tactile-genesis-heavy-mg-v2"
REVISION = "9eb6af694a4a519a8c47f50076af3dcf20ab3e4d"
DEFAULT_IDS = list(range(0, 216, 9))
FIELDS = ("state", "action", "obj_xy", "magnitude", "positions", "success")


def read_episode(path):
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in FIELDS}
    n = len(result["state"])
    if result["state"].shape != (n, 16) or result["action"].shape != (n, 8) or n != 158:
        raise ValueError(f"Unexpected dataset schema: {path}")
    for k in FIELDS:
        if not np.isfinite(result[k]).all():
            raise ValueError(f"Nonfinite {k}: {path}")
    if result["obj_xy"].shape != (2,) or result["positions"].shape != (4, 2):
        raise ValueError("Invalid object positions")
    if not bool(result["success"]):
        raise ValueError("Source episode is not marked successful")
    if not np.allclose(result["action"][:, 3:7], [0, 1, 0, 0], atol=1e-5):
        raise ValueError("This baseline assumes the source's fixed downward orientation")
    result["id"] = Path(path).stem
    return result


def compact_episode(source, destination):
    ep = read_episode(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **{k: ep[k] for k in FIELDS})
    return {
        "episode": ep["id"],
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "compact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "frames": len(ep["state"]),
    }


def download_dataset(destination="data/compact", ids=DEFAULT_IDS):
    """Download pinned public episodes and retain their compact numeric fields."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    raw = destination.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for i in ids:
        name = f"ep{i:04d}.npz"
        path = raw / name
        if not path.exists():
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
            for attempt in range(3):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "genesis-pickplace-lab/1"})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        temporary = path.with_suffix(".part")
                        with temporary.open("wb") as out:
                            while block := response.read(1024 * 1024):
                                out.write(block)
                    read_episode(temporary)
                    temporary.replace(path)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)
        records.append(compact_episode(path, destination / name))
        print(f"Prepared {name}", flush=True)
    manifest = {"repo": REPO, "revision": REVISION, "declared_license": "MIT",
                "processing": "Keep numeric state/action and scene metadata; discard RGB/tactile arrays.",
                "episodes": records}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_dataset(directory="data/compact"):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["repo"] != REPO or manifest["revision"] != REVISION:
        raise ValueError("Dataset provenance differs from this project's contract")
    episodes = []
    for item in manifest["episodes"]:
        name = item["episode"]
        if not (name.startswith("ep") and name[2:].isdigit()):
            raise ValueError("Invalid episode filename")
        path = directory / f"{name}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["compact_sha256"]:
            raise ValueError(f"Data checksum failed: {name}")
        episodes.append(read_episode(path))
    if len(episodes) < 8:
        raise ValueError("At least eight episodes are required for independent splits")
    return episodes, manifest


def split_episodes(episodes, seed=42):
    ids = np.random.default_rng(seed).permutation(len(episodes))
    n_test = 5
    n_val = max(2, len(ids) // 8)
    if len(ids) <= n_test + n_val:
        raise ValueError("Need more episodes for five independent tests")
    return {"train": [episodes[i] for i in ids[n_test+n_val:]],
            "validation": [episodes[i] for i in ids[n_test:n_test+n_val]],
            "test": [episodes[i] for i in ids[:n_test]]}


def make_features(xy, mass, phase):
    """Initial object coordinates, known simulator mass, and normalized phase.

    This is a learned trajectory generator, NOT a camera or recovery policy.
    Low-level Genesis motor control remains feedback-controlled.
    """
    phase = np.atleast_1d(phase).astype(np.float32)
    xy = (np.asarray(xy, np.float32) - [.64, 0.]) / [.08, .2]
    context = np.tile([xy[0], xy[1], (float(mass) - .7) / .5], (len(phase), 1))
    angles = phase[:, None] * np.arange(1, 17)[None, :] * np.pi
    return np.concatenate([context, phase[:, None], np.sin(angles), np.cos(angles)], axis=1).astype(np.float32)


def training_arrays(episodes):
    inputs, positions, closed, force = [], [], [], []
    for ep in episodes:
        # Source observations are AFTER each step. obs[t] pairs with action[t+1].
        a = ep["action"][1:]
        phase = np.arange(1, len(ep["state"]), dtype=np.float32) / (len(ep["state"]) - 1)
        inputs.append(make_features(ep["obj_xy"], ep["magnitude"], phase))
        positions.append(a[:, :3])
        closed.append((a[:, 7:8] > .05).astype(np.float32))
        force.append(a[:, 7:8])
    return tuple(np.concatenate(x).astype(np.float32) for x in (inputs, positions, closed, force))
