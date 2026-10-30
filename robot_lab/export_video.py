"""Concatenate five Genesis policy recordings without captions or sound."""
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
from .policy import load_policy

def export(output="outputs/submission_80s.mp4"):
    root = Path("outputs")
    _, metadata = load_policy(root / "model")
    ids = metadata["test_ids"]
    if len(ids) != 5:
        raise ValueError("5件のテストを含むモデルが必要です。再学習してください。")
    current_hash = hashlib.sha256((root / "model/policy.pt").read_bytes()).hexdigest()
    clips = []
    for eid in ids:
        directory = root / f"policy_{eid}"
        try:
            result = json.loads((directory / "result.json").read_text())
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{eid} の結果がありません。5件すべてを実行してください。") from exc
        if result["mode"] != "policy" or not result["held_out_test_episode"] or result["model_sha256"] != current_hash:
            raise ValueError(f"{eid} は現在の学習済みモデルによるテスト結果ではありません。")
        path = directory / "rollout.mp4"
        if not path.exists():
            raise FileNotFoundError(f"動画がありません: {path}")
        clips.append(path)
    # Each 30 fps clip is replayed at one quarter speed. Five 3.6 s clips
    # make approximately 72 s, meeting the 1–3 minute requirement.
    with imageio.get_writer(output, fps=30, codec="libx264", quality=7, macro_block_size=16, ffmpeg_params=["-threads", "2"]) as writer:
        for clip in clips:
            with imageio.get_reader(clip) as reader:
                for frame in reader:
                    for _ in range(4):
                        writer.append_data(frame)
    print(f"Saved {output}: 5 clips, 0.25x speed, no captions or audio.")

if __name__ == "__main__":
    export()
