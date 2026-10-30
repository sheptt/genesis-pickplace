"""Physical Genesis rollout. No welds, object teleportation, or scripted fallback."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

# These must be set before importing Genesis / OpenGL.
# Docker images install libraries as root; keep compilation caches user-writable.
_cache_root = f"/tmp/genesis-pickplace-{os.getuid()}" if hasattr(os, "getuid") else str(Path.home() / ".cache" / "genesis-pickplace")
os.environ.setdefault("XDG_CACHE_HOME", _cache_root)
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(os.environ["XDG_CACHE_HOME"]) / "numba"))
os.environ.setdefault("GS_CACHE_FILE_PATH", str(Path(os.environ["XDG_CACHE_HOME"]) / "genesis"))
os.environ.setdefault("QD_OFFLINE_CACHE_FILE_PATH", str(Path(os.environ["XDG_CACHE_HOME"]) / "quadrants"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

GOAL = np.array([.46, 0.])


def assess_success(trace, final_grip):
    """Require lifting AND release AND stable placement; sliding is insufficient."""
    xyz = np.asarray([row["cube_xyz"] for row in trace])
    tail = xyz[-30:]
    lifted = bool(xyz[:, 2].max() > .10)
    placed = bool(np.linalg.norm(xyz[-1, :2] - GOAL) < .045)
    supported = bool(.025 < xyz[-1, 2] < .065)
    stable = bool(np.max(np.linalg.norm(tail - tail[-1], axis=1)) < .005)
    released = bool(final_grip < .05)
    return {"success": lifted and placed and supported and stable and released,
            "lifted": lifted, "placed": placed, "supported": supported,
            "stable": stable, "released": released,
            "maximum_cube_height_m": float(xyz[:, 2].max()),
            "final_cube_xyz": xyz[-1].tolist(),
            "goal_distance_m": float(np.linalg.norm(xyz[-1, :2] - GOAL))}


def rollout(mode="policy", episode_id="ep0135", data_dir="data/compact",
            model_dir="outputs/model", output_dir="outputs/rollout", video=True):
    from .dataset import load_dataset
    from .policy import load_policy
    import genesis as gs
    import torch
    torch.set_num_threads(4)
    episodes, _ = load_dataset(data_dir)
    ep = next((e for e in episodes if e["id"] == episode_id), None)
    if ep is None:
        raise ValueError(f"Episode not found: {episode_id}")
    if mode not in ("policy", "replay"):
        raise ValueError("mode must be policy or replay")
    model, metadata = load_policy(model_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    xy, mass = ep["obj_xy"], float(ep["magnitude"])
    horizon = metadata["horizon"]
    # Only replay reads demonstration commands. Policy reads initial context only.
    if mode == "replay":
        commands = ep["action"][1:, [0, 1, 2, 7]].copy()
        initial_qpos = ep["state"][0, :9]
    else:
        commands = model.predict(xy, mass, np.arange(1, horizon + 1) / horizon)
        initial_qpos = np.array(metadata["initial_qpos"])
    if not np.isfinite(commands).all():
        raise ValueError("Non-finite policy output")
    if np.any(commands[:, :3] < [.25, -.35, .10]) or np.any(commands[:, :3] > [.85, .35, .55]):
        raise ValueError("Command out of supported workspace; inspect model")
    np.save(out / "commands.npy", commands)

    gs.init(backend=gs.cpu, seed=7, logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60, substeps=4), show_viewer=False)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.),
                     surface=gs.surfaces.Default(color=(.75, .72, .65, 1.)))
    robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    cube = scene.add_entity(
        gs.morphs.Box(size=(.04, .04, .04), pos=(*xy, .02)),
        material=gs.materials.Rigid(rho=mass / .04**3, friction=1.),
        surface=gs.surfaces.Default(color=(.90, .035, .025, 1.)))
    scene.add_entity(gs.morphs.Box(size=(.13, .13, .02), pos=(*GOAL, .01), fixed=True),
                     surface=gs.surfaces.Default(color=(.10, .38, .85, 1.)))
    camera = None
    if video:
        # Viewed from +y: smaller world x is screen-right.
        camera = scene.add_camera(res=(640, 480), pos=(.55, 1.45, .85),
                                  lookat=(.48, 0., .17), fov=38, GUI=False)
    scene.build()
    robot.set_dofs_kp(np.array([4500,4500,3500,3500,2000,2000,2000,100,100]))
    robot.set_dofs_kv(np.array([450,450,350,350,200,200,200,10,10]))
    limits = np.array([87,87,87,87,12,12,12,100,100])
    robot.set_dofs_force_range(-limits, limits)
    robot.set_dofs_position(initial_qpos)  # Initial reset only.
    hand = robot.get_link("hand")
    arm_idx, finger_idx = np.arange(7), np.arange(7, 9)
    trace = []
    writer = None
    if camera is not None:
        import imageio.v2 as imageio
        from PIL import Image
        writer = imageio.get_writer(str(out / "rollout.mp4"), fps=30, codec="libx264",
                                   quality=7, macro_block_size=16)
    try:
        # 157 learned commands + 60 steps holding the final command to assess rest.
        for step in range(horizon + 60):
            command = commands[min(step, horizon - 1)]
            if step < horizon:
                q = robot.inverse_kinematics(link=hand, pos=command[:3], quat=[0,1,0,0])
                robot.control_dofs_position(q[:7], arm_idx)
                if command[3] > .05:
                    robot.control_dofs_force(np.full(2, -45 * command[3]), finger_idx)
                else:
                    robot.control_dofs_position(np.array([.04, .04]), finger_idx)
            scene.step()
            p = cube.get_pos().cpu().numpy().copy()
            trace.append({"step": step + 1, "time_s": (step+1)/60,
                          "cube_xyz": p.tolist(), "command": command.tolist(),
                          "hand_xyz": hand.get_pos().cpu().numpy().tolist()})
            if writer is not None and step % 2 == 0:
                rgb, *_ = camera.render()
                frame = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
                writer.append_data(np.asarray(frame))
                if step == 0:
                    frame.save(out / "initial.png")
                if step == horizon + 59 - ((horizon + 59) % 2):
                    frame.save(out / "final.png")
            if (step + 1) % 40 == 0:
                print(f"{mode} {episode_id}: {step+1}/{horizon+60} cube={p.round(3)}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    result = assess_success(trace, float(commands[-1, 3]))
    result.update({"mode": mode, "episode": episode_id, "initial_xy": xy.tolist(),
                   "mass_kg": mass, "held_out_test_episode": episode_id in metadata["test_ids"],
                   "model_sha256": hashlib.sha256((Path(model_dir)/"policy.pt").read_bytes()).hexdigest(),
                   "physics_dt": 1/60, "substeps": 4, "command_steps": horizon,
                   "scene": "Reconstructed simplified scene; red target, blue platform, no distractors."})
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
    with (out / "trace.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step","time_s","cube_x","cube_y","cube_z","command_x","command_y","command_z","grip"])
        for row in trace:
            w.writerow([row["step"],row["time_s"],*row["cube_xyz"],*row["command"]])
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["policy", "replay"], default="policy")
    parser.add_argument("--episode", default="ep0135")
    parser.add_argument("--data-dir", default="data/compact")
    parser.add_argument("--model-dir", default="outputs/model")
    parser.add_argument("--output", default="outputs/rollout")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    rollout(args.mode, args.episode, args.data_dir, args.model_dir, args.output, not args.no_video)
