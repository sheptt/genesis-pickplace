"""Supervised behavior cloning: phase/context -> Cartesian command and grip."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .dataset import make_features, training_arrays, REPO, REVISION


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(36, 128), nn.SiLU(), nn.Linear(128, 128),
                                 nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 5))
        self.register_buffer("position_mean", torch.zeros(3))
        self.register_buffer("position_std", torch.ones(3))

    def forward(self, x):
        return self.net(x)

    @torch.inference_mode()
    def predict(self, xy, mass, phase):
        x = torch.from_numpy(make_features(xy, mass, phase)).to(self.position_mean.device)
        y = self(x)
        pos = y[:, :3] * self.position_std + self.position_mean
        closed = y[:, 3:4].sigmoid() >= .5
        grip = torch.where(closed, y[:, 4:5].sigmoid(), 0.)
        return torch.cat([pos, grip], dim=1).cpu().numpy()


def loss_terms(pred, pos, closed, force):
    position_loss = nn.functional.mse_loss(pred[:, :3], pos)
    grip_loss = nn.functional.binary_cross_entropy_with_logits(pred[:, 3:4], closed)
    force_loss = ((pred[:, 4:5].sigmoid() - force).square() * closed).sum() / closed.sum().clamp_min(1)
    return position_loss + .25 * grip_loss + force_loss


def train(splits, output="outputs/model", epochs=450, seed=7, device="cpu"):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_arrays = training_arrays(splits["train"])
    val_arrays = training_arrays(splits["validation"])
    model = Policy().to(device)
    model.position_mean.copy_(torch.from_numpy(train_arrays[1].mean(0)).to(device))
    model.position_std.copy_(torch.from_numpy(train_arrays[1].std(0).clip(.03)).to(device))
    def tensors(arrays):
        xs = [torch.from_numpy(a).to(device) for a in arrays]
        xs[1] = (xs[1] - model.position_mean) / model.position_std
        return xs
    x, p, c, f = tensors(train_arrays)
    vx, vp, vc, vf = tensors(val_arrays)
    optim = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    best, best_weights, history = float("inf"), None, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x), device=device)
        total = 0.
        for ix in order.split(512):
            optim.zero_grad(set_to_none=True)
            loss = loss_terms(model(x[ix]), p[ix], c[ix], f[ix])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optim.step()
            total += loss.item() * len(ix)
        model.eval()
        with torch.no_grad():
            valid = float(loss_terms(model(vx), vp, vc, vf))
        history.append({"epoch": epoch, "train_loss": total / len(x), "validation_loss": valid})
        if valid < best:
            best = valid
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"epoch {epoch:4d} train={total/len(x):.6f} validation={valid:.6f}", flush=True)
    model.load_state_dict(best_weights)
    model.cpu().eval()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    # Only a state_dict, never a serialized Python model object.
    torch.save(model.state_dict(), out / "policy.pt")
    meta = {"algorithm": "phase-conditioned MLP behavior cloning", "repo": REPO, "revision": REVISION,
            "seed": seed, "epochs": epochs, "best_epoch": best_epoch,
            "horizon": 157, "dt": 1/60,
            "train_ids": [e["id"] for e in splits["train"]],
            "validation_ids": [e["id"] for e in splits["validation"]],
            "test_ids": [e["id"] for e in splits["test"]],
            "initial_qpos": splits["train"][0]["state"][0, :9].tolist(),
            "xy_min": np.min([e["obj_xy"] for e in splits["train"]], axis=0).tolist(),
            "xy_max": np.max([e["obj_xy"] for e in splits["train"]], axis=0).tolist(),
            "limitations": "Uses initial xy, simulator-known mass and phase; no vision, language understanding or slip recovery."}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, meta, history


def load_policy(directory="outputs/model"):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    model = Policy()
    model.load_state_dict(torch.load(directory / "policy.pt", map_location="cpu", weights_only=True))
    return model.eval(), meta


def evaluate_offline(model, episodes):
    predictions, truth = [], []
    for ep in episodes:
        phase = np.arange(1, len(ep["state"])) / (len(ep["state"]) - 1)
        predictions.append(model.predict(ep["obj_xy"], ep["magnitude"], phase))
        truth.append(ep["action"][1:, [0, 1, 2, 7]])
    pred, target = np.concatenate(predictions), np.concatenate(truth)
    return {"position_rmse_m": float(np.sqrt(np.mean((pred[:, :3] - target[:, :3])**2))),
            "grip_open_close_accuracy": float(np.mean((pred[:, 3] > .05) == (target[:, 3] > .05))),
            "grip_force_fraction_mae": float(np.abs(pred[:, 3] - target[:, 3]).mean()),
            "frames": len(pred), "episodes": len(episodes),
            "note": "Prediction error only. Not a physical pick-and-place success rate."}
