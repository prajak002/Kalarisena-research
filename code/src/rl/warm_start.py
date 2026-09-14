"""Cross-stage weight transfer for the curriculum's warm-start chain
(paper Table 34: Stage B loads tracking_best, Stage C loads com_best,
Stage D trains from the momentum-refined policy's perturbed rollouts).

The paper's stages don't share an observation width - Stage A is 124-d,
Stage B/C add a 6-d CoM/support or momentum block (130-d), Stage D adds a
2-d impact-sensing block (126-d) - so a plain PPO.load(ckpt, env=new_env)
fails on the first layer's input shape. This does the transfer SB3 itself
doesn't: build a fresh policy sized for the new environment, then copy
every weight whose shape already matches (all of the second hidden layer,
the action head, the value head - everything downstream of the first
layer inherits completely), and for the first layer of policy_net and
value_net specifically, copy the old weights into the matching leading
columns and leave the new columns at their fresh initialization. Nothing
about the network's already-learned behavior is discarded; only the new
input dimensions start blank.
"""

from __future__ import annotations

import torch


def warm_start_from_smaller_obs(new_model, old_ckpt_path: str, device: str = "cpu") -> None:
    from stable_baselines3 import PPO

    old_model = PPO.load(old_ckpt_path, device=device)
    old_sd = old_model.policy.state_dict()
    new_sd = new_model.policy.state_dict()

    first_layer_keys = {"mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"}
    transferred, first_layer_partial = [], []

    for key, old_val in old_sd.items():
        if key not in new_sd:
            continue
        new_val = new_sd[key]
        if old_val.shape == new_val.shape:
            new_sd[key] = old_val.clone()
            transferred.append(key)
        elif key in first_layer_keys and old_val.shape[0] == new_val.shape[0] and old_val.shape[1] < new_val.shape[1]:
            merged = new_val.clone()
            with torch.no_grad():
                merged[:, : old_val.shape[1]] = old_val
            new_sd[key] = merged
            first_layer_partial.append((key, old_val.shape, new_val.shape))
        else:
            raise ValueError(
                f"warm_start_from_smaller_obs: can't reconcile shapes for {key}: "
                f"old {tuple(old_val.shape)} vs new {tuple(new_val.shape)}"
            )

    new_model.policy.load_state_dict(new_sd)
    print(f"warm-started from {old_ckpt_path}: "
          f"{len(transferred)} params copied exactly, "
          f"{len(first_layer_partial)} first-layer params partially copied "
          f"({first_layer_partial})")
