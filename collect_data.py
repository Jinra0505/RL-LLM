# train_task1_collect.py
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

import TD3
import utils
from DSO_environment.dso_env import make_env
from DSO_environment.dso_flow import price_data

# ========== Feature Collector ==========
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Optional

@dataclass
class FeatureLayout:
    n_bus: int = 33
    idx_pgrid_scaled: int = 33
    idx_qgrid_scaled: int = 34
    scale_opendss: float = 5000.0
    idx_linear_base: int = 35
    idx_pv_avail_rel: int = 0
    idx_pv_curt_rel: int = 1
    scale_pv: float = 10.0

@dataclass
class WindowCfg:
    win_sizes: Iterable[int] = (4, 8)
    eps: float = 1e-9

class LLMTaskFeatureCollector:
    def __init__(self, layout: FeatureLayout = FeatureLayout(), win_cfg: WindowCfg = WindowCfg()):
        self.L = layout
        self.W = win_cfg
        self.reset()

    def reset(self):
        self.v_hist = deque(maxlen=max(self.W.win_sizes))
        self.prev_state: Optional[np.ndarray] = None
        self.cum_cost_proxy: float = 0.0

    def _f(self, x, default=0.0) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) else default
        except Exception:
            return default

    def build_router_input(self, state: np.ndarray, *, t: int, price_now: float) -> Dict[str, Any]:
        L, W = self.L, self.W
        s = np.asarray(state, dtype=float)

        # Voltage stats
        V = s[:L.n_bus]
        abs_dev = np.abs(V - 1.0)
        mean_abs_dev = float(abs_dev.mean())
        max_abs_dev  = float(abs_dev.max())
        step_L1 = 0.0
        if self.prev_state is not None:
            V_prev = self.prev_state[:L.n_bus]
            step_L1 = float(np.mean(np.abs(V - V_prev)))
        self.v_hist.append(V.copy())

        win_stats = {}
        for ws in self.W.win_sizes:
            if len(self.v_hist) >= ws:
                arr = np.stack(list(self.v_hist)[-ws:], axis=0)
                win_stats[f"win_var_{ws}"]   = float(np.var(arr))
                win_stats[f"win_range_{ws}"] = float(arr.max() - arr.min())

        # Cost
        Pgrid_scaled = self._f(s[L.idx_pgrid_scaled])
        P_grid = - Pgrid_scaled * L.scale_opendss
        step_cost_proxy = price_now * max(P_grid, 0.0)
        self.cum_cost_proxy += step_cost_proxy

        # PV
        lin = L.idx_linear_base
        pv_avail   = self._f(s[lin + L.idx_pv_avail_rel]) * L.scale_pv
        pv_curtail = self._f(s[lin + L.idx_pv_curt_rel])  * L.scale_pv
        pv_used = max(pv_avail - pv_curtail, 0.0)
        pv_util = pv_used / (pv_avail + W.eps)

        out = {
            "voltage": {
                "mean_abs_dev": mean_abs_dev,
                "max_abs_dev": max_abs_dev,
                "step_L1": step_L1,
                **win_stats
            },
            "cost": {
                "price": price_now,
                "P_grid_proxy": P_grid,
                "step_cost_proxy": step_cost_proxy,
                "cum_cost_proxy": self.cum_cost_proxy
            },
            "pv": {
                "avail": pv_avail,
                "curtail": pv_curtail,
                "util": pv_util
            },
            "meta": {"t": int(t)}
        }

        self.prev_state = s.copy()
        return out

# ========== Main Training ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="DSOEnv-v0")
    parser.add_argument("--task_id", default=0, type=int)  # Task 1 = Minimize total cost
    parser.add_argument("--episodes", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cuda", default=0, type=int)

    # TD3 parameters
    parser.add_argument("--start_timesteps", default=1000, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)

    parser.add_argument("--output", default="./collectdata/train_task1_10.jsonl", type=str)

    args = parser.parse_args()

    TD3.set_device(args.cuda)
    utils.set_device(args.cuda)
    np.random.seed(args.seed)

    # Env
    args.num_train_tasks = 10
    env = make_env(args)
    env.action_space.seed(args.seed)

    # State/action dim
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # TD3
    policy = TD3.TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise * max_action,
        noise_clip=args.noise_clip * max_action,
        policy_freq=args.policy_freq,
    )
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Collector
    collector = LLMTaskFeatureCollector()

    fout = open(args.output, "w", encoding="utf-8")
    total_steps, rewards = 0, []

    # Training loop
    for ep in range(args.episodes):
        state, _ = env.reset()
        if isinstance(state, dict):
            state = np.concatenate([state['observation'], state['desired_goal']], axis=0)

        collector.reset()
        done, ep_reward, t = False, 0, 0

        while not done:
            if total_steps < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = policy.select_action(state)
                action = (action + np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )

            next_state, reward, terminated, truncated, _ = env.step(action)
            if isinstance(next_state, dict):
                next_state = np.concatenate([next_state['observation'], next_state['desired_goal']], axis=0)

            done_bool = float(terminated or truncated)
            replay_buffer.add(state, action, next_state, reward, done_bool)

            # Train TD3
            if total_steps >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)

            # Features
            price_now = float(price_data[t % len(price_data)])
            feats = collector.build_router_input(state=next_state, t=t, price_now=price_now)

            # Save trajectory step
            rec = {
                "episode": ep,
                "t": t,
                "state": state.tolist(),
                "action": action.tolist(),
                "reward": reward,
                "next_state": next_state.tolist(),
                "done": bool(done_bool),
                "features": feats
            }
            fout.write(json.dumps(rec) + "\n")

            state = next_state
            ep_reward += reward
            total_steps += 1
            t += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        print(f"Episode {ep} | Reward: {ep_reward:.2f}")

    fout.close()
    print(f"✅ Trajectory saved to {args.output}")

    # Plot reward curve
    os.makedirs("results", exist_ok=True)
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Task 1 Training with TD3")
    plt.legend()
    plt.grid()
    plt.savefig("results/task1_reward_curve.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
