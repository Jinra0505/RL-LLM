
# -*- coding: utf-8 -*-
# This is an adapted version of lesr_train.py that can switch between multiple RL algorithms (SAC / GRPO)
# via --policy. It keeps your original logging and evaluation flow intact.

import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import importlib
import sys
import re

import utils

# Optional imports guarded by name
from algorithms import SAC_new
from algorithms import TD3
from algorithms import DDPG


from DSO_environment.dso_env import make_env


def import_and_reload_module(file_path):
    import importlib.util
    module_name = os.path.basename(file_path).replace('.py', '').replace('-', '_')
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path + '.py')
    imported_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = imported_module
    spec.loader.exec_module(imported_module)
    return imported_module


def eval_policy(policy, args, eval_episodes=10):
    eval_env = make_env(args)
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = policy.exploit(revise_state(state))
            state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated
            avg_reward += reward
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def cal_lipschitz(state_change, reward_change):
    lipschitz = np.zeros([state_dim, ])
    for ii in range(state_dim):
        cur_index = np.argsort(state_change[ii])
        cur_s, cur_r = state_change[ii].copy(), reward_change.squeeze().copy()
        cur_s, cur_r = cur_s[cur_index], cur_r[cur_index]
        cur_lipschitz = np.abs((cur_r[:-1] - cur_r[1:])) / (np.abs((cur_s[:-1] - cur_s[1:])) + 1e-2)
        lipschitz[ii] = cur_lipschitz.max()
    return lipschitz


def detect_llm_enhancement(revise_path):
    if not revise_path or "baseline_noop" in revise_path:
        return False, ""
    
    # 是LLM增强的训练，尝试提取版本号
    version_match = re.search(r'run-v(\d+)-', revise_path)
    if version_match:
        return True, f"_LLM_v{version_match.group(1)}"
    else:
        # 如果无法提取版本号，使用通用标注
        return True, "_LLM_enhanced"

# class RewardNormalizer:
#     def __init__(self, ema=0.99, eps=1e-8):
#         self.ema_abs = 0.0
#         self.ema = ema
#         self.eps = eps

#     def _ema_update(self, old, new):
#         return self.ema * old + (1.0 - self.ema) * new

#     def normalize(self, r):
#         # 估计奖励尺度的指数滑动平均，然后用 tanh 压到 [-5,5]
#         self.ema_abs = self._ema_update(self.ema_abs, abs(float(r)))
#         scale = self.ema_abs + self.eps
#         return float(6.0 * np.tanh(r / scale))   # <-- 这里从 tanh(...) 改成 5 * tanh(...)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="SAC", help="RL algorithm to use")
    parser.add_argument("--env", default="DSOEnv-v0")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--start_timesteps", default=25e3, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")

    parser.add_argument("--revise_path", default="", type=str)
    parser.add_argument("--version", default="", type=str)

    parser.add_argument("--sid_result_path", default="", type=str)
    parser.add_argument("--corr_result_path", default="", type=str)
    parser.add_argument("--corr_tau", default=0.005, type=float)
    parser.add_argument("--intrinsic_w", default=1, type=float)
    parser.add_argument("--eval", default=0, type=int)
    parser.add_argument("--cuda", default=0, type=int)
    parser.add_argument("--log_freq", default=500, type=int)
    parser.add_argument("--num_train_tasks", default=10, type=int)
    parser.add_argument("--eval_task_id", default=0, type=int)
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--task_id", default=4, type=int)
    parser.add_argument("--td3_preset", choices=["default","weaker","much_weaker"], default="weaker")


    args = parser.parse_args()
    args.save_model = True

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    file_name = f"{args.version}_seed{args.seed}"

    os.makedirs("./results", exist_ok=True)
    if args.save_model:
        os.makedirs("./models", exist_ok=True)

    env = make_env(args)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Revise/intrinsic hooks
    if args.eval == 1:
        args.revise_path = f'LESR-resources/run-v{args.version}-{args.env}/best_result/v{args.version}-best-{args.env}'
        print('#'*20); print('Evaluate Stage:', args.revise_path); print('#'*20)

    src_state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    revise_state = import_and_reload_module(args.revise_path).revise_state
    intrinsic_reward = import_and_reload_module(args.revise_path).intrinsic_reward

    test_state = np.zeros([src_state_dim, ])
    global state_dim
    state_dim = revise_state(test_state).shape[0]

    print("-----------------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, State Dim: {state_dim}")
    print("-----------------------------------------------")

    # Initialize policy
    if args.policy == "SAC":
        policy = SAC_new.SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            gamma=args.discount,
            tau=args.tau,  # SAC_new 使用 tau 而不是 target_update_coef
            seed=args.seed,
            # 减少训练波动的改进参数

            # hidden=256,              # 网络容量
            # use_layer_norm=True,     # 使用LayerNorm稳定训练
            # actor_lr=5e-5,           # 进一步降低actor学习率（从1e-4降到5e-5）防止过拟合
            # critic_lr=1e-4,          # 降低critic学习率（从3e-4降到1e-4）更稳定
            # alpha_lr=1e-4,           # 降低alpha学习率，防止温度系数变化过快
            # init_temperature=0.05,   # 降低初始温度（从0.1降到0.05）减少探索噪声
            # learnable_temperature=True,  # 自适应温度
            # grad_clip=0.3,           # 更严格的梯度裁剪（从0.5降到0.3）
            # actor_update_freq=2,     # 延迟actor更新（每2步更新一次）
            # target_update_freq=1


            hidden=256,              # 增大网络容量
            use_layer_norm=True,     # 使用LayerNorm稳定训练
            actor_lr=1e-4,           # 降低actor学习率（从3e-4降到1e-4）
            critic_lr=3e-4,          # critic学习率保持不变
            alpha_lr=3e-4,           # alpha学习率保持不变
            init_temperature=0.1,    # 降低初始温度（从0.2降到0.1，减少探索）
            learnable_temperature=True,  # 自适应温度
            grad_clip=0.5,           # 梯度裁剪（从1.0降到0.5，更保守）
            actor_update_freq=2,     # 延迟actor更新（每2步更新一次）
            target_update_freq=1     # 每步更新target网络   # 每步更新target网络
        )
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            gamma=args.discount,
            target_update_coef=args.tau,
            seed=args.seed
        )
    elif args.policy == "TD3":
        policy = TD3.TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            gamma=args.discount,
            target_update_coef=args.tau,
            seed=args.seed,
            preset=getattr(args, "td3_preset", "weaker")
        )


    else:
        raise ValueError(f"Unsupported policy: {args.policy}")

    if args.load_model != "":
        print("Loading existing model...")
        if args.load_model == "default":
            # 检测是否是LLM增强的训练，以确定加载路径
            _, llm_version = detect_llm_enhancement(args.revise_path)
            # 使用新的命名规则
            model_path = f"./models/{args.policy}_task{args.task_id}_seed{args.seed}{llm_version}/{args.policy}_task{args.task_id}"
        else:
            # 用户指定的路径
            model_path = args.load_model
        policy.load_models(model_path)
        print(f"[OK] Model loaded from: {model_path}")

    # Replay buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # ext_rnorm = RewardNormalizer(ema=0.99, eps=1e-8)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args)]
    evaluations_steps = [[evaluations[-1], 0]]
    intrinsic_ratio = []

    soft_state_correlation = np.zeros([state_dim, ])

    state, _ = env.reset()
    episode_reward, episode_intrinsic_reward = 0, 0
    episode_timesteps = 0
    episode_num = 0
    episode_all_state, episode_all_reward = [], []

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        if t < args.start_timesteps:
            action = env.action_space.sample()
            unscaled_action = action / max_action
        else:
            revised_state = revise_state(state)
            action_tensor = policy.explore(revised_state.reshape(1, -1))
            unscaled_action = action_tensor.detach().cpu().numpy().flatten()
            action = unscaled_action * max_action

        # DSOEnv uses continuous action space, clip to valid range
        action = action.clip(-max_action, max_action)

        next_state, reward, terminated, truncated, info = env.step(action)

        # Store state for Lipschitz constant calculation
        episode_all_state.append(revise_state(state).reshape(-1, 1))
        episode_all_reward.append(reward)

        done = terminated
        done_bool = float(done)

        
        # store UN-SCALED action in buffer (range [-1,1])

        # env_r_norm = ext_rnorm.normalize(reward)
        intrinsic_raw = intrinsic_reward(revise_state(np.array(state).flatten()))
        intrinsic_r = args.intrinsic_w * intrinsic_raw
        # total_r = env_r_norm + intrinsic_r

        replay_buffer.add(
            revise_state(state).reshape(1, -1),
            unscaled_action,
            revise_state(next_state).reshape(1, -1),
            reward + intrinsic_r,
            done_bool
        )

        state = next_state
        episode_reward += reward
        episode_intrinsic_reward += intrinsic_r

        # Train when enough samples collected
        if t >= args.start_timesteps:
            state_b, action_b, next_state_b, reward_b, not_done_b = replay_buffer.sample(args.batch_size)

            states = torch.as_tensor(state_b, dtype=torch.float32, device=device)
            actions = torch.as_tensor(action_b, dtype=torch.float32, device=device)
            next_states = torch.as_tensor(next_state_b, dtype=torch.float32, device=device)
            rewards = torch.as_tensor(reward_b, dtype=torch.float32, device=device)
            dones = 1. - torch.as_tensor(not_done_b, dtype=torch.float32, device=device)

            batch_tuple = (states, actions, rewards, next_states, dones)
            policy.update_online_networks(batch_tuple)
            policy.update_target_networks()

        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, _ = env.reset()
            done = False

            state_change = np.concatenate(episode_all_state, axis=1)
            reward_change = np.array(episode_all_reward).reshape(1, -1)

            state_lipschitz_constant_correlation = cal_lipschitz(state_change, reward_change)
            assert state_lipschitz_constant_correlation.shape == soft_state_correlation.shape, state_lipschitz_constant_correlation.shape

            state_lipschitz_constant_correlation[np.isnan(state_lipschitz_constant_correlation)] = 0
            soft_state_correlation = args.corr_tau * state_lipschitz_constant_correlation + (1 - args.corr_tau) * soft_state_correlation

            intrinsic_ratio.append(abs(episode_intrinsic_reward) / (abs(episode_reward) + 1e-5))

            episode_reward, episode_intrinsic_reward = 0, 0
            episode_timesteps = 0
            episode_num += 1
            episode_all_state, episode_all_reward = [], []

        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args))
            evaluations_steps.append([evaluations[-1], t])
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model:
                # 检测是否是LLM增强的训练
                is_llm_enhanced, llm_version = detect_llm_enhancement(args.revise_path)
                
                # 改进的模型保存路径：算法名_taskID_seed[_LLM_vXXX]
                model_save_path = f"./models_new/{args.policy}_task{args.task_id}_seed{args.seed}{llm_version}"
                os.makedirs(model_save_path, exist_ok=True)
                # 保存时使用统一的前缀
                model_prefix = f"{model_save_path}/{args.policy}_task{args.task_id}"
                policy.save_models(model_prefix)
                
                # 打印保存信息，标注LLM增强
                if is_llm_enhanced:
                    print(f"[OK] Model saved (LLM Enhanced): {args.policy} | Task {args.task_id} | Seed {args.seed} | Version: {llm_version} -> {model_save_path}")
                else:
                    print(f"[OK] Model saved (Baseline): {args.policy} | Task {args.task_id} | Seed {args.seed} -> {model_save_path}")

    evaluations_steps = np.array(evaluations_steps)
    if args.sid_result_path:
        np.save(args.sid_result_path, evaluations_steps)
    if args.eval == 0 and args.corr_result_path:
        np.save(args.corr_result_path, soft_state_correlation)
