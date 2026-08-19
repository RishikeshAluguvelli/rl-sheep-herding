"""PPO training harness for ScatteredSheepDogEnv with eval + best-checkpoint saving."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scattered_env import ScatteredSheepDogEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


class CurriculumCallback(BaseCallback):
    """Widen the sheep scatter as the policy's success rate climbs."""
    def __init__(self, start=0.15, step=0.05, threshold=0.35, min_episodes=60,
                 cooldown_steps=30_000):
        super().__init__()
        self.scale = start
        self.step_size = step
        self.threshold = threshold
        self.min_episodes = min_episodes
        # Cooldown after a promotion: episodes already in flight at the easier
        # scale still land in the stats buffer, so promoting again immediately
        # cascades on stale evidence.
        self.cooldown_steps = cooldown_steps
        self.last_promotion_step = 0

    def _on_training_start(self):
        self.training_env.env_method("set_scatter_scale", self.scale)

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if (self.scale >= 1.0 or buf is None or len(buf) < self.min_episodes
                or self.num_timesteps - self.last_promotion_step < self.cooldown_steps):
            return
        succ = np.mean([e.get("is_success", False) for e in buf])
        if succ > self.threshold:
            self.scale = min(1.0, self.scale + self.step_size)
            self.training_env.env_method("set_scatter_scale", self.scale)
            buf.clear()
            self.last_promotion_step = self.num_timesteps
            print(f"[curriculum] success {succ:.2f} -> scatter_scale {self.scale:.2f} "
                  f"at {self.num_timesteps} steps", flush=True)

    def _on_step(self):
        return True

REPO = os.getcwd()  # tensorboard logs land in ./Sheepherding_scattered_tensorboard/


def make_env(seed):
    def _init():
        env = ScatteredSheepDogEnv(render_mode="rgb_array")
        env.reset(seed=seed)
        return Monitor(env, info_keywords=("is_success",))
    return _init


def build_vec(n_envs, seed0, normalize, training, stats_path=None):
    venv = DummyVecEnv([make_env(seed0 + i) for i in range(n_envs)])
    if normalize:
        if stats_path and os.path.exists(stats_path):
            venv = VecNormalize.load(stats_path, venv)
            venv.training = training
        else:
            venv = VecNormalize(venv, norm_obs=True, norm_reward=training, clip_reward=50.0)
            venv.training = training
        if not training:
            venv.norm_reward = False
    return venv


def evaluate(model, venv_eval, n_episodes, normalize):
    """Deterministic eval. Returns dict of metrics. venv_eval must be a 1-env vec."""
    successes, final_dists, final_radii, ep_lens, ep_rewards = [], [], [], [], []
    inner = venv_eval.venv if normalize else venv_eval
    raw_env = inner.envs[0].unwrapped
    ns, tgt, thr = raw_env.num_sheep, raw_env.target_position, raw_env.target_distance_threshold
    for ep in range(n_episodes):
        obs = venv_eval.reset()
        done = False
        total_r, steps = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = venv_eval.step(action)
            done = bool(dones[0])
            steps += 1
            total_r += float(r[0])
        # after auto-reset the raw env state is the NEW episode; use terminal_observation
        tobs = np.asarray(infos[0]["terminal_observation"], dtype=np.float64)
        if normalize:
            tobs = venv_eval.unnormalize_obs(tobs)
        sheep = tobs[: 2 * ns].reshape(ns, 2)
        d = np.linalg.norm(sheep - tgt, axis=1)
        success = bool(np.all(d < thr))
        center = sheep.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(sheep - center, axis=1)))
        successes.append(success)
        final_dists.append(float(np.mean(d)))
        final_radii.append(radius)
        ep_lens.append(steps)
        ep_rewards.append(total_r)
    return {
        "success_rate": float(np.mean(successes)),
        "mean_final_dist": float(np.mean(final_dists)),
        "mean_final_flock_radius": float(np.mean(final_radii)),
        "mean_ep_len": float(np.mean(ep_lens)),
        "mean_ep_reward": float(np.mean(ep_rewards)),
        "n_episodes": n_episodes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--net", type=str, default="256,256")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--curriculum", type=float, default=None,
                   help="starting scatter_scale; enables the curriculum callback")
    p.add_argument("--init-model", type=str, default=None,
                   help="path to a saved PPO model to continue training from")
    args = p.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    tb_dir = os.path.join(REPO, "Sheepherding_scattered_tensorboard")

    venv = build_vec(args.n_envs, args.seed, args.normalize, training=True)
    eval_env = build_vec(1, args.seed + 1000, args.normalize, training=False)

    net = [int(x) for x in args.net.split(",")]
    if args.init_model:
        model = PPO.load(args.init_model, env=venv,
                         learning_rate=args.lr, clip_range=args.clip,
                         ent_coef=args.ent_coef, tensorboard_log=tb_dir)
    else:
        model = PPO(
        "MlpPolicy", venv, verbose=1,
        tensorboard_log=tb_dir,
        learning_rate=args.lr, clip_range=args.clip,
        n_steps=args.n_steps, batch_size=args.batch_size,
        gamma=args.gamma, ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=net),
        seed=args.seed,
    )

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=out_dir,
        log_path=out_dir, eval_freq=max(20_000 // args.n_envs, 1),
        n_eval_episodes=5, deterministic=True, render=False,
    )
    callbacks = [eval_cb]
    if args.curriculum is not None:
        callbacks.append(CurriculumCallback(start=args.curriculum))

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name=args.run_name)
    train_time = time.time() - t0

    model.save(os.path.join(out_dir, "final_model"))
    if args.normalize:
        venv.save(os.path.join(out_dir, "vecnormalize.pkl"))

    # Final eval: compare final model vs best-by-callback, keep whichever is better
    stats_path = os.path.join(out_dir, "vecnormalize.pkl") if args.normalize else None
    results = {}
    for tag, path in [("final", os.path.join(out_dir, "final_model.zip")),
                      ("best_cb", os.path.join(out_dir, "best_model.zip"))]:
        if not os.path.exists(path):
            continue
        m = PPO.load(path)
        ev = build_vec(1, 9000, args.normalize, training=False, stats_path=stats_path)
        results[tag] = evaluate(m, ev, args.eval_episodes, args.normalize)
        ev.close()

    summary = {
        "run_name": args.run_name,
        "config": vars(args),
        "train_time_sec": round(train_time, 1),
        "results": results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
