"""Scripted Strombom-style shepherd to verify the v4 env is solvable."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scattered_env import ScatteredSheepDogEnv


def _go(dog, goal, center, flock_rmax, max_speed):
    """Velocity toward goal, detouring AROUND the flock instead of through it."""
    v = goal - dog
    dist_goal = np.linalg.norm(v)
    # does the straight path pass close to the flock center?
    to_c = center - dog
    seg = np.linalg.norm(v) + 1e-8
    t = np.clip(np.dot(to_c, v) / seg**2, 0.0, 1.0)
    closest = dog + t * v
    clearance = flock_rmax + 0.06
    if np.linalg.norm(closest - center) < clearance and np.linalg.norm(to_c) < seg:
        # detour: steer tangentially around the flock
        away = dog - center
        away_n = np.linalg.norm(away) + 1e-8
        tang = np.array([-away[1], away[0]]) / away_n
        if np.dot(tang, v) < 0:
            tang = -tang
        v = tang + 0.5 * (away / away_n) * max(0.0, (clearance - away_n))
        dist_goal = 1.0
    out = v / (np.linalg.norm(v) + 1e-8) * max_speed
    if dist_goal < 0.05:  # damp near the goal so the dog holds position
        out *= dist_goal / 0.05
    return out


def heuristic_action(env, state=None):
    sheep, dogs = env.sheep_positions, env.dog_positions
    center = sheep.mean(axis=0)
    d_center = np.linalg.norm(sheep - center, axis=1)
    rmax = d_center.max()
    # the env tracks the gather/drive phase (with hysteresis) and exposes it in
    # the observation, so the expert is Markovian w.r.t. what the policy sees
    gathered = env.gathered
    # build one goal per dog, then match goals to dogs geometrically (nearest
    # free dog) so the plan is stable even though array order changes per step
    goals = []
    if not gathered:
        # collect the k farthest strays: stand behind each stray (opposite side
        # from flock center) and push it inward
        order = np.argsort(-d_center)
        for k in range(env.num_dogs):
            stray = sheep[order[k % len(order)]]
            away = stray - center
            n = np.linalg.norm(away) + 1e-8
            goals.append(np.clip(stray + (away / n) * 0.12, 0.0, env.world_size))
    else:
        # drive: an arc of positions behind the flock relative to the target
        away = center - env.target_position
        away = away / (np.linalg.norm(away) + 1e-8)
        base_ang = np.arctan2(away[1], away[0])
        for k in range(env.num_dogs):
            ang = base_ang + (k - (env.num_dogs - 1) / 2) * 0.4
            goal = center + (rmax + 0.10) * np.array([np.cos(ang), np.sin(ang)])
            goals.append(np.clip(goal, 0.0, env.world_size))

    action = np.zeros((env.num_dogs, 2))
    free_dogs = list(range(env.num_dogs))
    free_goals = list(range(len(goals)))
    while free_goals:
        pairs = [(np.linalg.norm(dogs[i] - goals[j]), i, j) for i in free_dogs for j in free_goals]
        _, i, j = min(pairs)
        action[i] = _go(dogs[i], goals[j], center, rmax, env.max_speed_dog)
        free_dogs.remove(i)
        free_goals.remove(j)
    return action.flatten()


if __name__ == "__main__":
    successes, lens = [], []
    for seed in range(10):
        env = ScatteredSheepDogEnv(render_mode="rgb_array")
        obs, _ = env.reset(seed=seed)
        done = trunc = False
        steps = 0
        while not (done or trunc):
            obs, r, done, trunc, _ = env.step(heuristic_action(env))
            steps += 1
        d = np.linalg.norm(env.sheep_positions - env.target_position, axis=1)
        ok = bool(np.all(d < env.target_distance_threshold))
        successes.append(ok)
        lens.append(steps)
        print(f"seed {seed}: success={ok} steps={steps} mean_dist={d.mean():.3f} "
              f"in={int((d < env.target_distance_threshold).sum())}/25 radius={env._flock_radius():.3f}")
    print(f"\nheuristic success rate: {np.mean(successes):.0%}, mean steps: {np.mean(lens):.0f}")


# ---------------- Smooth expert (MLP-friendly) ----------------
def _steer(p, goal, center, clearance, vmax):
    """Head toward goal with a smooth blend that routes around the flock."""
    v = goal - p
    dv = np.linalg.norm(v)
    dirn = v / (dv + 1e-8)
    away = p - center
    da = np.linalg.norm(away) + 1e-8
    if da < clearance and dv > 0.05:
        w = (clearance - da) / clearance
        tang = np.array([-away[1], away[0]]) / da
        if np.dot(tang, dirn) < 0:
            tang = -tang
        dirn = (1 - w) * dirn + w * (0.7 * tang + 0.3 * away / da)
        dirn = dirn / (np.linalg.norm(dirn) + 1e-8)
    out = dirn * vmax
    if dv < 0.05:
        out *= dv / 0.05
    return out


def smooth_expert(env):
    """
    Per-dog action is a SMOOTH function of the state (soft attention over strays,
    lateral projection for drive slots) so an MLP policy can imitate it.
    """
    sheep, dogs = env.sheep_positions, env.dog_positions
    c = sheep.mean(axis=0)
    t = env.target_position
    d_center = np.linalg.norm(sheep - c, axis=1)
    rmax = d_center.max()
    clearance = rmax + 0.06
    actions = np.zeros((env.num_dogs, 2))
    if env.gathered:
        u_away = (c - t) / (np.linalg.norm(c - t) + 1e-8)
        D = c + (rmax + 0.10) * u_away
        perp = np.array([-u_away[1], u_away[0]])
        for i in range(env.num_dogs):
            lat = np.clip(np.dot(dogs[i] - c, perp), -0.2, 0.2)
            goal = np.clip(D + lat * perp, 0.0, env.world_size)
            actions[i] = _steer(dogs[i], goal, c, clearance, env.max_speed_dog)
    else:
        # soft attention: far-from-center sheep matter more; each dog favors
        # strays it is already close to, so dogs split up naturally
        w_stray = np.exp(d_center / 0.06)
        behind = sheep + 0.12 * (sheep - c) / (d_center[:, None] + 1e-8)
        for i in range(env.num_dogs):
            d_ps = np.linalg.norm(behind - dogs[i], axis=1)
            w = w_stray * np.exp(-d_ps / 0.25)
            w = w / w.sum()
            goal = np.clip((w[:, None] * behind).sum(axis=0), 0.0, env.world_size)
            actions[i] = _steer(dogs[i], goal, c, clearance, env.max_speed_dog)
    return actions.flatten()


def pack_expert(env):
    """
    All dogs work together on ONE goal (behind the farthest stray while
    collecting, behind the flock while driving), spreading out sideways via a
    smooth lateral projection. One discrete choice (argmax sheep), no
    combinatorial assignment — learnable by an MLP.
    """
    sheep, dogs = env.sheep_positions, env.dog_positions
    c = sheep.mean(axis=0)
    d_center = np.linalg.norm(sheep - c, axis=1)
    rmax = d_center.max()
    clearance = rmax + 0.06
    if env.gathered:
        anchor = c
        u_away = (c - env.target_position)
        push_r = rmax + 0.10
    else:
        stray = sheep[np.argmax(d_center)]
        anchor = stray
        u_away = (stray - c)
        push_r = 0.12
    u_away = u_away / (np.linalg.norm(u_away) + 1e-8)
    base = anchor + push_r * u_away
    perp = np.array([-u_away[1], u_away[0]])
    actions = np.zeros((env.num_dogs, 2))
    for i in range(env.num_dogs):
        lat = np.clip(np.dot(dogs[i] - anchor, perp), -0.15, 0.15)
        goal = np.clip(base + lat * perp, 0.0, env.world_size)
        actions[i] = _steer(dogs[i], goal, c, clearance, env.max_speed_dog)
    return actions.flatten()
