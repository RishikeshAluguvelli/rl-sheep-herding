# Scattered Sheep-Herding: Experiment Log & Results

**Task**: variation of the sheep-herding problem (`Herding_env.ipynb`) where the 25 sheep start
**scattered across the whole field** instead of clustered in a corner. The 5 RL-controlled dogs
must first gather the flock, then drive it to the target circle (radius 0.15 at (0.75, 0.75)).
Success = every sheep inside the target circle.

**Deliverables**
- `Herding_scattered_env.ipynb` — env (`ScatteredSheepDogEnv`), scripted baseline, curriculum PPO training, test cell
- `ppo_sheep_dog_scattered.zip` — best trained PPO model
- `Sheep-Herding-Scattered.gif` — a full success episode by the trained model (gather -> drive -> all 25 in target, 131 steps)
- `./Sheepherding_scattered_tensorboard/` — training curves for all runs

## Final results (30-episode evaluations, stochastic policy)

| Policy | scatter 0.5 | scatter 0.9 | full scatter (1.0) |
|---|---|---|---|
| **PPO, curriculum (~9M steps total)** | **77%** | 30% | **27%** |
| Scripted shepherd (upper-bound baseline) | — | — | 84% (mean 116 steps) |
| PPO, no curriculum / no action repeat | — | — | 0% |

Notes:
- `scatter s` = sheep spawn in an s-sized square at a random field location; 1.0 = full field.
- The stochastic policy far outperforms its deterministic mean (~0%): action noise breaks
  positional deadlocks. Demo/eval with `deterministic=False`.
- The scripted Strombom-style shepherd (collect farthest strays from behind -> drive from a rear
  arc, routing around the flock) is included in the notebook as a baseline and solvability proof.

## What it took to make the task learnable (chronological findings)

Every failure below scored 0% success at full scatter until the fix landed.

1. **Naive port of the original env fails.** With sheep scattered, the original global `1/dist`
   dog repulsion means any dog pushes every sheep at max speed from anywhere; random dog motion
   blasts all sheep into the far corner. PPO learns to pile sheep in the corner (perfectly
   gathered, 0.36 from target) and stalls — the corner was reward-neutral.
2. **Reward fixes alone were not enough.** Added an absolute per-step distance penalty (parking a
   gathered flock far away now bleeds reward) and a driving-position term (dogs rewarded for
   standing on the far side of the flock, Strombom-style). Also fixed a perverse incentive: the
   per-sheep in-target bonus (2/step) out-earned the one-time success reward (10000), teaching the
   agent to hold one sheep out; cut to 0.2/step.
3. **Local repulsion** (`repulsion_radius = 0.3`): sheep only flee nearby dogs. This is what makes
   controlled herding physically possible with a scattered start. A scripted shepherd then solves
   the task 84-90% of the time, proving solvability (episodes need ~1100-2000 physics steps, so
   `max_steps` raised accordingly).
4. **Imitation learning failed — instructively.** Behavior-cloning the 90% scripted expert into
   the MLP policy underfit badly (MSE plateau, 0% rollouts): the expert's stray-to-dog assignment
   is combinatorial (tiny state changes teleport goals between dogs), which an MLP cannot
   represent. Smoothed single-goal experts were learnable in principle but too weak (20%).
5. **Observation fixes**: sheep/dog arrays kept in canonical (sorted) order — 25 interchangeable
   sheep in arbitrary slots is permutation noise an MLP has to waste capacity ignoring — and the
   gather/drive phase flag (with hysteresis) exposed in the observation, since that 1 bit of
   memory is otherwise invisible to a feedforward policy.
6. **Action repeat** (`decision_repeat = 10`): each policy action is held for 10 physics substeps,
   turning 2500-decision episodes into 250. Before this, PPO's per-step exploration noise never
   composed into herding maneuvers (8% success even on the easiest curriculum stage after 1.8M
   steps; 30-40% after).
7. **Curriculum learning** (the winning recipe): sheep start in a 0.1-scale cluster at a random
   location; scatter widens by 0.05 whenever the recent success rate exceeds ~40%. First attempt
   over-promoted (episodes in flight at the old scale polluted the promotion stats — 5 stages in
   60k steps); fixed with a 30k-step promotion cooldown and a larger evidence window.

## Training runs (PPO, SB3 2.5.0, 8 parallel envs, MlpPolicy 256x256)

| Run | Steps | Setup | Full-scatter success |
|---|---|---|---|
| A-C | 300k each | original-style reward, global repulsion | 0% (corner exploit) |
| D-E | 400k each | + distance penalty | 0% (corner exploit) |
| F-G | 400k each | + drive-point shaping | 0% |
| H-I | 600k each | + local repulsion (v4 dynamics) | 0% (can't even gather) |
| J | BC + 400k | behavior cloning + fine-tune | 0% (BC underfits) |
| K | 2M | curriculum, no action repeat | 8% at easiest stage only |
| L | 600k | + action repeat (v5) | 30-36% at stage 0.2 |
| M | 3M | curriculum 0.1 -> 0.9 (over-promoted) | ~10% stochastic |
| N | 3M | continue M, fixed curriculum gating | 43% @ 0.9, 10% @ 1.0 |
| **O** | 3M | continue N from 0.9 -> 1.0 | **27% @ 1.0 (shipped model)** |

Hyperparameters for the shipped model: lr 3e-4 -> 1.5e-4 across continuations, clip 0.2,
n_steps 256, batch 512, gamma 0.99, ent_coef 0.005 -> 0.003, net [256, 256].

## Honest assessment & next steps

The curriculum PPO model genuinely gathers and drives (see the GIF), solving over a quarter of
full-scatter episodes and three quarters of half-scatter ones, but remains well below the scripted
expert (84%). The bottleneck is the flat MLP over 63 raw dimensions for a combinatorial
multi-agent coordination problem. Likely next wins, in order: (1) permutation-invariant policy
(deep sets / attention over sheep), (2) 3-5x more curriculum training, (3) per-dog decentralized
policies with shared weights, (4) DAgger from the scripted expert with an attention policy.
