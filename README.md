# Multi-Agent Sheep Herding with Reinforcement Learning

A custom multi-agent herding environment where **5 RL-controlled dogs** must gather **25 sheep** and drive them into a target zone — built from scratch on Gymnasium and trained with PPO (Stable-Baselines3).

Two tasks of increasing difficulty:

| Task 1 — sheep start clustered | Task 2 — sheep start scattered |
|:---:|:---:|
| ![original](assets/demo_original.gif) | ![scattered](assets/demo_scattered.gif) |
| PPO drives the cluster to the target | PPO **gathers** a scattered flock, then **drives** it home |

## The problem

Shepherding is a classic multi-agent control problem: dogs cannot move sheep directly — they can only *repel* them. To move the flock somewhere, a dog has to stand on the **opposite side** of the flock and push it by proximity, and gathering strays means physically circling around them. The policy controls all 5 dogs jointly from a single observation of the whole field, so it must learn spatial coordination (who covers which stray, how to form a driving arc) purely from reward.

- **World**: 1x1 continuous 2D field, target circle of radius 0.15 centered at (0.75, 0.75)
- **Success**: *every* sheep inside the target circle → episode ends with a large bonus
- **Failure**: time limit reached with any sheep outside

## Environment specification

### Observation space

A flat `Box` vector of positions, everything in `[0, 1]` field coordinates:

| Component | Original env | Scattered env |
|---|---|---|
| Sheep positions (x, y) x 25 | 50 dims | 50 dims, **sorted canonically** |
| Dog positions (x, y) x 5 | 10 dims | 10 dims, **sorted canonically** |
| Target position (x, y) | 2 dims | 2 dims |
| Gathered-phase flag | — | 1 dim (0/1, with hysteresis) |
| **Total** | **62 dims** | **63 dims** |

The scattered env sorts the (interchangeable) sheep and dog arrays every step so the MLP doesn't waste capacity learning permutation invariance, and exposes a gather/drive phase bit — 1 bit of memory a feedforward policy cannot infer on its own (details in the findings below).

### Action space

`Box(-10, 10, shape=(10,))` — a `(vx, vy)` velocity command per dog, integrated with `dt = 0.01` and clipped to the field. In the scattered env each action is **held for 10 physics substeps** (action repeat), so an episode is 250 decisions over 2,500 physics steps instead of 2,500 tiny decisions.

### Sheep dynamics (the part the dogs exploit)

Each sheep's velocity is a sum of simple steering terms, clipped to a max speed 20x slower than the dogs:

- **Flee dogs** — inverse-square repulsion `sum_j (sheep - dog_j) / |sheep - dog_j|^2`. In the original env this is **global** (any dog affects every sheep); in the scattered env it is **local** (only dogs within radius 0.3 count) — see finding #1 for why this change was necessary.
- **Cohesion** (scattered env only) — a weak pull toward the centroid of neighbours within radius 0.2, so sheep pushed together clump and *stay* a flock.

### Reward structure

**Original env** (dense shaping toward the fixed target):

| Term | Value |
|---|---|
| Dogs' progress toward target | `prev_dog_dist - dog_dist` |
| Sheep progress toward target | `0.5 x (prev_sheep_dist - sheep_dist)` |
| Flock spread penalty | `-0.5 x flock_radius` |
| Time penalty | `-0.01` per step |
| Per-sheep bonus inside target | `+2` per sheep per step |
| **All sheep inside target** | **+10,000, episode ends** |

**Scattered env** (two-phase: gather, then drive — the phase flag gates the weights):

| Term | Collecting phase | Gathered phase |
|---|---|---|
| Gather progress (flock radius shrinking) | `5.0 x` | `5.0 x` |
| Drive progress (flock moving to target) | `2.0 x` | `10.0 x` |
| Dogs' progress toward the **flock** | `1.0 x` | — |
| Dogs near the **driving point** (just beyond the flock, opposite the target — Strömbom-style) | — | `-2.0 x` distance to it |
| Absolute distance penalty | `-1.0 x` mean sheep-to-target distance (both phases) | |
| Flock spread + time | `-0.5 x radius - 0.01` (both phases) | |
| Per-sheep bonus inside target | `+0.2` per sheep per step (kept small — see finding #2) | |
| **All sheep inside target** | **+10,000, episode ends** | |

## Task 1 — Original clustered task

Sheep spawn together in a small patch near the dogs' corner; the target is diagonal across the field. Trained with PPO (`MlpPolicy`, lr 1e-3, `clip_range=0.1`, 10k steps).

![original training](assets/training_curves_original.png)

**Reading the curves**: mean episode reward sits at the +10,000 success bonus from the *first* rollout — this task succeeds almost by construction. With global repulsion and dogs spawning behind the flock, any dog movement pushes the cluster diagonally toward the target; PPO's actual learning shows up in the **episode length** falling from ~137 to ~117 steps as it refines the push. This near-freebie is precisely what motivated building the scattered variant: the interesting parts of shepherding (gathering, flock maintenance, driving position) never get exercised when the geometry does the work for you.

## Task 2 — Scattered task

Sheep spawn uniformly over the whole field (outside the target circle). Now the dogs must cross the field, collect strays into a flock, keep it together, and drive it home — none of which global-repulsion geometry gives for free.

### Training: curriculum PPO

Plain PPO on the full scattered task scores 0% no matter the reward. The winning recipe: sheep start in a small cluster at a **random** location (`scatter_scale = 0.1`) and the scatter widens by 0.05 whenever the recent success rate exceeds ~40%, until the full field. ~9M steps across three chained runs:

![scattered training](assets/training_curves.png)

(Success rate is measured at the *current* curriculum stage, which is why it saw-tooths downward as stages get harder — each drop is a promotion to a wider scatter.)

### Results (30-episode evaluations, stochastic policy)

| Policy | scatter 0.5 | scatter 0.9 | full scatter |
|---|---|---|---|
| **PPO, curriculum (~9M steps)** | **77%** | 30% | **27%** |
| Scripted shepherd (upper-bound baseline) | — | — | 84% |
| PPO without curriculum / action repeat | — | — | 0% |

The demo GIF above is a real full-scatter success episode: all 25 sheep gathered and delivered in 131 decisions. Evaluate with `deterministic=False` — the stochastic policy herds far better than its deterministic mean (action noise breaks positional deadlocks).

### How we got there (the interesting part)

Every naive approach scored **exactly 0%**. Each plateau exposed a real defect — the full chronological log is in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md):

1. **Physics was the first bug.** Global `1/dist` repulsion meant any dog pushed *every* sheep at max speed — random dog motion blasted the whole field into a corner. PPO happily learned to pile sheep in the corner and stall there (it was reward-neutral). Fix: local repulsion radius + an absolute distance penalty so parking far away bleeds reward.
2. **Reward exploits are real.** The original env's `+2`/sheep/step in-target bonus out-earned the one-time success reward — the optimal policy would hold one sheep out forever to avoid termination. Caught and cut to `+0.2` before the agent found it.
3. **A 90%-success scripted expert couldn't be imitated.** Behavior cloning underfit badly: the expert assigns dogs to strays combinatorially, and tiny state changes teleport goals between dogs — a function class an MLP cannot represent. Imitation was abandoned for pure RL, informed by why it failed.
4. **Decision density killed exploration.** 2,500 tiny decisions per episode meant PPO's noise never composed into maneuvers. Action repeat (10 substeps per decision) tripled success at every curriculum stage overnight.
5. **Curriculum gating needs hygiene.** The first curriculum over-promoted — episodes still in flight at the easier scale polluted the promotion stats (5 stages in 60k steps). A promotion cooldown and larger evidence window fixed it.

## Repo structure

```
notebooks/   Herding_env.ipynb (original task), Herding_scattered_env.ipynb (scattered task, full pipeline)
src/         scattered_env.py, scripted_expert.py, train.py  — importable versions of the env/baseline/trainer
models/      ppo_sheep_dog.zip (original), ppo_sheep_dog_scattered.zip (scattered, best checkpoint)
assets/      demo GIFs + training curves for both tasks
tensorboard/ original/ and scattered/ (all 15 experiment runs)
docs/        EXPERIMENT_LOG.md — chronological record of every failure and fix
```

## Run it

```bash
pip install -r requirements.txt
# train the scattered task from scratch (several hours on CPU)
python src/train.py --run-name my_run --timesteps 3000000 --curriculum 0.1 \
    --lr 0.0003 --n-steps 256 --batch-size 512 --gamma 0.99 --ent-coef 0.005
# or open either notebook and run cells top to bottom
tensorboard --logdir tensorboard
```

## What's next

The gap to the 84% scripted expert is the flat MLP on a combinatorial multi-agent problem. Ranked next steps: permutation-invariant policy (deep sets / attention over sheep), longer curriculum training, per-dog decentralized policies with shared weights, DAgger with an attention policy.
