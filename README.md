# Multi-Agent Sheep Herding with Reinforcement Learning

A custom multi-agent herding environment where **5 RL-controlled dogs** must gather **25 sheep** and drive them into a target zone — built from scratch on Gymnasium and trained with PPO (Stable-Baselines3).

Two tasks of increasing difficulty:

| Original task — sheep start clustered | Scattered task — sheep start anywhere |
|:---:|:---:|
| ![original](assets/demo_original.gif) | ![scattered](assets/demo_scattered.gif) |
| PPO drives the cluster to the target | PPO **gathers** a scattered flock, then **drives** it home |

## The environment

- 2D continuous field, 25 sheep + 5 dogs + a target circle
- Dogs are controlled by one policy outputting a velocity per dog (10-dim continuous action)
- Sheep are boids-like: they **flee nearby dogs** (inverse-square repulsion within a radius) and **cohere with neighbours** — so once pushed together they behave like a flock
- Success = every sheep inside the target circle
- The scattered variant adds: canonical (sorted) observations, a gather/drive phase flag with hysteresis, action repeat (10 physics substeps per decision), and a two-phase shaped reward (gather → drive) with a Strömbom-style driving-position term

## Results

![training curves](assets/training_curves.png)

30-episode evaluations of the shipped scattered-task model (stochastic policy):

| Policy | scatter 0.5 | scatter 0.9 | full scatter |
|---|---|---|---|
| **PPO, curriculum (~9M steps)** | **77%** | 30% | **27%** |
| Scripted shepherd (upper-bound baseline) | — | — | 84% |
| PPO without curriculum / action repeat | — | — | 0% |

The demo GIF above is a real full-scatter success episode: all 25 sheep gathered and delivered in 131 decisions.

## How we got there (the interesting part)

Every naive approach scored **exactly 0%**. Each plateau exposed a real defect that had to be found and fixed — the full log is in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md):

1. **Physics was the first bug.** Global `1/dist` repulsion meant any dog pushed *every* sheep at max speed — random dog motion blasted the whole field into a corner. PPO happily learned to pile sheep in the corner and stall there (it was reward-neutral). Fix: local repulsion radius + an absolute distance penalty so parking far away bleeds reward.
2. **Reward exploits are real.** The per-step "sheep in target" bonus out-earned the one-time success reward — the optimal policy would hold one sheep out forever to avoid termination. Caught and rebalanced before the agent found it.
3. **A 90%-success scripted expert couldn't be imitated.** Behavior cloning underfit badly: the expert assigns dogs to strays combinatorially, and tiny state changes teleport goals between dogs — a function class an MLP cannot represent. Imitation was abandoned for pure RL, informed by why it failed.
4. **Decision density killed exploration.** 2,500 tiny decisions per episode meant PPO's noise never composed into maneuvers. Action repeat (10 substeps per decision) tripled success at every curriculum stage overnight.
5. **Curriculum learning was the winning recipe** — sheep start in a small cluster at a random spot and the scatter widens whenever recent success exceeds 40%. The first version over-promoted (in-flight episodes polluted the promotion stats — 5 stages in 60k steps); a promotion cooldown fixed it.

## Repo structure

```
notebooks/   Herding_env.ipynb (original task), Herding_scattered_env.ipynb (scattered task, full pipeline)
src/         scattered_env.py, scripted_expert.py, train.py  — importable versions of the env/baseline/trainer
models/      ppo_sheep_dog.zip (original), ppo_sheep_dog_scattered.zip (scattered, best checkpoint)
assets/      demo GIFs + training curves
tensorboard/ original/ and scattered/ (all 15 experiment runs)
docs/        EXPERIMENT_LOG.md — chronological record of every failure and fix
```

## Run it

```bash
pip install -r requirements.txt
# train the scattered task from scratch (several hours on CPU)
python src/train.py --run-name my_run --timesteps 3000000 --curriculum 0.1 \
    --lr 0.0003 --n-steps 256 --batch-size 512 --gamma 0.99 --ent-coef 0.005
# or open notebooks/Herding_scattered_env.ipynb and run cells top to bottom
tensorboard --logdir tensorboard/scattered
```

Evaluate/demo with `deterministic=False` — the stochastic policy herds far better than its deterministic mean (action noise breaks positional deadlocks).

## What's next

The gap to the 84% scripted expert is the flat MLP on a combinatorial multi-agent problem. Ranked next steps: permutation-invariant policy (deep sets / attention over sheep), longer curriculum training, per-dog decentralized policies with shared weights, DAgger with an attention policy.
