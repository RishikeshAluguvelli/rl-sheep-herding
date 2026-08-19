import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


class ScatteredSheepDogEnv(gym.Env):
    """
    Variation of SheepDogEnv where the sheep start SCATTERED across the whole
    field instead of together in one corner. The dogs must first gather the
    flock and then drive it to the target position.
    """
    def __init__(self, num_sheep=25, num_dogs=5, max_speed_dog=10, max_speed_sheep=0.5, dt=0.01,
                 world_size=1.0, target_position=np.array([0.75, 0.75]), render_mode="human",
                 scatter_margin=0.05, gather_radius=0.08, cohesion_radius=0.2, cohesion_gain=0.3,
                 repulsion_radius=0.3, decision_repeat=10):
        super().__init__()
        self.num_sheep = num_sheep
        self.num_dogs = num_dogs
        self.dt = dt
        self.world_size = world_size
        self.target_position = target_position
        self.current_step = 0
        self.max_speed_dog = max_speed_dog
        self.max_speed_sheep = max_speed_sheep
        # Each policy decision is held for decision_repeat physics substeps.
        # Without this, episodes are thousands of tiny decisions and PPO's
        # exploration never composes into herding maneuvers.
        self.decision_repeat = decision_repeat
        # Scattered start needs gather + drive, which takes longer than the
        # clustered original (sheep move at most max_speed_sheep*dt per substep)
        self.max_steps = 250
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.scat_sheep = None
        self.scat_dogs = None
        self.target_plot = None
        self.target_distance_threshold = 0.15

        # Scatter / gathering parameters
        self.scatter_margin = scatter_margin      # keep spawns away from the walls
        self.gather_radius = gather_radius        # flock counts as "gathered" below this radius
        self.cohesion_radius = cohesion_radius    # sheep are attracted to neighbours within this range
        self.cohesion_gain = cohesion_gain        # strength of that attraction
        self.repulsion_radius = repulsion_radius  # sheep only flee dogs within this range
        # Curriculum knob: 1.0 = sheep scattered over the whole field (the real
        # task); smaller values spawn them in a proportionally smaller square at
        # a random location, which is an easier version of the same task
        self.scatter_scale = 1.0

        # Action space: (vx, vy) per dog
        self.action_space = spaces.Box(low=-self.max_speed_dog, high=self.max_speed_dog,
                                       shape=(2 * self.num_dogs,), dtype=np.float32)

        # Observation: all sheep (x,y) + all dog (x,y) + target (x,y) + gathered flag
        obs_dim = 2 * self.num_sheep + 2 * self.num_dogs + 2 + 1
        obs_low = np.full(obs_dim, 0.0, dtype=np.float32)
        obs_high = np.full(obs_dim, max(world_size, 1.0), dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Sheep scattered over a scatter_scale-sized square at a random location
        # (the whole field when scatter_scale=1.0), outside the target circle so
        # the episode never starts partially solved
        lo = self.scatter_margin
        hi = self.world_size - self.scatter_margin
        side = self.scatter_scale * (hi - lo)
        while True:
            ox = np.random.uniform(lo, hi - side)
            oy = np.random.uniform(lo, hi - side)
            # the square must not sit entirely inside the target circle, or the
            # out-of-circle resampling below could never terminate
            corners = np.array([[ox, oy], [ox + side, oy], [ox, oy + side], [ox + side, oy + side]])
            if np.any(np.linalg.norm(corners - self.target_position, axis=1) >= self.target_distance_threshold):
                break
        self.sheep_positions = np.stack([np.random.uniform(ox, ox + side, self.num_sheep),
                                         np.random.uniform(oy, oy + side, self.num_sheep)], axis=1)
        for i in range(self.num_sheep):
            while np.linalg.norm(self.sheep_positions[i] - self.target_position) < self.target_distance_threshold:
                self.sheep_positions[i] = np.array([np.random.uniform(ox, ox + side),
                                                    np.random.uniform(oy, oy + side)])

        # Dogs still start together in a corner
        self.dog_positions = np.random.uniform(0.0, 0.1, size=(self.num_dogs, 2))

        self._canonicalize()
        self.current_step = 0
        self.gathered = self._flock_radius() < self.gather_radius
        self.prev_mean_dist = self._mean_dist_to_target()
        self.prev_flock_radius = self._flock_radius()
        self.prev_mean_dog_dist = self._mean_dog_dist_to_flock()
        return self._get_observation(), {}

    def _canonicalize(self):
        """
        Sheep and dogs are interchangeable agents, so keep both arrays in a
        canonical (lexicographic) order. This removes permutation noise from the
        observation, which an MLP policy otherwise has to learn to ignore.
        Observation slot k and action slot k stay consistent within a step.
        """
        self.sheep_positions = self.sheep_positions[
            np.lexsort((self.sheep_positions[:, 1], self.sheep_positions[:, 0]))]
        self.dog_positions = self.dog_positions[
            np.lexsort((self.dog_positions[:, 1], self.dog_positions[:, 0]))]

    def _update_gathered(self):
        # Hysteresis: flip to "gathered" below gather_radius, back to "collecting"
        # only above 1.5x, so the phase doesn't flip-flop right at the threshold.
        r = self._flock_radius()
        if r < self.gather_radius:
            self.gathered = True
        elif r > 1.5 * self.gather_radius:
            self.gathered = False

    def _get_observation(self):
        return np.concatenate([self.sheep_positions.flatten(),
                               self.dog_positions.flatten(),
                               self.target_position.flatten(),
                               [float(self.gathered)]]).astype(np.float32)

    def _mean_dist_to_target(self):
        return np.mean(np.linalg.norm(self.sheep_positions - self.target_position, axis=1))

    def _flock_radius(self):
        center = self.sheep_positions.mean(axis=0)
        return np.mean(np.linalg.norm(self.sheep_positions - center, axis=1))

    def _mean_dog_dist_to_flock(self):
        center = self.sheep_positions.mean(axis=0)
        return np.mean(np.linalg.norm(self.dog_positions - center, axis=1))

    def step(self, action):
        action = np.clip(action, -self.max_speed_dog, self.max_speed_dog)
        dog_velocities = action.reshape(self.num_dogs, 2)
        self.current_step += 1
        for _ in range(self.decision_repeat):
            self._substep(dog_velocities)
            if self._all_in_target():
                break
        self._canonicalize()
        self._update_gathered()
        reward = self._compute_reward()
        done, truncated = self._check_done()
        info = {"is_success": done} if (done or truncated) else {}
        return self._get_observation(), reward, done, truncated, info

    def _all_in_target(self):
        d = np.linalg.norm(self.sheep_positions - self.target_position, axis=1)
        return bool(np.all(d < self.target_distance_threshold))

    def _substep(self, dog_velocities):
        self.dog_positions += dog_velocities * self.dt
        self.dog_positions = np.clip(self.dog_positions, 0.0, self.world_size)

        # ----- Sheep dynamics -----
        # 1) Repulsion from every dog (same rule as the original env, vectorised):
        #    v_i += sum_j (sheep_i - dog_j) / |sheep_i - dog_j|^2
        diff = self.sheep_positions[:, None, :] - self.dog_positions[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2, keepdims=True) + 1e-6
        # LOCAL repulsion: sheep only react to dogs within repulsion_radius.
        # With the original global 1/dist repulsion, any dog anywhere pushes every
        # sheep at max speed, so scattered sheep just get blasted into the walls
        # and controlled herding is impossible.
        in_range = dist_sq < self.repulsion_radius ** 2
        sheep_velocities = np.sum((diff / dist_sq) * in_range, axis=1)

        # 2) Weak cohesion: each sheep drifts toward the centroid of its
        #    neighbours within cohesion_radius, so pushed-together sheep clump
        #    and stay a flock once gathered
        pair_diff = self.sheep_positions[None, :, :] - self.sheep_positions[:, None, :]
        pair_dist = np.linalg.norm(pair_diff, axis=2)
        neighbor_mask = (pair_dist < self.cohesion_radius) & (pair_dist > 0)
        counts = neighbor_mask.sum(axis=1, keepdims=True)
        cohesion = (pair_diff * neighbor_mask[:, :, None]).sum(axis=1) / np.maximum(counts, 1)
        sheep_velocities += self.cohesion_gain * cohesion

        sheep_velocities = np.clip(sheep_velocities, -self.max_speed_sheep, self.max_speed_sheep)
        self.sheep_positions += sheep_velocities * self.dt
        self.sheep_positions = np.clip(self.sheep_positions, 0.0, self.world_size)

    def set_scatter_scale(self, scale):
        self.scatter_scale = float(np.clip(scale, 0.05, 1.0))

    def _compute_reward(self):
        """
        Two-phase shaping:
          - GATHER: shrinking the flock radius is rewarded strongly while the
            sheep are still spread out.
          - DRIVE: once the flock is gathered (radius < gather_radius), moving
            the flock toward the target dominates the reward.
        Dogs also get credit for approaching the flock center (not the target),
        since with scattered sheep they must reach the flock first.
        """
        distances_to_target = np.linalg.norm(self.sheep_positions - self.target_position, axis=1)
        mean_dist_to_target = distances_to_target.mean()
        flock_radius = self._flock_radius()
        mean_dog_dist_to_flock = self._mean_dog_dist_to_flock()

        gather_progress = self.prev_flock_radius - flock_radius
        drive_progress = self.prev_mean_dist - mean_dist_to_target
        dog_progress = self.prev_mean_dog_dist - mean_dog_dist_to_flock
        self.prev_flock_radius = flock_radius
        self.prev_mean_dist = mean_dist_to_target
        self.prev_mean_dog_dist = mean_dog_dist_to_flock

        gathered = self.gathered
        drive_weight = 10.0 if gathered else 2.0

        reward = 5.0 * gather_progress + drive_weight * drive_progress
        # Absolute distance penalty: without it, a gathered flock parked far from
        # the target (e.g. pinned in a corner) is reward-neutral and PPO stalls there
        reward += -1.0 * mean_dist_to_target
        reward += -0.5 * flock_radius - 0.01

        if gathered:
            # Driving-position shaping (Strombom-style): reward dogs for standing
            # just beyond the flock on the side OPPOSITE the target, where their
            # repulsion naturally pushes the flock toward the target
            center = self.sheep_positions.mean(axis=0)
            away = center - self.target_position
            away = away / (np.linalg.norm(away) + 1e-8)
            max_radius = np.max(np.linalg.norm(self.sheep_positions - center, axis=1))
            drive_point = np.clip(center + (max_radius + 0.08) * away, 0.0, self.world_size)
            mean_dog_dist_to_dp = np.mean(np.linalg.norm(self.dog_positions - drive_point, axis=1))
            reward += -2.0 * mean_dog_dist_to_dp
        else:
            reward += 1.0 * dog_progress

        # Small per-sheep in-target bonus (kept << success reward so holding the
        # flock at the target without finishing never beats terminating)
        reward += 0.2 * np.sum(distances_to_target < self.target_distance_threshold)

        if np.all(distances_to_target < self.target_distance_threshold):
            reward = 10000
        return reward

    def _check_done(self):
        done = False
        truncated = False
        distances_to_target = np.linalg.norm(self.sheep_positions - self.target_position, axis=1)
        if np.all(distances_to_target < self.target_distance_threshold):
            done = True
        elif self.current_step >= self.max_steps:
            truncated = True
        return done, truncated

    def render(self, mode="human"):
        if self.fig is None or self.ax is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.set_xlim(0, self.world_size)
            self.ax.set_ylim(0, self.world_size)
            self.ax.set_title("Scattered Sheep-Herding Environment")
            self.ax.set_facecolor('green')
            self.scat_sheep = self.ax.scatter([], [], c="white", label="Sheep")
            self.scat_dogs = self.ax.scatter([], [], c="brown", label="Dogs")
            self.target_plot = self.ax.scatter(self.target_position[0], self.target_position[1],
                                               c="red", marker="X", s=100, label="Target")
            self.target_circle = plt.Circle(self.target_position, self.target_distance_threshold,
                                            color='red', fill=False, linestyle='--')
            self.ax.add_patch(self.target_circle)
            self.ax.legend(loc="upper left")

        self.scat_sheep.set_offsets(self.sheep_positions)
        self.scat_dogs.set_offsets(self.dog_positions)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        buf = BytesIO()
        self.fig.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        return np.array(image)

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == "__main__":
    # Smoke test: random actions, check shapes / bounds / reward phases
    import matplotlib
    matplotlib.use("Agg")
    env = ScatteredSheepDogEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, obs.shape
    print("initial flock radius:", round(env._flock_radius(), 3))
    total = 0.0
    for t in range(200):
        obs, r, done, trunc, info = env.step(env.action_space.sample())
        total += r
        assert obs.shape == env.observation_space.shape
        assert np.all(obs >= 0.0) and np.all(obs <= env.world_size + 1e-9)
        if done or trunc:
            break
    print("200 random steps ok, cumulative reward:", round(total, 2))
    print("flock radius after:", round(env._flock_radius(), 3))
    frame = env.render()
    print("render frame shape:", frame.shape)
    env.close()
    print("SMOKE TEST PASSED")
