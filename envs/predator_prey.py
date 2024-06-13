# File: multiagent-predator-prey/envs/predator_prey.py

import gym
from gym import spaces
import numpy as np

class PredatorPreyEnv(gym.Env):
    """
    A simple predator-prey environment
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, num_predators=3, num_obstacles=2, max_steps=1000):
        super(PredatorPreyEnv, self).__init__()

        self.num_predators = num_predators
        self.num_obstacles = num_obstacles
        self.prey_speed = 2.0  # Prey is faster
        self.predator_speed = 1.0  # Predators are slower
        self.view_radius = 5.0  # View radius for partial observability
        self.arena_size = 20.0  # Size of the arena (toroidal plane)
        self.max_steps = max_steps
        self.current_step = 0

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * (self.num_predators + self.num_obstacles + 1),), dtype=np.float32)

        # Initialize state
        self.state = None

    def reset(self):
        # Initialize the positions of predators, prey, and obstacles
        self.predators = np.random.uniform(-self.arena_size / 2, self.arena_size / 2, (self.num_predators, 2))
        self.prey = np.random.uniform(-self.arena_size / 2, self.arena_size / 2, 2)
        self.obstacles = np.random.uniform(-self.arena_size / 2, self.arena_size / 2, (self.num_obstacles, 2))
        self.state = np.concatenate((self.predators.flatten(), self.prey.flatten(), self.obstacles.flatten()))
        self.current_step = 0
        return self._get_observations()

    def _get_observations(self):
        observations = []
        for i in range(self.num_predators):
            predator_pos = self.predators[i]
            obs = []
            for pos in np.vstack((self.predators, [self.prey], self.obstacles)):
                if np.linalg.norm(pos - predator_pos) <= self.view_radius:
                    obs.extend(pos - predator_pos)
                else:
                    obs.extend([0, 0])  # Not visible
            observations.append(np.array(obs))
        return np.array(observations)

    def step(self, actions):
        self.current_step += 1

        # Apply actions to predators
        for i in range(self.num_predators):
            self.predators[i] += self.predator_speed * actions[i]
            self.predators[i] = np.clip(self.predators[i], -self.arena_size / 2, self.arena_size / 2)

        # Move prey with more random movement
        prey_direction = np.random.uniform(-1, 1, 2)
        if np.linalg.norm(prey_direction) > 0:
            prey_direction /= np.linalg.norm(prey_direction)  # Normalize direction
        self.prey += self.prey_speed * prey_direction
        self.prey = np.clip(self.prey, -self.arena_size / 2, self.arena_size / 2)

        # Debug prey movement
        # print(f"Step: {self.current_step}, Prey direction: {prey_direction}, Prey position: {self.prey}")

        # Check for collisions and calculate rewards
        rewards = np.zeros(self.num_predators)
        done = False
        for i in range(self.num_predators):
            distance_to_prey = np.linalg.norm(self.predators[i] - self.prey)
            if distance_to_prey < 1.0:  # Catch radius
                rewards[i] = 10  # Team reward
                done = True
                break  # End the episode if prey is caught

        # Wrap around the arena (toroidal plane)
        self.predators = np.mod(self.predators + self.arena_size / 2, self.arena_size) - self.arena_size / 2
        self.prey = np.mod(self.prey + self.arena_size / 2, self.arena_size) - self.arena_size / 2

        if self.current_step >= self.max_steps:
            done = True

        self.state = np.concatenate((self.predators.flatten(), self.prey.flatten(), self.obstacles.flatten()))
        return self._get_observations(), rewards, done, {}

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError("Only console mode is supported.")
        print(f"Predators: {self.predators}")
        print(f"Prey: {self.prey}")
        print(f"Obstacles: {self.obstacles}")

# Example usage
if __name__ == "__main__":
    env = PredatorPreyEnv()
    obs = env.reset()
    print("Initial Observation:", obs)
    done = False
    while not done:
        actions = np.random.uniform(-1, 1, (env.num_predators, 2))
        obs, rewards, done, _ = env.step(actions)
        env.render()
        print("Rewards:", rewards)
