# File: multiagent-predator-prey/maddpg_train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.predator import Predator
from envs.predator_prey import PredatorPreyEnv
import matplotlib.pyplot as plt

class MADDPG:
    def __init__(self, input_dim, action_dim, num_agents):
        self.num_agents = num_agents
        self.agents = [Predator(input_dim, action_dim, algorithm='MADDPG') for _ in range(num_agents)]
        self.gamma = 0.99
        self.tau = 0.01  # Soft update parameter

    def update_agents(self, batch):
        for agent in self.agents:
            agent.update_critic_network(batch)
            agent.update_actor_network(batch, self.agents)

    def update_target_networks(self):
        for agent in self.agents:
            agent.update_target_network(self.tau)

    def train(self, env, num_episodes=500, averaging_window=10):
        episode_returns = []
        environment_steps = 0

        for episode in range(num_episodes):
            states = env.reset()
            done = False
            episode_rewards = np.zeros(self.num_agents)
            steps_in_episode = 0

            while not done:
                actions = [agent.act(state) for agent, state in zip(self.agents, states)]
                next_states, rewards, done, _ = env.step(actions)

                # Convert lists to numpy arrays
                states_np = np.array(states)
                actions_np = np.array(actions)
                rewards_np = np.array(rewards)
                next_states_np = np.array(next_states)
                dones_np = np.array([done] * self.num_agents, dtype=np.float32)  # Ensure it's float for computations

                # Collect experience
                batch = (torch.tensor(states_np).float(),
                         torch.tensor(actions_np).float(),
                         torch.tensor(rewards_np).float().unsqueeze(1),  # Ensure rewards are column vectors
                         torch.tensor(next_states_np).float(),
                         torch.tensor(dones_np).float().unsqueeze(1))  # Ensure dones are column vectors

                self.update_agents(batch)

                states = next_states
                episode_rewards += rewards
                steps_in_episode += 1

                # Debugging information
                if steps_in_episode % 100 == 0:
                    print(f"Episode {episode + 1}, Step {steps_in_episode}:")
                    print(f"Predator Positions: {env.predators}")
                    print(f"Prey Position: {env.prey}")

            environment_steps += steps_in_episode
            episode_returns.append(np.sum(episode_rewards))

            # Update target networks periodically
            if episode % 10 == 0:
                self.update_target_networks()

            print(f"Episode {episode + 1}: Total Reward: {np.sum(episode_rewards)}, Steps: {steps_in_episode}")

        # Calculate running average of episode returns
        running_avg_returns = np.convolve(episode_returns, np.ones(averaging_window)/averaging_window, mode='valid')

        # Plotting episode returns against environment steps
        plt.plot(np.arange(len(running_avg_returns)), running_avg_returns)
        plt.xlabel('Environment Steps')
        plt.ylabel('Episode Returns')
        plt.title('Episode Returns vs. Environment Steps')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Example environment setup
    input_dim = 2 * (3 + 2 + 1)  # Update this to match your environment's state space for the predators
    action_dim = 2  # This should match the action space for the predators
    num_agents = 3  # Number of predators

    maddpg = MADDPG(input_dim, action_dim, num_agents)

    # Initialize the actual environment
    env = PredatorPreyEnv(num_predators=num_agents)
    maddpg.train(env)
