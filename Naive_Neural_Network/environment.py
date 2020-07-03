import gym
import matplotlib.pyplot as plt
from agent import Agent
import numpy as np

EPISODES = 10_000
rewards_per_episode = []
epsilon_over_time = []
rewards_overall = []

env = gym.make('CartPole-v0')
action_space_dims = env.action_space.n
observation_space_dims = env.observation_space.shape[0]

agent = Agent(state_dims=observation_space_dims, action_dims=action_space_dims,
              gamma=0.99, epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.9985, learning_rate=0.001)


for i in range(EPISODES):

    state = env.reset()
    done = False
    score = 0

    while not done:

        env.render()
        # take some action for now just random
        action = agent.choose_action(state)
        new_state, reward, done, info = env.step(action)

        agent.learn(state, action, new_state, reward)
        score += reward
        state = new_state

    rewards_per_episode.append(score)

    if i % 100 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        epsilon_over_time.append(agent.epsilon)
        rewards_overall.append(avg_reward)

        if i % 1000 == 0:
            print('episode: ', i, 'avg reward:  %.2f ' % avg_reward, 'epsilon: %.2f ' % agent.epsilon)


env.close()

plt.plot(rewards_overall)
plt.plot(epsilon_over_time)
plt.xlabel('Number of Episodes per 100 episodes')
plt.ylabel('avg reward & epsilon')
plt.legend(['Avg Reward', 'Epsilon'])
plt.show()







