import gym
import numpy as np
import time

env = gym.make('CartPole-v1',render_mode='human')
(state,_)=env.reset()
env.render()
env.step(0)
env.observation_space
env.observation_space.high
env.observation_space.low
env.action_space
env.spec
env.spec.max_episode_steps
env.spec.reward_threshold
episodeNumber=5
timeSteps=100
for episodeIndex in range(episodeNumber):
    initial_state=env.reset()
    print(episodeIndex)
    env.render()
    appendedObservations=[]
    for timeIndex in range(timeSteps):
        print(timeIndex)
        random_action=env.action_space.sample()
        observation, reward, terminated, truncated, info =env.step(random_action)
        appendedObservations.append(observation)
        time.sleep(0.1)
        if (terminated):
            time.sleep(1)
            break
env.close()