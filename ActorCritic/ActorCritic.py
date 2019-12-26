
import sys, os
curr_dir = os.path.dirname(os.path.realpath(__file__))
upPath = curr_dir+'/../'
sys.path.append(upPath)

import gym

import numpy as np

from Actor import Actor

from Critic import Critic

from ReplayMemory import ReplayMemory

import tensorflow as tf

def build_summaries():
	episode_reward =   tf.Variable(0.)
	tf.summary.scalar("Reward",episode_reward)
	summary_vars = [episode_reward]
	summary_ops = tf.summary.merge_all()
	return summary_ops, summary_vars



if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    #env = gym.make('CartPole-v0')

    state = env.reset()
    actor = Actor(env.action_space, env.observation_space)
    critic = Critic(env.action_space, env.observation_space)
    replayMemory = ReplayMemory()

    summary_ops, summary_vars = build_summaries()

    writer = tf.summary.FileWriter("./log", tf.Session().graph)

    episode_reward = 0

    step = 0

    while True:
        #env.render()

        state1 = state[np.newaxis, :]

        action, action_matrix = actor.predict(state1)

        next_state, reward, done, info = env.step(action)

        replayMemory.add(state, action_matrix, reward, done, next_state)

        state = next_state

        episode_reward += reward

        #train
        if replayMemory.size() % 128 == 0 or done == True:

            state_b, action_matrix_b, reward_b, done_b, next_state_b = replayMemory.miniAll()

            reward_b = reward_b[:, np.newaxis]

            c_pre = critic.predict(next_state_b)

            state_pre_value = reward_b + c_pre*0.6

            state_value = critic.predict(state_b)

            critic.train(state_b, state_pre_value)

            actor.train(state_b, state_value, state_pre_value, action_matrix_b)

            replayMemory.clear()
        ########################


        if done:

            summary_str = tf.Session().run(summary_ops, feed_dict={summary_vars[0]: episode_reward})
            writer.add_summary(summary_str, step)
            writer.flush()

            print("step = ", step, "episode_reward = ", episode_reward)

            state = env.reset()

            episode_reward = 0

            step += 1