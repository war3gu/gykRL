

import sys, os
curr_dir = os.path.dirname(os.path.realpath(__file__))
upPath = curr_dir+'/../'
sys.path.append(upPath)



import gym

import numpy as np

from Actor import Actor

from Critic import Critic

from ReplayMemory import ReplayMemory

import argparse

import tensorflow as tf






parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--mode', default=0, help='0.train 1.play', dest='mode', type=int)
args = parser.parse_args()

mode = args.mode




def build_summaries():
	episode_reward =   tf.Variable(0.)
	tf.summary.scalar("Reward",episode_reward)
	summary_vars = [episode_reward]
	summary_ops = tf.summary.merge_all()
	return summary_ops, summary_vars

def train():

    env = gym.make('LunarLander-v2')

    state = env.reset()

    actor = Actor(env.action_space, env.observation_space)

    critic = Critic(env.action_space, env.observation_space)


    actor.load()
    critic.load()


    replayMemory = ReplayMemory()

    summary_ops, summary_vars = build_summaries()

    writer = tf.summary.FileWriter("./log", tf.Session().graph)

    episode_reward = 0

    step = 1

    while True:

        #env.render()

        state1 = state[np.newaxis, :]

        action, action_matrix, prob = actor.predict(state1)

        next_state, reward, done, info = env.step(action)

        replayMemory.add(state, action_matrix, reward, done, next_state, prob)

        state = next_state

        episode_reward += reward

        #train
        if replayMemory.size() % 128 == 0 or done == True:

            state_b, action_matrix_b, reward_b, done_b, next_state_b, prob_b = replayMemory.miniAll()

            reward_b = reward_b[:, np.newaxis]

            c_pre = critic.predict(next_state_b)

            state_pre_value = reward_b + c_pre*0.7

            state_value = critic.predict(state_b)

            count = 5000//step

            if count > 500:
                count = 500

            if count < 1:
                count = 1

            count = 10

            for _ in range(count):
                critic.train(state_b, state_pre_value)

            for _ in range(count):
                actor.train(state_b, state_value, state_pre_value, action_matrix_b, prob_b)

            replayMemory.clear()
        ########################


        if done:

            summary_str = tf.Session().run(summary_ops, feed_dict={summary_vars[0]: episode_reward})
            writer.add_summary(summary_str, step)
            writer.flush()

            ##print("step = ", step, "episode_reward = ", episode_reward)

            state = env.reset()

            episode_reward = 0

            step += 1

            if step%25 == 0:
                actor.save()
                critic.save()


def play():
    print("play")

    env = gym.make('LunarLander-v2')

    state = env.reset()

    actor = Actor(env.action_space, env.observation_space)

    actor.load()

    #critic = Critic(env.action_space, env.observation_space)

    #replayMemory = ReplayMemory()

    #summary_ops, summary_vars = build_summaries()

    #writer = tf.summary.FileWriter("./log", tf.Session().graph)

    #episode_reward = 0

    #step = 1

    while True:

        env.render()

        state1 = state[np.newaxis, :]

        action, action_matrix, prob = actor.predict(state1)

        next_state, reward, done, info = env.step(action)

        #replayMemory.add(state, action_matrix, reward, done, next_state, prob)

        state = next_state

        if done:
            #summary_str = tf.Session().run(summary_ops, feed_dict={summary_vars[0]: episode_reward})
            #writer.add_summary(summary_str, step)
            #writer.flush()
            state = env.reset()

    return 0



if __name__ == '__main__':

    if mode == 0:
        train()
    else:
        play()