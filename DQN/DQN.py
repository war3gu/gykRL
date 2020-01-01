

import sys, os
curr_dir = os.path.dirname(os.path.realpath(__file__))
upPath = curr_dir+'/../'
sys.path.append(upPath)



import gym

import numpy as np

from Actor import Actor

from  ReplayMemory import ReplayMemory

import tensorflow as tf

def build_summaries():
	episode_reward =   tf.Variable(0.)
	tf.summary.scalar("Reward",episode_reward)
	summary_vars = [episode_reward]
	summary_ops = tf.summary.merge_all()
	return summary_ops, summary_vars



if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    state = env.reset()

    actor = Actor(env.action_space, env.observation_space)

    replayMemory = ReplayMemory()

    summary_ops, summary_vars = build_summaries()

    writer = tf.summary.FileWriter("./log", tf.Session().graph)

    episode_reward = 0

    step = 1

    while True:
        #env.render()

        state1 = state[np.newaxis, :]

        action, action_matrix, prob = actor.act(state1)

        next_state, reward, done, info = env.step(action)

        replayMemory.add(state, action, reward, done, next_state, prob)

        state = next_state

        episode_reward += reward
##############################train######################
        if replayMemory.size() >= 128:
            state_b, action_b, reward_b, done_b, next_state_b, prob_b = replayMemory.miniBatch(int(64))
            next_state_b_value = actor.predict(next_state_b)
            state_b_value = actor.predict(state_b)
            length = state_b.shape[0]

            for i in range(length):
                target_next = reward_b[i]
                if not done_b[i]:
                    action_values = next_state_b_value[i]
                    target_next = (reward_b[i] + 0.7 * np.amax(action_values))
                state_b_value[i][action_b[i]] = target_next
            actor.train(state_b, state_b_value)

        if done:
            summary_str = tf.Session().run(summary_ops, feed_dict={summary_vars[0]: episode_reward})
            writer.add_summary(summary_str, step)
            writer.flush()

            print("step = ", step, "episode_reward = ", episode_reward)

            state = env.reset()

            episode_reward = 0

            step += 1
