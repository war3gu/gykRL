
import sys, os
curr_dir = os.path.dirname(os.path.realpath(__file__))
upPath = curr_dir+'/../'
sys.path.append(upPath)

import gym
from Actor import Actor
from ReplayMemory import ReplayMemory
import tensorflow as tf



def build_summaries():
	episode_reward =   tf.Variable(0.)
	tf.summary.scalar("Reward",episode_reward)
	summary_vars = [episode_reward]
	summary_ops = tf.summary.merge_all()
	return summary_ops, summary_vars


if __name__ == '__main__':
    print('hello world')
    #env = gym.make('LunarLander-v2')
    env = gym.make('CartPole-v0')
    state = env.reset()
    actor = Actor(env.action_space, env.observation_space)
    replayMemory = ReplayMemory()

    summary_ops, summary_vars = build_summaries()

    writer = tf.summary.FileWriter("./log", tf.Session().graph)

    episode_reward = 0

    step = 0

    while True:
        env.render()

        action, action_matrix = actor.predict(state)

        next_state, reward, done, info = env.step(action)

        replayMemory.add(state, action_matrix, reward, done, next_state)

        state = next_state

        episode_reward += reward

        if done:
            state_b, action_matrix_b, transform_reward_b, done_b, next_state_b = replayMemory.miniAllAfterTransform()

            actor.train(state_b, transform_reward_b, action_matrix_b)

            summary_str = tf.Session().run(summary_ops, feed_dict={summary_vars[0]: episode_reward})
            writer.add_summary(summary_str, step)
            writer.flush()

            print("step = ", step)


            state = env.reset()

            replayMemory.clear()

            episode_reward = 0

            step += 1
































