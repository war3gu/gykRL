

import gym

import numpy as np

import tensorflow as tf

from Actor import Actor

from Critic import Critic

from ReplayMemory import ReplayMemory

from ExplorationNoise import OrnsteinUhlenbeckActionNoise  as OUNoise


def build_summaries():
	episode_reward =   tf.Variable(0.)
	tf.summary.scalar("Reward",episode_reward)
	summary_vars = [episode_reward]
	summary_ops = tf.summary.merge_all()
	return summary_ops, summary_vars


if __name__=='__main__':
    sess = tf.Session()

    env = gym.make('LunarLanderContinuous-v2')
    state = env.reset()

    actor = Actor(sess, env.action_space, env.observation_space)
    critic = Critic(sess, env.action_space, env.observation_space)
    sess.run(tf.global_variables_initializer())

    replayMemory = ReplayMemory(max_size=1000000)

    oun = OUNoise(mu=np.zeros(env.action_space.shape[0]))

    actor.update_target()
    critic.update_target()


    summary_ops, summary_vars = build_summaries()

    writer = tf.summary.FileWriter("./log", sess.graph)

    episode_reward = 0

    step = 1

    while True:
        #env.render()

        actionPure = actor.act(state)
        actionNoise = actionPure + oun()
        next_state, reward, done, info = env.step(actionNoise)

        replayMemory.add(state, actionNoise, reward, done, next_state, None)

        state = next_state

        episode_reward += reward

        if replayMemory.size() >= 10000:
            state_b, action_b, reward_b, done_b, next_state_b, prob_b = replayMemory.miniBatch(int(64))

            targetQ = critic.predict_target(next_state_b, actor.predict_target(next_state_b))
            yi = []
            for k in range(int(64)):
                if done_b[k]:  # 结束，没有下个状态
                    yi.append(reward_b[k])
                else:
                    yi.append(reward_b[k] + 0.99 * targetQ[k])

            yx = np.reshape(yi, (int(64), 1))  # critic的目标
            critic.train(state_b, action_b, yx)

            actions_pred = actor.predict(state_b)  # actions_pred与a_batch不一样，a_batch是加了噪声的
            grads = critic.get_action_gradients(state_b, actions_pred)  # 算出Q对action的梯度
            actor.train(state_b, grads)  # actor的参数朝Q变大的方向上稍微移动一点
            actor.update_target()
            critic.update_target()

            #replayMemory.clear()
        if done:

            summary_str = sess.run(summary_ops, feed_dict={summary_vars[0]: episode_reward})
            writer.add_summary(summary_str, step)
            writer.flush()

            print("step = ", step, "episode_reward = ", episode_reward)

            state = env.reset()

            oun.reset()

            episode_reward = 0

            step += 1

