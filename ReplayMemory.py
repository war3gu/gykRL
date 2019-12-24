from collections import deque
import random
import numpy as np

class ReplayMemory(object):

	def __init__(self,max_size = 100000,random_seed = 123):
		self.max_size = max_size
		self.buffer = deque(maxlen = self.max_size)
		self.gamma = 0.99
		random.seed(random_seed)

	def add(self,state,action,reward,done,next_state, prob):
		exp = (state,action,reward,done,next_state,prob)
		self.buffer.append(exp) 
	
	def size(self):
		return len(self.buffer)

	def miniBatch(self,batch_size):
		miniBatch = random.sample(self.buffer,min(self.size(),batch_size))
		state_batch = np.array([_[0] for _ in miniBatch])
		action_batch = np.array([_[1] for _ in miniBatch])
		reward_batch = np.array([_[2] for _ in miniBatch])
		done_batch = np.array([_[3] for _ in miniBatch])
		next_state_batch = np.array([_[4] for _ in miniBatch])
		prob_batch = np.array([_[5] for _ in miniBatch])
		return state_batch,action_batch,reward_batch,done_batch,next_state_batch,prob_batch

	def miniAll(self):
		miniBatch = self.buffer
		state_batch = np.array([_[0] for _ in miniBatch])
		action_batch = np.array([_[1] for _ in miniBatch])
		reward_batch = np.array([_[2] for _ in miniBatch])
		done_batch = np.array([_[3] for _ in miniBatch])
		next_state_batch = np.array([_[4] for _ in miniBatch])
		prob_batch = np.array([_[5] for _ in miniBatch])
		return state_batch, action_batch, reward_batch, done_batch, next_state_batch, prob_batch

	def miniAllAfterTransform(self):
		state_batch, action_batch, reward_batch, done_batch, next_state_batch, prob_batch = self.miniAll()

		transform_reward_batch = self.discount_rewards(reward_batch)

		return state_batch, action_batch, transform_reward_batch, done_batch, next_state_batch, prob_batch

	def discount_rewards(self, rewards):
		discounted_rewards = np.zeros_like(rewards)
		running_add = 0
		for t in reversed(range(0, rewards.size)):
			#if rewards[t] != 0:
				#running_add = 0
			running_add = running_add * self.gamma + rewards[t]
			discounted_rewards[t] = running_add

		#求均值，归一化
		center  = discounted_rewards - np.mean(discounted_rewards)
		discounted_rewards = center / np.std(center)

		return discounted_rewards

	def clear(self):
		self.buffer.clear()