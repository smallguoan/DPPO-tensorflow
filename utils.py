import tensorflow as tf
import numpy as np

from collections import namedtuple
import random

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.atari_wrappers import FrameStack

class utils(object):
    def __init__(self,memory_size,batch_size,gamma,gae_lambda):
        self.memory_size=memory_size
        self.batch_size=batch_size
        self.gamma=gamma
        self.gae_lambda=gae_lambda
        self.Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        self.replay_memo = []
        self.gae=self.gamma*self.gae_lambda

    def store_transition(self,s,a,r,s_,done):

        if len(self.replay_memo)>=128:
            self.replay_memo.pop(0)
        self.replay_memo.append(self.Transition(s, a, r, s_, done))

    def clear_memory(self):
        self.replay_memo=[]

    # def sample_memory(self,size):
    #     batch_memo = random.sample(self.replay_memo, size)
    #     states_batch, action_batch, reward_batch, next_states_batch, done_batch,advantage_batch = map(np.array, zip(*batch_memo))
    #
    #     return states_batch, action_batch, reward_batch, next_states_batch, done_batch,advantage_batch

    def calculate_advantage(self,reward,state_value,next_state_value,done):
        advantage=np.zeros_like(reward)
        advantage_t=0
        for t in reversed(range(self.memory_size)):

            delta=reward[t]+(1-done[t])*self.gamma*np.squeeze(next_state_value)[t]-np.squeeze(state_value)[t]
            advantage_t=delta+self.gae*(1-done[t])*advantage_t
            advantage[t]=advantage_t

        returns=state_value[:self.memory_size,0]+advantage
        #assert advantage.shape==(128,8)   #Just confirming for atari games
        return advantage,returns

# class Preprocess(object):
#     def __init__(self,sess):
#         self.sess=sess
#         with tf.variable_scope("state_processor"):
#             self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
#             self.output = tf.image.rgb_to_grayscale(self.input_state)
#             self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
#             self.output = tf.image.resize_images(
#                 self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#             self.output = tf.squeeze(self.output)
#
#     def process(self,s):
#         return self.sess.run(self.output,feed_dict={self.input_state:s})


def make_env(env_id, seed, rank):
    env = make_atari(env_id)
    env.seed(rank + seed)
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False)
    return env