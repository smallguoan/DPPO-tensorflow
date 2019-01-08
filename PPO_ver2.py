import tensorflow as tf
import numpy as np

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.atari_wrappers import FrameStack
from gym.wrappers import Monitor

from models import Models
from utils import utils
import sys

class DPPO(object):
    def __init__(self, sess,venv, model, total_steps, test_env, tools, batch_size,work_steps,lr_init,test_iter):
        self.sess=sess
        self.venv = venv
        self.model = model
        self.total_steps = total_steps
        self.test_env = test_env
        self.tools = tools
        self.num_workers = self.venv.num_envs
        self.batch_size = batch_size
        self.epsilon = tf.placeholder(tf.float32, [], name="epsilon")
        self.max_grad_norm=0.5
        self.lr = tf.placeholder(tf.float32, [], name="LR")
        self.work_steps=work_steps
        self.opt_epoch=3
        self.taken_steps=0
        self.lr_init=lr_init
        self.epsilon_init=0.1
        self.last_states=self.venv.reset()
        self.test_iter=test_iter
        self.saver=tf.train.Saver([t for t in tf.trainable_variables()])

        with tf.variable_scope("c_loss"):
            TD_error = self.model.returns - self.model.value
            self.c_loss = 0.5 * tf.reduce_mean(tf.square(TD_error))

        with tf.variable_scope("a_loss"):
            indice = tf.expand_dims(tf.range(self.batch_size) * tf.shape(self.model.pi)[1],
                                         axis=1) + self.model.acts
            pi_prob=tf.gather(tf.reshape(tf.nn.softmax(self.model.pi), [-1]), indice)
            oldpi_prob=tf.gather(tf.reshape(tf.nn.softmax(self.model.oldpi), [-1]), indice)
            logpi_prob=tf.gather(tf.reshape(tf.nn.log_softmax(self.model.pi), [-1]), indice)
            ratio = pi_prob / (oldpi_prob + 1e-10)
            surr = ratio * self.model.advantages
            self.a_loss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1. - self.epsilon,
                                                                            1. + self.epsilon) * self.model.advantages))
        with tf.variable_scope("entropy_loss"):
            #self.entropy_loss=0.01*tf.reduce_mean(pi_prob*logpi_prob)
            self.entropy_loss=0.01*tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(self.model.oldpi)*tf.nn.log_softmax(self.model.pi),axis=1))

        with tf.variable_scope("update_pi"):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.model.pi_params, self.model.oldpi_params)]

        with tf.variable_scope("total_loss"):
            self.loss=self.c_loss+self.a_loss+self.entropy_loss

        with tf.variable_scope("train_op"):
            opt = tf.train.AdamOptimizer(self.lr, epsilon=1e-5)
            grad_var = opt.compute_gradients(self.loss, self.model.pi_params)
            grads, var = zip(*grad_var)
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            grads_and_var = list(zip(grads, var))
            self.train_op = opt.apply_gradients(grads_and_var)

        self.sess.run(tf.global_variables_initializer())
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss)])
        self.writer = tf.summary.FileWriter("log/", self.sess.graph)
        #self.update_oldpi()

    def choose_actions(self,state,test_mode):
        if test_mode:
            state=np.expand_dims(state,axis=0)
            action_values = self.sess.run(self.model.pi, feed_dict={self.model.s: state})
            noise=np.random.rand(self.venv.action_space.n)
            action = np.argmax(action_values - np.log(-np.log(noise)))
        else:
            action_values=self.sess.run(self.model.pi,feed_dict={self.model.s:state})
            noise=np.random.rand(self.num_workers,self.venv.action_space.n)
            action=np.argmax(action_values-np.log(-np.log(noise)),axis=1)

        return action

    def update_oldpi(self):
        self.sess.run(self.update_oldpi_op)

    def run(self):

        N = self.num_workers
        T = self.work_steps
        E = self.opt_epoch
        A = self.venv.action_space.n

        while self.taken_steps<=self.total_steps:
            decay=self.taken_steps/self.total_steps
            lr=self.lr_init*(1.-decay)
            epsilon=self.epsilon_init*(1.-decay)
            memo,steps=self.interact()
            states_m,actions_m,rewards_m,next_states_m,dones_m=map(np.array,zip(*memo))
            rewards_m=np.reshape(rewards_m,(T,N,1))
            dones_m=np.reshape(dones_m,(T,N,1))
            states_m=np.reshape(states_m,(N*T,)+states_m.shape[2:])
            next_states_m=np.reshape(next_states_m,(N*T,)+next_states_m.shape[2:])

            states_m=states_m.astype(np.float32)        #(1024,84,84,4)
            next_states_m = next_states_m.astype(np.float32)

            states_values=self.get_v(states_m)       #(1024,1)
            next_states_values=self.get_v(next_states_m)

            states_values=np.reshape(states_values,(T,N,1))     #(128,8,1)
            next_states_values=np.reshape(next_states_values,(T,N,1))

            advantages,returns=self.tools.calculate_advantage(rewards_m,states_values,next_states_values,dones_m)
            # advantages=np.expand_dims(advantages,axis=2)
            # returns=np.expand_dims(returns,axis=2)

            # Training stage
            print("\rUpdate oldpi")
            self.update_oldpi()
            for i in range(E):

                states_values=np.reshape(states_values,(steps,1))
                next_states_values=np.reshape(next_states_values,(steps,1))
                advantages=np.reshape(advantages,(steps,1))
                returns=np.reshape(returns,(steps,1))
                actions_m=np.reshape(actions_m,(steps,1))
                rewards_m=np.reshape(rewards_m,(steps,1))
                dones_m=np.reshape(dones_m,(steps,1))



                index=np.arange(steps)
                np.random.shuffle(index)

                for start in range(0, steps, self.batch_size):
                    mb_inds=index[start:start+self.batch_size]
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones,batch_advantages,batch_returns = \
                        [arr[mb_inds] for arr in [states_m, actions_m,rewards_m,next_states_m,dones_m,advantages,returns]]

                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-5)
                    self.sess.run(self.train_op, feed_dict={self.model.s: batch_states, self.model.acts: batch_actions,
                                                            self.model.R: batch_rewards,
                                                            self.model.advantages: batch_advantages,
                                                            self.model.returns: batch_returns,self.lr:lr,self.epsilon:epsilon})

            self.tools.clear_memory()
            self.taken_steps+=steps
            print("Taken_steps: {}/{}".format(self.taken_steps,self.total_steps))

            if self.taken_steps%self.test_iter==0:
                ep_aver=self.test()
                print("Test mode in taken_steps: {}, average_reward: {}".format(self.taken_steps,ep_aver))
        self.saver.save(self.sess,"./actor.ckpt")
    def interact(self):
        N = self.num_workers
        T = self.work_steps

        for i in range(T):
            states=self.last_states
            actions=self.choose_actions(states,test_mode=False)
            self.last_states,rewards,dones,_=self.venv.step(actions)
            self.tools.store_transition(states,actions,rewards,self.last_states,dones)

        steps = N * T

        return self.tools.replay_memo,steps
    def test(self):
        test_env=Monitor(self.test_env,"./exp3",video_callable=lambda count:count%1==0,resume=True)
        ep_aver=0
        ep=[]
        for i in range(self.num_workers):
            obs = test_env.reset()
            ep_r = 0
            while True:
                #test_env.render()
                action = self.choose_actions(obs,test_mode=True)
                obs_, reward, done, _ = test_env.step(action)
                ep_r+=reward
                if done:
                    ep.append(ep_r)
                    print("episode_i: {},ep_r: {}".format(i,ep_r))
                    break
                obs = obs_
        ep_aver=sum(ep)/self.num_workers
        return ep_aver

    def get_v(self,state):
        return self.sess.run(self.model.value,feed_dict={self.model.s:state})



# def make_env(env_id, seed, rank):
#     env = make_atari(env_id)
#     env.seed(rank + seed)
#     env = wrap_deepmind(env, episode_life=False, clip_rewards=False)
#     return env
#
# ENV_ID = "BreakoutNoFrameskip-v4"
#
# env_fn = []
# for rank in range(8):
#     env_fn.append(lambda: make_env(ENV_ID, 0 + rank, rank=rank))
#
# venv = SubprocVecEnv(env_fn)
# venv = VecFrameStack(venv, 4)
#
# test_env = make_env("BreakoutNoFrameskip-v4", 0, 0)
# test_env = FrameStack(test_env, 4)
#
# sess = tf.Session()
# models = Models(venv.action_space.n)
# tools = utils(memory_size=128, batch_size=256, gamma=0.99, gae_lambda=0.95)
#
# dppo = DPPO(sess=sess, venv=venv, model=models, total_steps=1e7, test_env=test_env, tools=tools, batch_size=256,
#             work_steps=128, lr_init=2.5e-4, test_iter=1024 * 100)
# dppo.run()
