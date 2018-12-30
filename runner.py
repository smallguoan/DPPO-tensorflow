import argparse
import tensorflow as tf
import numpy as np

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.atari_wrappers import FrameStack
from gym.wrappers import Monitor

from models import Models
from utils import utils
from PPO_ver2 import DPPO
from utils import make_env


parser = argparse.ArgumentParser(description='DPPO', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env_id', type=str,default='PongNoFrameskip-v4',help='Gym environment id')
parser.add_argument('--num_workers', type=int, default=8, help='number of parallel actors')
parser.add_argument('--opt_epochs', type=int, default=3, help='optimization epochs between environment interaction')
parser.add_argument('--total_steps', type=int, default=1e7, help='total number of environment steps to take')
parser.add_argument('--lr_init',type=float,default=2.5e-4,help='learning rate at the beginning of training')
parser.add_argument('--work_steps',type=int,default=128,help='steps of collecting data for workers')
parser.add_argument('--test_iter',type=int,default=1024*100,help='steps of testing agent during training')
parser.add_argument('--frame_stack',type=int,default=4,help='the number of frames for every state')
parser.add_argument('--batch_size',type=int,default=256,help='the number of training data per step')
parser.add_argument('--gamma',type=float,default=0.99,help='decay factor of values of reward')
parser.add_argument('--gae_lambda',type=float,default=0.95,help='decay factor for calculating advantages')

args = parser.parse_args()

print("env ID",args.env_id)

env_fn = []
for rank in range(args.num_workers):
    env_fn.append(lambda: make_env(args.env_id, 0 + rank, rank=rank))

venv = SubprocVecEnv(env_fn)
venv = VecFrameStack(venv, args.frame_stack)

test_env = make_env(args.env_id, 0, 0)
test_env = FrameStack(test_env, args.frame_stack)

sess = tf.Session()
models = Models(venv.action_space.n)

tools = utils(memory_size=args.work_steps, batch_size=args.batch_size, gamma=args.gamma,
              gae_lambda=args.gae_lambda)

dppo = DPPO(sess=sess, venv=venv, model=models, total_steps=args.total_steps, test_env=test_env, tools=tools,
            batch_size=args.batch_size, work_steps=args.work_steps, lr_init=args.lr_init, test_iter=args.test_iter)
dppo.run()
