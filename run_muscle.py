#!/usr/bin/env python3

from baselines.common.cmd_util import common_arg_parser
from baselines.common import tf_util as U
from baselines import logger

import sys
sys.path.insert(0,"/home/visak/Documents/MuscleDynamics")
from TwoDofArm import TwoDofArmEnv


def train(num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=128, num_hid_layers=2)
    env = TwoDofArmEnv(ActiveMuscles='antagonistic',actionParameterization=True,sim_length=0.005,
        traj_track=True,exo=True,exo_gain=70.,delay=0.020)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=1048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()

def main():
    args = common_arg_parser().parse_args()
    logger.configure()
    train(num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()