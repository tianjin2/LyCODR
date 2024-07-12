import argparse  # 导入命令行参数解析库 argparse
import os
import uuid
from coma.algos import coma
from coma.examples.variants import parse_domain_and_transaction, get_variants
from coma.misc.utils import timestamp, unflatten
from coma.envs import constant
from gym.envs.registration import register
from coma.envs.gym_env import GymEnv, GymEnvDelayed
from coma.misc.instrument import run_coma_experiment
from coma.replay_buffer import SimpleReplayBuffer
from coma.misc.sampler import SimpleSampler
from coma.algos.coma import COMA



DELAY_CONST = 20


ENVIRONMENTS = {
    'mecs': {
        'default': lambda: GymEnv('MECS-v0',len(shard_list)),
    },
}

ENVS = ['mecs1',
        'mecs2',
        'mecs3',
        'mecs4',
        'mecs5',
        'mecs6',
        'mecs7',
        'mecs8',
        'mecs9',
        'mecs10',
        'swimmer-gym', # 2 DoF
        'hopper', # 3 DoF
        'half-cheetah', # 6 DoF
        'walker', # 6 DoF
        'humanoid-gym', # 17 DoF # gym_humanoid
        'humanoid-standup-gym']

AVAILABLE_DOMAINS = set(ENVIRONMENTS.keys())
AVAILABLE_TRANSACTIONS = set(y for x in ENVIRONMENTS.values() for y in x.keys())



def parse_args():

    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument('--cost_type', type=float, default=10)
    parser.add_argument('--domain', type=str, choices=AVAILABLE_DOMAINS, default=None)
    parser.add_argument('--transaction', type=str, choices=AVAILABLE_TRANSACTIONS, default='default')
    parser.add_argument('--envn',  type=str, default='mecs')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')


    args = parser.parse_args()
    args.env = args.envn


    log_dir = os.getcwd()
    log_dir += '_c%s' % args.cost_type

    log_dir += '/coma'
    if not args.scale == 1.0:
        log_dir += '_s%s' % str(args.scale)
    args.log_dir = log_dir
    return args


shard_list = {}
i = 0
for e in ENVS:
    if e[:4] == 'mecs' and i <= 3:
        shard_list[e] = uuid.uuid4()
        i += 1


def run_experiment(variants, agent_num):
    for _, variant in enumerate(variants):

        env_params = variant['env_params']
        algorithm_params = variant['algorithm_params']
        replay_buffer_params = variant['replay_buffer_params']
        variant['sampler_params']['agent_num'] = agent_num
        sampler_params = variant['sampler_params']

        transaction = variant['transaction']
        domain = variant['domain']


        constant.COST_TYPE = variant['algorithm_params']['cost_type']


        register(
            id='MECS-v0',
            entry_point='coma.envs.env_V_sweep_v:MEC',
            max_episode_steps=5000,
            kwargs={
                'shard_list': shard_list,
            }
        )

        env = ENVIRONMENTS[domain][transaction](**env_params)


        pool = SimpleReplayBuffer(agent_num=agent_num, env=env, **replay_buffer_params)


        sampler = SimpleSampler(**sampler_params)

        base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

        algorithm = COMA(
            base_kwargs=base_kwargs,
            env=env,
            pool=pool,
            agent_num=agent_num,
            lr_a=algorithm_params['lr_a'],
            lr_c=algorithm_params['lr_c'],
            tau=algorithm_params['tau'],
            gamma=algorithm_params['discount'],
            scale_reward=algorithm_params['scale'] * algorithm_params['scale_reward'],
            initial_exploration_done=False,
            target_update_steps=algorithm_params['target_update_interval'],
            save_full_state=False,
        )
        algorithm.train()
    

def launch_experiments(variant, args):
    # agent_num = len(variant_generator)
    agent_num = 4

    variants = variant.variants()

    variants = [unflatten(variant, separator='.') for variant in variants]
    for j,var in enumerate(variants):
        print("实验: {}/{}".format(i, agent_num))
        run_params = var['run_params']
        algo_params = var['algorithm_params']
        var['algorithm_params']['scale'] = args.scale
        var['algorithm_params']['cost_type'] = args.cost_type


        experiment_prefix = var['prefix'] + '/' + args.exp_name
        experiment_name = '{prefix}-{exp_name}-{i:02}'.format(
            prefix=var['prefix'], exp_name=args.exp_name, i=1
        )

        run_coma_experiment(
            run_experiment(variants,agent_num),
            mode=args.mode,
            variant=variant,
            exp_prefix=experiment_prefix,
            exp_name=experiment_name,
            n_parallel=1,
            seed=run_params['seed'],
            log_dir=args.log_dir,
            terminate_machine=True,
            snapshot_mode=run_params['snapshot_mode'],
            snapshot_gap=run_params['snapshot_gap'],
            sync_s3_pkl=run_params['sync_pkl']
        )


def main():
    args = parse_args()

    domain, transaction = args.env, args.transaction

    if (not domain) or (not transaction):
        domain, transaction = parse_domain_and_transaction(args.env)


    variant_generator = get_variants(domains=domain, transaction=transaction)

    launch_experiments(variant_generator, args)
if __name__ == '__main__':
    main()