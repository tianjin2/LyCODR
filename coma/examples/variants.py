from coma.misc.utils import get_git_rev, deep_update, flatten,VariantGenerator


M = 256
REPARAMETERIZE = True
DOMAINS = [
    'swimmer-gym', # 2 DoF
    'hopper', # 3 DoF
    'half-cheetah', # 6 DoF
    'walker', # 6 DoF
    'humanoid-gym', # 17 DoF # gym_humanoid
    'humanoid-standup-gym', # 17 DoF # gym_humanoid
    'mecs', # 17 DoF # gym_humanoid
]


TRANSACTIONS = {
    'swimmer-gym': [
        'default',
        'delayed',
    ],
    'hopper': [
        'default',
        'delayed',
    ],
    'half-cheetah': [
        'default',
        'delayed',
    ],
    'walker': [
        'default',
        'delayed',
    ],
    'humanoid-gym': [
        'default',
        'delayed',
    ],
    'humanoid-standup-gym': [
        'default',
        'delayed',
    ],
    'mecs': [
        'default',
    ],
}

ENV_PARAMS = {
    'swimmer-gym': { # 2 DoF
    },
    'hopper': { # 3 DoF
    },
    'half-cheetah': { # 6 DoF
    },
    'walker': { # 6 DoF
    },
    'humanoid-gym': { # 17 DoF
        'resume-training': {
            'low_level_policy_path': [
                # 'humanoid-low-level-policy-00-00/itr_4000.pkl',
            ]
        }
    },
    'humanoid-standup-gym': { # 17 DoF
    },
    'mecs': { # 17 DoF
    },
}
VALUE_FUNCTION_PARAMS = {
    'layer_size': M,
}



ALGORITHM_PARAMS_BASE = {
    'lr_c': 1e-4,
    'lr_a': 5e-3,
    'discount': 0.999,
    'target_update_interval': 1,
    'tau': 0.005,
    'reparameterize': REPARAMETERIZE,

    'base_kwargs': {
        'epoch_length': 5000,
        'n_train_repeat': 1,
        'n_initial_exploration_steps': 5000,
        'eval_render': False,
        'eval_n_episodes': 1,
        'eval_deterministic': True,
    }
}

ALGORITHM_PARAMS = {
    'swimmer-gym': { # 2 DoF
        'scale_reward': 25,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    'hopper': { # 3 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 3e3,
        }
    },
    'half-cheetah': { # 6 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 5e3,
            'n_initial_exploration_steps': 10000,
        }
    },
    'walker': { # 6 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 5e3,
        }
    },
    'mecs': { # 17 DoF
        'scale_reward': 200,
        'base_kwargs': {
            'n_epochs': 1e4,
            'n_initial_exploration_steps': 50000,
        }
    },
}

ALGORITHM_PARAMS_DELAYED = {
    'swimmer-gym': { # 2 DoF
        'scale_reward': 25,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    'hopper': { # 3 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 3e3,
        }
    },
    'half-cheetah': { # 6 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 5e3,
            'n_initial_exploration_steps': 10000,
        }
    },
    'walker': { # 6 DoF
        'scale_reward': 5,
        'base_kwargs': {
            'n_epochs': 5e3,
        }
    },
    'humanoid-gym': { # 17 DoF
        'scale_reward': 20,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
    'humanoid-standup-gym': { # 17 DoF
        'scale_reward': 1,
        'base_kwargs': {
            'n_epochs': 1e4,
        }
    },
}


REPLAY_BUFFER_PARAMS = {
    'max_replay_buffer_size': 1e6,
}

SAMPLER_PARAMS = {
    'max_path_length': 5000,
    'min_pool_size': 5000,
    'batch_size': 256,
}


RUN_PARAMS_BASE = {
    'seed': [1],
    'snapshot_mode': 'gap',
    'snapshot_gap': 5000,
    'sync_pkl': True,
}

RUN_PARAMS = {
    'swimmer-gym': { # 2 DoF
        'snapshot_gap': 50000
    },
    'hopper': { # 3 DoF
        'snapshot_gap': 50000
    },
    'half-cheetah': { # 6 DoF
        'snapshot_gap': 50000
    },
    'walker': { # 6 DoF
        'snapshot_gap': 50000
    },
    'humanoid-gym': { # 21 DoF
        'snapshot_gap': 50000
    },
    'humanoid-standup-gym': {  # 21 DoF
        'snapshot_gap': 50000
    },
    'mecs': { # 21 DoF
        'snapshot_gap': 50000
    },
}



def parse_domain_and_transaction(env_name):

    domain = next(domain for domain in DOMAINS if domain in env_name)
    domain_tasks = TRANSACTIONS[domain]
    transaction = next((task for task in domain_tasks if task in env_name), 'default')

    return domain, transaction

def get_variants(domains, transaction):

    vg = VariantGenerator()
    params = {
        'prefix': '{}/{}'.format(domains, transaction),
        'domain': domains,
        'transaction': transaction,
        'git_sha': get_git_rev(),
        'env_params': ENV_PARAMS[domains].get(transaction, {}),
        'value_fn_params': VALUE_FUNCTION_PARAMS,
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS[domains]
        ),
        'replay_buffer_params': REPLAY_BUFFER_PARAMS,
        'sampler_params': SAMPLER_PARAMS,
        'run_params': deep_update(RUN_PARAMS_BASE, RUN_PARAMS[domains]),
    }

    params = flatten(params, separator='.')

    for key, val in params.items():
        if isinstance(val, list) or callable(val):
            vg.add(key,val)
        else:
            vg.add(key, [val])
    return vg


