

import collections
import datetime
import dateutil.tz
import os
import numpy as np
from coma.misc.utils import timestamp
from coma.misc.utils import run_experiment_lite
import uuid


PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))
DEFAULT_LOG_DIR = PROJECT_PATH + "/data"



def run_coma_experiment(main, mode, include_folders=None, log_dir=None,
                       exp_prefix="experiment", exp_name=None, **kwargs):

    if exp_name is None:
        exp_name = timestamp()


    if log_dir is None:
        log_dir = os.path.join(
            DEFAULT_LOG_DIR,
            "local",
            exp_prefix.replace("_", "-"),
            exp_name,
            "iter"+str(kwargs["seed"]))
    else:
        log_dir = os.path.join(log_dir, "iter" + str(kwargs["seed"]))


    if include_folders is None:
        include_folders = list()


    if mode == 'ec2':
        include_folders.append('sac')
        all_symlinks = list()

        for folder in include_folders:
            all_symlinks.append(_create_symlink(folder))


        kwargs.update(added_project_directories=all_symlinks)


    run_experiment_lite(
        stub_method_call=None,
        mode=mode,
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        log_dir=log_dir,
        **kwargs,
    )

def _create_symlink(folder):



    include_path = os.path.join('/tmp/', str(uuid.uuid4()))
    os.makedirs(include_path)

    os.symlink(os.path.join(PROJECT_PATH, folder),
               os.path.join(include_path, folder))

    return include_path