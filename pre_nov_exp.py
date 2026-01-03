import ray
import os
import subprocess
import sys
from datetime import datetime

@ray.remote(num_gpus=1)
def run_trial(args):
    os.environ["WANDB_PROJECT"] = f"{args[12]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    cmd = [sys.executable, "dreamer.py"] + args
    subprocess.run(cmd, check=True)

task = "rwc_cartpole_realworld_balance"
# seeds = [0, 10, 20]
seeds = [0]
steps = "1e5"
use_testing = True

configs = [
    ("dreamer", None),
    # ("mg-dreamer", "mg_lambda=0.8"),
    # ("mg-dreamer", "mg_lambda=1.0"),
    # ("mg-dreamer", "mg_lambda=1.2"),
    # ("mvpi-dreamer", "mvpi_lambda=0.2"),
    # ("mvpi-dreamer", "mvpi_lambda=0.4"),
    # ("mvpi-dreamer", "mvpi_lambda=0.6"),
    ("exp-dreamer", "beta=-0.001"),
    # ("exp-dreamer", "beta=-0.005"),
    # ("exp-dreamer", "beta=-0.01"),
    ("exp-dreamer", "beta=0.001"),
]

trials = []
for seed in seeds:
    for algorithm, hyper in configs:
        if hyper is not None:
            hp_name = hyper.replace("=", "_")
            extra_args = [f"--{hyper}"]
        else:
            hp_name = "none"
            extra_args = []

        logdir = f"./logdir/{task}/{algorithm}/{hp_name}/{seed}"

        config_names = ["rwc"]
        if use_testing:
            config_names.append("testing")

        args = [
            "--configs", *config_names,
            "--output", "wandb",
            "--steps", steps,
            "--seed", str(seed),
            "--algorithm", algorithm,
            "--task", task,
            "--logdir", logdir,
        ] + extra_args

        trials.append(args)

futures = [run_trial.remote(args) for args in trials]
ray.get(futures)