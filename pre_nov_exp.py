import ray
import os
import subprocess
import sys
from datetime import datetime
from uuid import uuid4

@ray.remote(num_gpus=1/3)
def run_trial(args):
    cmd = [sys.executable, "dreamer.py"] + args
    subprocess.run(cmd, check=True)

task = "rwc_quadruped_realworld_walk"
seeds = [0, 10, 20]
steps = "5e5"
use_testing = False
identifier = str(uuid4())

configs = [
    ("dreamer", None, False),
    # ("mg-dreamer", "mg_lambda=0.8"),
    # ("mg-dreamer", "mg_lambda=1.0"),
    # ("mg-dreamer", "mg_lambda=1.2"),
    # ("mvpi-dreamer", "mvpi_lambda=0.2"),
    # ("mvpi-dreamer", "mvpi_lambda=0.4"),
    # ("mvpi-dreamer", "mvpi_lambda=0.6"),
    ("exp", "beta=-0.001", False),
    ("exp", "beta=0.001", False),
    # ("exp-dreamer", "beta=-0.005"),
    # ("exp-dreamer", "beta=-0.01"),
    # ("exp", "beta=0.001", False),
    ("exp", "beta=0.001", True),
]

trials = []
for seed in seeds:
    for algorithm, hyper, switch in configs:
        if hyper is not None:
            hp_name = hyper.replace("=", "_")
            extra_args = [f"--{hyper}"]
        else:
            hp_name = "none"
            extra_args = []

        logdir = f"./logdir/{task}/{algorithm}/{hp_name}/{switch}/{seed}"

        config_names = ["rwc"]
        if use_testing:
            config_names.append("testing")

        args = [
            "--configs", *config_names,
            "--output", "wandb",
            "--project", f"{task}-{identifier}",
            "--steps", steps,
            "--switch", str(switch),
            "--seed", str(seed),
            "--algorithm", algorithm,
            "--logdir", logdir,
        ] + extra_args + ["--task", task]

        trials.append(args)

futures = [run_trial.remote(args) for args in trials]
ray.get(futures)