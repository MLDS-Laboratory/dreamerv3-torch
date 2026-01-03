import ray
import os
import subprocess
import shutil
from pathlib import Path

@ray.remote(num_gpus=1/4, resources={"gpu_slot": 1})
def run_trial(args):
    os.environ["WANDB_PROJECT"] = args[11]
    cmd = ["python", "dreamer.py"] + args
    subprocess.run(cmd, check=True)

tasks = ["rwc_walker_friction_novelty", "rwc_walker_noise_novelty"]

base = Path("./logdir/rwc_walker_realworld_walk")
for name in tasks:
    dst = Path("./logdir") / name
    if not dst.exists():
        shutil.copytree(base, dst)

seeds = [0]
steps = "1e6"

configs = [
    ("dreamer", None),
    ("mg-dreamer", "mg_lambda=1.0"),
    ("mvpi-dreamer", "mvpi_lambda=0.4"),
    ("exp-dreamer", "beta=-0.001")
]

trials = []
for task in tasks:
    for seed in seeds:
        for algorithm, hyper in configs:
            if hyper is not None:
                hp_name = hyper.replace("=", "_")
                extra_args = [f"--{hyper}"]
            else:
                hp_name = "none"
                extra_args = []

            logdir = f"./logdir/{task}/{algorithm}/{hp_name}/{seed}"

            args = [
                "--configs", "rwc",
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