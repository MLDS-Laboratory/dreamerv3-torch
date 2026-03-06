import ray
import os
import subprocess

@ray.remote(num_gpus=1/3, resources={"gpu_slot": 1})
def run_trial(args):
    # os.environ["WANDB_PROJECT"] = args[11]
    cmd = ["python", "evaluate.py"] + args
    subprocess.run(cmd, check=True)

task = "rwc_quadruped_realworld_walk"
seeds = [0]

configs = [
    ("dreamer", None),
    ("mg-dreamer", "mg_lambda=0.8"),
    ("mvpi-dreamer", "mvpi_lambda=0.2"),
    ("mvpi-dreamer", "mvpi_lambda=0.4"),
    ("mvpi-dreamer", "mvpi_lambda=0.6"),
    ("exp-dreamer", "beta=-0.01"),
]

novelties = ["rwc_quadruped_perturb_novelty", "rwc_quadruped_damping_novelty", "rwc_quadruped_noise_novelty"]


trials = []
for seed in seeds:
    for algorithm, hyper in configs:
        for novelty in novelties:
            if hyper is not None:
                hp_name = hyper.replace("=", "_")
                extra_args = [f"--{hyper}"]
            else:
                hp_name = "none"
                extra_args = []

            hp_name = hp_name.split('=')[-1]
            logdir = f"./logdir/{task}/{algorithm}/{hp_name}/{seed}"

            args = [
                "--configs", "rwc",
                "--output", "wandb",
                "--seed", str(seed),
                "--algorithm", algorithm,
                "--task", novelty,
                "--logdir", logdir,
            ] + extra_args

            trials.append(args)

futures = [run_trial.remote(args) for args in trials]
ray.get(futures)