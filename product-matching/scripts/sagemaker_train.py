import os
import shutil
import datetime
import glob
from tarfile import TarFile
from argparse import ArgumentParser

import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.pytorch import PyTorch

import wandb
from dotenv import load_dotenv; load_dotenv()


ml_instances = [
    "ml.g4dn.xlarge",  # t4 gpu (4 vcores) ~0.2$
    "ml.g4dn.2xlarge",  # t4 gpu (8 vcores) ~0.25$
    "ml.g4dn.4xlarge",  # t4 gpu (16 vcores) ~0.4$
    "ml.p3.2xlarge",  # v100 gpu (8 vcores) ~1$
    "ml.p2.xlarge",
    "ml.m5.4xlarge",
    "ml.m4.16xlarge",
    "ml.p4d.24xlarge",
    "ml.c5n.xlarge",
    "ml.p3.16xlarge",
    "ml.m5.large",
    "ml.p2.16xlarge",
    "ml.c4.2xlarge",
    "ml.c5.2xlarge",
    "ml.c4.4xlarge",
    "ml.c5.4xlarge",
    "ml.c5n.18xlarge",
    "ml.g4dn.12xlarge",
    "ml.c4.8xlarge",
    "ml.c5.9xlarge",
    "ml.c5.xlarge",
    "ml.g4dn.16xlarge",
    "ml.c4.xlarge",
    "ml.g4dn.8xlarge",
    "ml.c5n.2xlarge",
    "ml.c5n.4xlarge",
    "ml.c5.18xlarge",
    "ml.p3dn.24xlarge",
    "ml.m5.xlarge",
    "ml.m4.10xlarge",
    "ml.c5n.9xlarge",
    "ml.m5.12xlarge",
    "ml.m4.xlarge",
    "ml.m5.24xlarge",
    "ml.m4.2xlarge",
    "ml.p2.8xlarge",
    "ml.m5.2xlarge",
    "ml.p3.8xlarge",
    "ml.m4.4xlarge",
]

launch_args = ["python3", "run.py", "-m"]

launch_code = f"""
import os
import sys

if __name__ == "__main__":
    launch_args = {launch_args}
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        launch_rgs.append(args[i].replace("--", "") + "=" + args[i+1])
    os.system(" ".join(launch_args))
"""[1:]

def build_tarball():
    with TarFile("source.tar.gz", "w") as tar:
        # source dir
        for file in glob.glob("src/**/*.py", recursive=True):
            tar.add(file)
        # configuration yamls
        for file in glob.glob("conf/**/*.py", recursive=True):
            tar.add(file)
        # .env
        tar.add(".env")
        # requirements
        tar.add("requirements.txt")
        # hydra launch
        tar.add("run.py")
        # sagemaker->hydra launch
        with open("sm_launch.py", "w") as f:
            f.write(launch_code)
        tar.add("sm_launch.py")
        os.remove("sm_launch.py")
    return "source.tar.gz"


def get_argument_parser():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="config file")
    parser.add_argument(
        "--instance-type", type=str, default="ml.g4dn.xlarge", choices=ml_instances
    )
    parser.add_argument("--wait", dest="wait", action="store_true")
    parser.add_argument("--spot", type=bool, default=True)
    parser.add_argument("--max_jobs", type=int, default=50)
    parser.add_argument("--max_parallel_jobs", type=int, default=2)
    return parser



def main(args):
    build_tarball()

    # estimator = PyTorch(
    #     entry_point="sm_launch.py",
    #     hyperparameters=launch_args,
    #     role=os.environ["SM_ROLE"],
    #     py_version="py3",
    #     source_dir=args.src_dir,
    #     framework_version="1.8.1",
    #     instance_count=1,
    #     instance_type=args.instance_type,
    #     use_spot_instances=True,
    #     max_wait=86400,
    #     disable_profiler=True,
    #     environment={"WANDB_API_KEY": os.environ["WANDB_API_KEY"]},
    #     volume_size=100,
    # )

    # # TODO: Allow ingesting tuning hyperparams from a yaml
    # hyperparameter_ranges = {
    # 	# 'trainer.accumulate_grad_batches' : IntegerParameter(1, 4),
    # 	# 'optimizer.init_args.lr' : ContinuousParameter(1e-4, 1e-1, scaling_type="Logarithmic"),
    #     'trainer.gradient_clip_val': ContinuousParameter(0.01, 2),
    # 	'data.init_args.batch_size': CategoricalParameter([16, 32, 64])
    # 	# 'model.arcface_m' : ContinuousParameter(0.5, 1.2),
    # 	# 'trainer.callbacks.init_args.initial_lr' : ContinuousParameter(1e-6, 1e-2, scaling_type="Logarithmic"),
    # 	# 'trainer.callbacks.init_args.lr_gamma' : ContinuousParameter(0.0, 1.2),
    # }

    # tuner = HyperparameterTuner(estimator,
    #                             objective_metric_name='val/acc',
    #                             objective_type="Maximize",
    # 							metric_definitions=[{
    # 								'Name': 'val/acc',
    # 								'Regex' : r"val\/acc=([0-9\.]+)"
    # 							}],

    #                             hyperparameter_ranges=hyperparameter_ranges,
    # 							strategy="Random",
    #                             max_jobs=args.max_jobs,
    #                             max_parallel_jobs=args.max_parallel_jobs,
    # 							)
    # tuner.fit()
    estimator.fit(wait=args.wait)


if __name__ == "__main__":
    args = get_argument_parser().parse_args()
    main(args)
