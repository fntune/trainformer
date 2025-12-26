import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def format_hyperparams(cfg: DictConfig, ignore_groups: list = ["xp"]):
    primitive_cfg = OmegaConf.to_container(cfg, resolve=True)
    for key in ignore_groups:
        primitive_cfg.pop(key, None)
    return primitive_cfg


class CommandError(Exception):
    pass


def run_command(command, **kwargs):
    proc = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs
    )
    if proc.returncode:
        command_str = " ".join(shlex.quote(c) for c in command)
        raise CommandError(
            f"Command {command_str} failed ({proc.returncode}): \n"
            + proc.stdout.decode()
        )
    return proc.stdout.decode().strip()


def get_git_root():
    return Path(run_command(["git", "rev-parse", "--show-toplevel"])).resolve()


def get_git_commit(repo: Path = Path(".")):
    return run_command(["git", "log", "-1", "--format=%H"], cwd=repo)


def get_git_branch(repo: Path = Path(".")):
    return run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)


def get_git_status(repo: Path = Path(".")):
    return run_command(["git", "status", "-s"], cwd=repo)


def get_git_diff(repo: Path = Path(".")):
    return run_command(["git", "diff", "--shortstat"], cwd=repo)


def get_git_version(repo: Path = Path(".")):
    commit = get_git_commit(repo)
    branch = get_git_branch(repo)
    return f"{branch}@{commit}"


def get_xp_sig(cfg: DictConfig, truncate_hash: int = 8):
    primitive_cfg = format_hyperparams(
        cfg, ignore_groups=cfg.xp.get("ignore_groups", [])
    )
    return hashlib.sha1(
        (get_git_version() + json.dumps(primitive_cfg, sort_keys=True)).encode()
    ).hexdigest()[:truncate_hash]
