import rich.tree
import rich.syntax

from typing import Sequence
from omegaconf import DictConfig, OmegaConf


def print_config(
    config: DictConfig,
    fields_to_ignore: Sequence[str] = ("logging",),
    resolve: bool = True,
) -> None:
    """
    Prints a omegaconf config in a pretty tree format using rich library.
    if the color scheme can be modified using `cfg.loggging.rich_style`
    """
    style = config.logging.rich_style
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    fields_to_ignore = set(fields_to_ignore)
    for field in config.keys():
        if field not in fields_to_ignore:
            branch = tree.add(field, style=style, guide_style=style)
            config_section = config.get(field)
            branch_content = str(config_section)
            if isinstance(config_section, DictConfig):
                branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
            branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)
