# calculate the number of trainable/total parameters in a model
# this script does it like pytorch lightning which was a nice handy feature

import json
import math
import numpy as np
import pandas as pd

import torch

from rich.console import Console
from rich.table import Table


PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]

def get_human_readable_count(number: int) -> str:
    """Abbreviates an integer number with K, M, B, T for thousands, millions, billions and trillions, respectively.
    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number
    Return:
        A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10 ** shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    return f"{number:,.1f} {labels[index]}"


def df_to_rich(df):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    for column in df.columns:
        table.add_column(column)
    for i, row in df.iterrows():
        table.add_row(*list(map(str, row)))
    console.print(table)


def print_model_summary(model: torch.nn.Module(), max_depth: int = 1):
    summary = pd.DataFrame(
        [
            (name, param.numel(), int(param.requires_grad))
            for name, param in model.named_parameters()
        ],
        columns=["Layer", "Total", "Trainable"],
    )
    summary.Trainable = summary.Total * summary.Trainable
    summary.Layer = summary.Layer.str.split(".").apply(
        lambda x: ".".join(x[:max_depth])
    )
    summary = summary.groupby(["Layer"], sort=False, as_index=False).sum()
    summary.loc[len(summary)] = (
        "[bold]Total[/]",
        summary.Total.sum(),
        summary.Trainable.sum(),
    )
    summary.Total = summary.Total.apply(get_human_readable_count)
    summary.Trainable = summary.Trainable.apply(get_human_readable_count)
    df_to_rich(summary)

if __name__ == "__main__":
    # a handy place to look at different models
    # run this script with `model_name` string and get a summary of the model
    import timm
    import sys
    model_name = sys.argv[1]
    print(f"model_name: {model_name}")
    model = timm.create_model(model_name, pretrained=False)
    print(model)
    print(json.dumps(model.default_cfg, indent=2))
    print_model_summary(model)