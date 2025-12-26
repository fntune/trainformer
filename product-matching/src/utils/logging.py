import os
import time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageOps, ImageFont, ImageDraw

from tqdm import tqdm
from loguru import logger


def make_image_grid(
    image_grid: List[List[str]],
    targets: List[List[bool]],
    dists: List[List[float]],
    image_size: int = 300,
):
    t0 = time.perf_counter()
    rows = len(image_grid)
    cols = len(image_grid[0])

    grid = Image.new("RGBA", size=(cols * image_size, rows * image_size))
    grid_w, grid_h = grid.size

    # font = ImageFont.truetype("sans-serif.ttf", 16)
    logger.debug(f"creating grid..")
    for i, row in tqdm(enumerate(image_grid), total=len(image_grid)):
        for j, img in enumerate(row):
            offset_x, offset_y = j * image_size, i * image_size
            img = (
                Image.open(img)
                .convert("RGBA")
                .resize((image_size - 20, image_size - 20))
            )
            img = ImageOps.expand(
                img,
                border=10,
                fill="black" if j == 0 else ("blue" if targets[i][j] else "red"),
            )
            draw = ImageDraw.Draw(img)
            draw.text(
                (img.size[0] // 2, 30),
                f"dist : {dists[i][j]:.3f}",
                fill="black",
                align="left",
            )  # , font=font)
            grid.paste(img, box=(offset_x, offset_y))
    t1 = time.perf_counter()
    logger.info(f"grid shape : {grid.size} creation took {t1 - t0:.2f} seconds")
    return grid


def log_retrieval_images(
    NN: np.ndarray,
    D: np.ndarray,
    targets: np.ndarray,
    images: np.ndarray,
    epoch: int,
    n_samples: int = 50,
    k: int = 20,
):
    idxs = np.random.randint(0, len(NN) - 1, n_samples)
    NN, D = NN[idxs, :k], D[idxs, :k]
    dists = D.tolist()
    targets = [
        [imid in target for imid in imids] for imids, target in zip(NN, targets[idxs])
    ]
    image_grid = [images[row] for row in NN]
    out_path = Path("retrievals")
    out_path.mkdir(parents=True, exist_ok=True)
    grid = make_image_grid(image_grid, targets, dists)
    grid.save(out_path / f"{epoch}.png")
