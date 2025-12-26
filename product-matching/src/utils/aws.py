import os
import s3fs
from tqdm.contrib.concurrent import thread_map

from loguru import logger
from typing import List


def download_images_from_s3(
    images: List[str], root_dir: str = "/data/", threads: int = 100
):
    """
    Downloads images from s3 to a local directory.

    Parameters:
            images: list of s3 path URIs to download.
            root_dir: local directory to download images to.
            threads: Number of threads to use for downloading.
    """

    def download_file(s3_path):
        s3 = s3fs.S3FileSystem()
        out_path = os.path.join(root_dir, s3_path.replace("s3://", ""))
        if not os.path.isfile(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                s3.download(s3_path, out_path)
            except Exception as e:
                print(f"Failed to download {s3_path} : to {out_path} because {e}")
        return out_path

    print(f"Downloading {images[0]} and all {len(images)} images to : {root_dir}")
    downloaded_images = thread_map(download_file, images, max_workers=threads)
    return downloaded_images
