#%%
import os
import boto3
from argparse import ArgumentParser

import sys

sys.path.append("src")
import torch
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from common.utils import get_best_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument(
        "--modality", type=str, default="image", choices=["image", "text"]
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--bucket", type=str, default="aisle3-ml-models")
    parser.add_argument("--out-prefix", type=str, default="")
    return parser.parse_args()


def main(args):
    model_path = get_best_model(args.run)
    print(f"downloaded model: {model_path}")

    model = None
    trace_input = None
    if args.modality == "image":
        model = ImageEncoder.load_from_checkpoint(model_path).eval()
        trace_input = torch.randn(1, 3, args.image_size, args.image_size)
    elif args.modality == "text":
        model = TextEncoder.load_from_checkpoint(model_path).eval()
        # max_length = getattr(model, "max_length", args.max_length)
        trace_input = (
            torch.randint(1, 100, (1, args.max_length)).int(),
            torch.ones(1, args.max_length).int(),
        )
    print(model)
    print("trace input:")
    print(trace_input)

    print("tracing model graph..")
    with torch.no_grad():
        trace = torch.jit.trace(model, trace_input)
    print(trace)
    print(trace.code)
    print("saving traced model..")
    torch.jit.save(trace, "model.pth")

    s3 = boto3.client("s3")
    print("uploading model..")
    out_prefix = args.out_prefix
    if out_prefix == "":
        out_prefix = "/".join(args.run.split("/")[1:])
    print(f"uploading model.pth to s3://{args.bucket}/{out_prefix}/model.pth")
    s3.upload_file("model.pth", args.bucket, "/".join((out_prefix, "model.pth")))


if __name__ == "__main__":
    args = parse_args()
    main(args)
