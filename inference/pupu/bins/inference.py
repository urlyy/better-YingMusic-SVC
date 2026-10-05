import argparse
import torch

from models.afgen.afgen_inference import AFGenInference
from utils.util import load_config


def build_inference(args, cfg):
    supported_inference = {"PupuVocoder": AFGenInference, "PupuCodec": AFGenInference}

    inference_class = supported_inference[cfg.model_type]
    return inference_class(args, cfg)


def cuda_relevant():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON/YAML file for configurations.",
    )
    parser.add_argument(
        "--infer_datasets",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--vocoder_dir",
        type=str,
        required=True,
        help="Vocoder checkpoint directory. Searching behavior is the same as "
        "the acoustics one.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result",
        help="Output directory. Default: ./result",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="warning",
        help="Logging level. Default: warning",
    )
    return parser


def main():
    # Parse arguments
    args = build_parser().parse_args()

    # Parse config
    cfg = load_config(args.config)

    # CUDA settings
    cuda_relevant()

    # Build inference
    trainer = build_inference(args, cfg)

    # Run inference
    trainer.inference()


if __name__ == "__main__":
    main()
