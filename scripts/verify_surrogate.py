import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a trained surrogate by running classical evaluation scenarios"
    )
    parser.add_argument("--checkpoint", type=str, default="/tmp/ecsfm/surrogate_model.eqx", help="Path to .eqx checkpoint")
    parser.add_argument("--dataset", "--data-path", dest="data_path", type=str, default="/tmp/ecsfm/dataset_massive", help="Dataset path used to rebuild normalizers")
    parser.add_argument("--output-dir", type=str, default="/tmp/ecsfm", help="Directory for evaluation plots and scorecard")
    parser.add_argument("--hidden-size", type=int, default=128, help="Model hidden size used during training")
    parser.add_argument("--depth", type=int, default=3, help="Model depth used during training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split for normalizer recreation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cmd = [
        sys.executable,
        "-m",
        "ecsfm.fm.eval_classical",
        "--checkpoint",
        args.checkpoint,
        "--dataset",
        args.data_path,
        "--output-dir",
        args.output_dir,
        "--hidden-size",
        str(args.hidden_size),
        "--depth",
        str(args.depth),
        "--seed",
        str(args.seed),
        "--val-split",
        str(args.val_split),
    ]

    print("Running surrogate verification via classical scenario evaluation...")
    subprocess.run(cmd, check=True)
    print("Verification complete.")


if __name__ == "__main__":
    main()
