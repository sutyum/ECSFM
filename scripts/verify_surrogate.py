import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a trained surrogate by running classical evaluation scenarios"
    )
    parser.add_argument("--checkpoint", type=str, default="/tmp/ecsfm/surrogate_model.eqx", help="Path to .eqx checkpoint")
    parser.add_argument("--dataset", "--data-path", dest="data_path", type=str, default=None, help="Dataset path used only if normalizer artifacts are missing")
    parser.add_argument("--normalizers", type=str, default=None, help="Path to normalizers NPZ (optional)")
    parser.add_argument("--meta", type=str, default=None, help="Path to model metadata JSON (optional)")
    parser.add_argument("--output-dir", type=str, default="/tmp/ecsfm", help="Directory for evaluation plots and scorecard")
    parser.add_argument("--hidden-size", type=int, default=None, help="Model hidden size override")
    parser.add_argument("--depth", type=int, default=None, help="Model depth override")
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
        "--output-dir",
        args.output_dir,
        "--seed",
        str(args.seed),
        "--val-split",
        str(args.val_split),
    ]
    if args.data_path is not None:
        cmd.extend(["--dataset", args.data_path])
    if args.normalizers is not None:
        cmd.extend(["--normalizers", args.normalizers])
    if args.meta is not None:
        cmd.extend(["--meta", args.meta])
    if args.hidden_size is not None:
        cmd.extend(["--hidden-size", str(args.hidden_size)])
    if args.depth is not None:
        cmd.extend(["--depth", str(args.depth)])

    print("Running surrogate verification via classical scenario evaluation...")
    subprocess.run(cmd, check=True)
    print("Verification complete.")


if __name__ == "__main__":
    main()
