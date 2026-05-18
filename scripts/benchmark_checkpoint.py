import argparse
import contextlib
import io
import json
import os
import random
import site
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _bootstrap_torch_runtime():
    if os.environ.get("CONNECT4_RL_TORCH_BOOTSTRAPPED") == "1":
        return

    packages = site.getsitepackages()
    if not packages:
        return

    nvidia_base = Path(packages[0]) / "nvidia"
    if not nvidia_base.exists():
        return

    lib_dirs = sorted({str(path.parent) for path in nvidia_base.rglob("lib/*.so*")})
    if not lib_dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [part for part in current.split(":") if part]
    missing = [lib_dir for lib_dir in lib_dirs if lib_dir not in current_parts]
    if not missing:
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(missing + current_parts)
    os.environ["CONNECT4_RL_TORCH_BOOTSTRAPPED"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


def get_latest_checkpoint(checkpoint_dir: str) -> Path:
    checkpoint_path = Path(checkpoint_dir)
    candidates = sorted(checkpoint_path.glob("checkpoint_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    return candidates[0]


def run_benchmark(checkpoint: str, eval_games: int, self_play_simulations: int, seed: int) -> dict:
    import numpy as np
    import torch

    from neural_network_components.load_model import load_model_for_inference
    from training_loop_components.evaluate_progress import evaluate_progress

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = load_model_for_inference(checkpoint)
    if model is None:
        raise RuntimeError(f"Failed to load checkpoint: {checkpoint}")

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        results = evaluate_progress(
            neural_net=model,
            num_evaluation_games=eval_games,
            num_simulations=self_play_simulations,
            verbose=False,
        )

    self_play = results.get("self_play_results", {})
    random_results = results.get("baseline_results", {}).get("random", {})
    score = results.get("overall_rating", 0.0)

    return {
        "checkpoint": checkpoint,
        "benchmark_date": time.strftime("%Y-%m-%d"),
        "seed": seed,
        "evaluation_games_per_stage": eval_games,
        "self_play_simulations": self_play_simulations,
        "overall_rating": score,
        "self_play": self_play,
        "vs_random": random_results,
        "notes": [
            "This repository uses a custom internal score, not a true Elo ladder.",
            "Higher score means better balance in self-play and stronger results versus a random opponent.",
        ],
    }


def main():
    _bootstrap_torch_runtime()

    parser = argparse.ArgumentParser(description="Benchmark a Connect4 RL checkpoint with the repo's internal evaluation.")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path. Defaults to the newest file in checkpoints/.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to scan for the latest checkpoint.")
    parser.add_argument("--eval-games", type=int, default=12, help="Number of games for self-play and random-opponent evaluation.")
    parser.add_argument("--self-play-simulations", type=int, default=24, help="MCTS simulations per move during self-play evaluation.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else get_latest_checkpoint(args.checkpoint_dir)
    results = run_benchmark(
        checkpoint=str(checkpoint),
        eval_games=args.eval_games,
        self_play_simulations=args.self_play_simulations,
        seed=args.seed,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
