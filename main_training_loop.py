try:
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    TORCH_IMPORT_ERROR = None
except ImportError as exc:
    torch = None
    optim = None
    StepLR = None
    TORCH_IMPORT_ERROR = exc

import argparse
import gc
import getpass
import sys
import time


def validate_runtime():
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyTorch could not be imported in this environment. "
            f"Original error: {TORCH_IMPORT_ERROR}"
        )


def prompt_hf_config(args):
    from training_loop_components.hf_sync import HFConfig

    if args.no_hf_prompt or not sys.stdin.isatty():
        return build_hf_config_from_args(args)

    if args.hf_username and args.hf_token and args.hf_collection_name:
        return HFConfig(
            username=args.hf_username,
            token=args.hf_token,
            collection_name=args.hf_collection_name,
            private=args.hf_private,
        )

    username = input("Hugging Face username (leave blank to skip uploads): ").strip()
    if not username:
        return None

    token = getpass.getpass("Hugging Face token: ").strip()
    collection_name = input("Hugging Face collection name: ").strip()
    if not token or not collection_name:
        return None

    return HFConfig(
        username=username,
        token=token,
        collection_name=collection_name,
        private=args.hf_private,
    )


def build_hf_config_from_args(args):
    if not (args.hf_username and args.hf_token and args.hf_collection_name):
        return None

    from training_loop_components.hf_sync import HFConfig

    return HFConfig(
        username=args.hf_username,
        token=args.hf_token,
        collection_name=args.hf_collection_name,
        private=args.hf_private,
    )


def make_status_callback(tui):
    def callback(event, payload):
        if event == "self_play_game_start":
            tui.set_phase("self-play", f"game {payload['game']}/{payload['total_games']}")
        elif event == "self_play_game_done":
            tui.log(
                f"Self-play game {payload['game']}/{payload['total_games']} produced "
                f"{payload['examples']} examples | buffer {payload['replay_buffer_size']}"
            )
        elif event == "training_batch":
            tui.set_phase(
                "training",
                f"epoch {payload['epoch']}/{payload['num_epochs']} batch {payload['batch']}",
            )
        elif event == "training_epoch":
            tui.log(
                f"Epoch {payload['epoch']}/{payload['num_epochs']} "
                f"loss {payload['loss']:.4f} policy {payload['policy_loss']:.4f} value {payload['value_loss']:.4f}"
            )
    return callback


def main():
    validate_runtime()
    args = parse_arguments()

    from neural_network_components.neural_network import NeuralNetwork
    from training_loop_components.evaluate_progress import evaluate_progress
    from training_loop_components.hf_sync import HFCheckpointUploader
    from training_loop_components.save_checkpoint import get_latest_checkpoint, load_checkpoint, save_checkpoint
    from training_loop_components.training_iteration import training_iteration
    from training_loop_components.tui import TrainingTUI

    hf_config = prompt_hf_config(args)
    hf_uploader = HFCheckpointUploader(hf_config) if hf_config is not None else None

    tui = TrainingTUI(enabled=not args.no_tui and not args.quiet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tui.configure(
        total_iterations=args.total_iterations,
        device=str(device),
        args_snapshot={
            "num_games": args.num_games,
            "num_simulations": args.num_simulations,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "checkpoint_frequency": args.checkpoint_frequency,
        },
        hf_repo=hf_config.repo_id if hf_config else "disabled",
        resume_mode=args.resume,
    )

    tui.set_phase("initializing", "loading model")
    neural_net = NeuralNetwork(input_channels=3, device=device)
    optimizer = optim.Adam(neural_net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    start_iteration = 0
    if args.resume:
        checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path:
            tui.log(f"Resuming from {checkpoint_path}")
            checkpoint_info = load_checkpoint(checkpoint_path, neural_net, optimizer, logger=tui.log)
            if checkpoint_info:
                start_iteration = checkpoint_info.get("iteration", 0) + 1
        else:
            tui.log("No checkpoint found, starting from scratch")

    best_rating = 0.0
    training_start_time = time.time()
    last_10_iterations = []
    status_callback = make_status_callback(tui)
    completed_iterations = start_iteration

    for iteration in range(start_iteration, args.total_iterations):
        tui.update_iteration(iteration=iteration, learning_rate=optimizer.param_groups[0]["lr"], best_rating=best_rating)
        tui.set_phase("iteration", f"{iteration + 1}/{args.total_iterations}")

        current_params = {
            "learning_rate": optimizer.param_groups[0]["lr"],
            "num_simulations": args.num_simulations,
            "batch_size": args.batch_size,
        }

        try:
            iteration_metrics = training_iteration(
                neural_net=neural_net,
                optimizer=optimizer,
                nself_play=args.num_games,
                num_simulations=current_params["num_simulations"],
                batch_size=current_params["batch_size"],
                num_epochs=args.num_epochs,
                verbose=False,
                replay_buffer_path="training_data.pkl",
                status_callback=status_callback,
            )
            tui.log("Training iteration completed")
        except Exception as exc:
            iteration_metrics = {"error": str(exc)}
            tui.log(f"[error]: exception: {exc}")

        lr_scheduler.step()
        tui.log(f"Learning rate stepped to {optimizer.param_groups[0]['lr']:.6f}")

        if (iteration + 1) % args.eval_frequency == 0:
            tui.set_phase("evaluation", f"iteration {iteration + 1}")
            try:
                eval_metrics = evaluate_progress(
                    neural_net=neural_net,
                    num_evaluation_games=args.eval_games,
                    num_simulations=max(1, current_params["num_simulations"] // 2),
                    verbose=False,
                )
                iteration_metrics["evaluation"] = eval_metrics
                current_rating = eval_metrics.get("overall_rating", 0.0)
                if current_rating > best_rating:
                    best_rating = current_rating
                    tui.log(f"New best rating {best_rating:.2f}")
                else:
                    tui.log(f"Evaluation rating {current_rating:.2f}")
                tui.update_iteration(iteration + 1, last_eval=current_rating, best_rating=best_rating)
            except Exception as exc:
                iteration_metrics["evaluation_error"] = str(exc)
                tui.log(f"[error]: exception: {exc}")

        if (iteration + 1) % args.checkpoint_frequency == 0:
            tui.set_phase("checkpoint", f"iteration {iteration + 1}")
            checkpoint_path = save_checkpoint(
                neural_net=neural_net,
                optimizer=optimizer,
                iteration=iteration + 1,
                metrics=iteration_metrics,
                checkpoint_dir=args.checkpoint_dir,
                save_frequency=1,
                logger=tui.log,
            )
            if checkpoint_path and hf_uploader is not None:
                try:
                    repo_path = hf_uploader.upload_checkpoint(checkpoint_path, final=False)
                    tui.log(f"Uploaded checkpoint to {hf_config.repo_id}:{repo_path}")
                except Exception as exc:
                    tui.log(f"[error]: exception: {exc}")

        if (iteration + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        final_loss = iteration_metrics.get("final_loss")
        tui.update_iteration(
            iteration=iteration + 1,
            learning_rate=current_params["learning_rate"],
            last_loss=final_loss,
            best_rating=best_rating,
        )
        completed_iterations = iteration + 1

        last_10_iterations.append(iteration_metrics)
        if len(last_10_iterations) > 10:
            last_10_iterations.pop(0)

        if iteration > 50:
            recent_losses = [metrics.get("final_loss", float("inf")) for metrics in last_10_iterations]
            if len(recent_losses) >= 10 and all(loss < 0.01 for loss in recent_losses):
                tui.log("Early stop: converged on low loss window")
                break

    total_training_time = time.time() - training_start_time
    tui.set_phase("finalizing", f"{total_training_time / 3600:.2f}h total")

    final_checkpoint = save_checkpoint(
        neural_net=neural_net,
        optimizer=optimizer,
        iteration=max(1, completed_iterations),
        metrics={"final_training": True, "best_rating": best_rating},
        checkpoint_dir=args.checkpoint_dir,
        save_frequency=1,
        logger=tui.log,
    )
    if final_checkpoint and hf_uploader is not None:
        try:
            repo_path = hf_uploader.upload_checkpoint(final_checkpoint, final=True)
            tui.log(f"Uploaded final checkpoint to {hf_config.repo_id}:{repo_path}")
        except Exception as exc:
            tui.log(f"[error]: exception: {exc}")

    tui.finish(
        f"Training completed in {total_training_time / 3600:.2f}h | "
        f"iterations {completed_iterations} | best rating {best_rating:.2f}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Connect4 RL Agent")
    parser.add_argument("--total-iterations", type=int, default=1000, help="Total number of training iterations")
    parser.add_argument("--num-games", type=int, default=10, help="Number of self-play games per iteration")
    parser.add_argument("--num-simulations", type=int, default=100, help="Number of MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs per iteration")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--eval-frequency", type=int, default=10, help="Evaluate progress every N iterations")
    parser.add_argument("--eval-games", type=int, default=20, help="Number of games for evaluation")
    parser.add_argument("--checkpoint-frequency", type=int, default=25, help="Save checkpoint every N iterations")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    parser.add_argument("--gradient-accumulation", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--no-tui", action="store_true", help="Disable the ANSI terminal UI")
    parser.add_argument("--hf-username", type=str, default="", help="Hugging Face username for checkpoint uploads")
    parser.add_argument("--hf-token", type=str, default="", help="Hugging Face token for checkpoint uploads")
    parser.add_argument("--hf-collection-name", type=str, default="", help="Hugging Face repo name for checkpoint uploads")
    parser.add_argument("--hf-private", action="store_true", help="Create the Hugging Face repo as private")
    parser.add_argument("--no-hf-prompt", action="store_true", help="Disable the interactive Hugging Face prompt")
    return parser.parse_args()


if __name__ == "__main__":
    main()
