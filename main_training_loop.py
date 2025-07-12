import torch
import torch.optim as optim
import time
import os
import argparse
from typing import Dict, Any
import gc
from torch.optim.lr_scheduler import StepLR

def main():
    args = parse_arguments()
    print("="*60)
    print("CONNECT4 RL AGENT TRAINING")
    print("="*60)

    from neural_network_components.neural_network import NeuralNetwork
    from training_loop_components.training_iteration import training_iteration
    from training_loop_components.save_checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
    from training_loop_components.evaluate_progress import evaluate_progress
    from training_loop_components.adjust_hyperparameters import create_default_scheduler

    print("Initializing neural network...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    neural_net = NeuralNetwork(
        input_channels=3, # 3 channels: empty, player1, player2
        device=device
    )

    initial_lr = args.learning_rate
    optimizer = optim.Adam(neural_net.parameters(), lr=initial_lr, weight_decay=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    scheduler = create_default_scheduler()
    scheduler.current_params.update({
        'learning_rate': initial_lr,
        'total_iterations': args.total_iterations,
        'batch_size': args.batch_size,
        'num_simulations': args.num_simulations
    })

    #resume from a checkpoint, if some bs expects/requests, he must request mf
    start_iteration = 0
    if args.resume:
        checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint_info = load_checkpoint(checkpoint_path, neural_net, optimizer)
            start_iteration = checkpoint_info.get('iteration', 0) + 1
        else:
            print("No checkpoint found, starting from scratch")

    print(f"\nStarting training from iteration {start_iteration}")
    print(f"Target iterations: {args.total_iterations}")
    print(f"Initial learning rate: {initial_lr}")

    best_rating = 0.0
    training_start_time = time.time()
    last_10_iterations = []

    for iteration in range(start_iteration, args.total_iterations):
        iteration_start_time = time.time()
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration + 1}/{args.total_iterations}")
        print(f"{'='*50}")

        current_params = {
            'learning_rate': optimizer.param_groups[0]['lr'],
            'num_simulations': args.num_simulations,
            'batch_size': args.batch_size
        }

        try:
            iteration_metrics = training_iteration(
                neural_net=neural_net,
                optimizer=optimizer,
                nself_play=args.num_games,
                num_simulations=current_params['num_simulations'],
                batch_size=current_params['batch_size'],
                num_epochs=args.num_epochs,
                verbose=args.verbose
            )

            print(f"Training iteration completed successfully")

            if 'final_loss' not in iteration_metrics or iteration_metrics['final_loss'] is None:
                print("WARNING: No loss computed - check training_iteration function")
                continue

            final_loss = iteration_metrics.get('final_loss', float('inf'))
            if final_loss > 100 or final_loss < 0:
                print(f"WARNING: Unusual loss value: {final_loss}")

        except Exception as e:
            print(f"Error during training iteration: {e}")
            iteration_metrics = {'error': str(e)}

        lr_scheduler.step()
        print(f"Learning rate after scheduler step: {optimizer.param_groups[0]['lr']:.6f}")

        #evauating the progress periodically, to not overload the system
        # and to allow for better monitoring of the training process
        if (iteration + 1) % args.eval_frequency == 0:
            print(f"\nEvaluating progress...")
            try:
                eval_metrics = evaluate_progress(
                    neural_net=neural_net,
                    num_evaluation_games=args.eval_games,
                    num_simulations=current_params['num_simulations'] // 2,
                    verbose=args.verbose
                )

                current_rating = eval_metrics.get('overall_rating', 0.0)
                if current_rating > best_rating:
                    best_rating = current_rating
                    print(f"New best rating achieved: {best_rating:.2f}")

                iteration_metrics.update({'evaluation': eval_metrics})

            except Exception as e:
                print(f"Error during evaluation: {e}")
                iteration_metrics['evaluation_error'] = str(e)

        #save the current checkpoint periodically
        if (iteration + 1) % args.checkpoint_frequency == 0:
            try:
                checkpoint_path = save_checkpoint(
                    neural_net=neural_net,
                    optimizer=optimizer,
                    iteration=iteration + 1,
                    metrics=iteration_metrics,
                    checkpoint_dir=args.checkpoint_dir,
                    save_frequency=1 #saving every call since we're controlling frequency here
                )

                if checkpoint_path:
                    print(f"Checkpoint saved: {checkpoint_path}")

            except Exception as e:
                print(f"Error saving checkpoint: {e}")

        if (iteration + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        iteration_time = time.time() - iteration_start_time
        total_time = time.time() - training_start_time

        print(f"\nIteration {iteration + 1} Summary:")
        print(f"  Time: {iteration_time:.2f}s (Total: {total_time/3600:.2f}h)")
        print(f"  Learning rate: {current_params['learning_rate']:.2e}")
        print(f"  Final loss: {iteration_metrics.get('final_loss', 'N/A')}")

        if 'evaluation' in iteration_metrics:
            rating = iteration_metrics['evaluation'].get('overall_rating', 0)
            print(f"  Current rating: {rating:.2f} (Best: {best_rating:.2f})")

        #stroing iteration checkpoints for checking if its converging
        last_10_iterations.append(iteration_metrics)
        if len(last_10_iterations) > 10:
            last_10_iterations.pop(0)

        #eraly stopping -> converging
        if iteration > 50:
            recent_losses = [metrics.get('final_loss', float('inf')) 
                            for metrics in last_10_iterations]
            if len(recent_losses) >= 10 and all(loss < 0.01 for loss in recent_losses):
                print("Training converged based on consistent low losses")
                break

    total_training_time = time.time() - training_start_time

    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Total training time: {total_training_time/3600:.2f} hours")
    print(f"Iterations completed: {iteration + 1}")
    print(f"Best rating achieved: {best_rating:.2f}")

    try:
        final_checkpoint = save_checkpoint(
            neural_net=neural_net,
            optimizer=optimizer,
            iteration=iteration + 1,
            metrics={'final_training': True, 'best_rating': best_rating},
            checkpoint_dir=args.checkpoint_dir,
            save_frequency=1
        )

        print(f"Final model saved: {final_checkpoint}")

    except Exception as e:
        print(f"Error saving final model: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Connect4 RL Agent')

    #just for fun😅, kep this as args, if you want just play around with it
    parser.add_argument('--total-iterations', type=int, default=1000,
                        help='Total number of training iterations')
    parser.add_argument('--num-games', type=int, default=10,
                        help='Number of self-play games per iteration')
    parser.add_argument('--num-simulations', type=int, default=100,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--batch-size', type=int, default=64, #increased from 32 to 64 according to my gpu cap
                        help='Training batch size')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of training epochs per iteration')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--eval-frequency', type=int, default=10,
                        help='Evaluate progress every N iterations')
    parser.add_argument('--eval-games', type=int, default=20,
                        help='Number of games for evaluation')
    parser.add_argument('--checkpoint-frequency', type=int, default=25,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed progress information')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    parser.add_argument('--gradient-accumulation', type=int, default=2,
                        help='Gradient accumulation steps')

    args = parser.parse_args()

    if args.quiet:
        args.verbose = False

    return args

if __name__ == "__main__":
    main()
