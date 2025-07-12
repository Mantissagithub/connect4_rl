# Training Loop Components for Connect4 RL Agent

from .training_iteration import training_iteration
from .network_training import network_training
from .save_checkpoint import (
    save_checkpoint, 
    load_checkpoint, 
    cleanup_old_checkpoints, 
    get_latest_checkpoint, 
    save_best_model
)
from .evaluate_progress import (
    evaluate_progress, 
    evaluate_self_play, 
    evaluate_against_random, 
    evaluate_against_baselines, 
    evaluate_head_to_head, 
    calculate_overall_rating, 
    print_evaluation_summary
)
from .adjust_hyperparameters import (
    adjust_hyperparameters, 
    HyperparameterScheduler, 
    create_default_scheduler,
    apply_learning_rate_schedule,
    apply_exploration_schedule,
    apply_batch_size_schedule,
    apply_simulation_schedule,
    apply_adaptive_adjustments
)

__all__ = [
    # Main training loop functions
    'training_iteration',
    'network_training',
    
    # Checkpoint management
    'save_checkpoint',
    'load_checkpoint',
    'cleanup_old_checkpoints',
    'get_latest_checkpoint',
    'save_best_model',
    
    # Progress evaluation
    'evaluate_progress',
    'evaluate_self_play',
    'evaluate_against_random',
    'evaluate_against_baselines',
    'evaluate_head_to_head',
    'calculate_overall_rating',
    'print_evaluation_summary',
    
    # Hyperparameter adjustment
    'adjust_hyperparameters',
    'HyperparameterScheduler',
    'create_default_scheduler',
    'apply_learning_rate_schedule',
    'apply_exploration_schedule',
    'apply_batch_size_schedule',
    'apply_simulation_schedule',
    'apply_adaptive_adjustments',
]