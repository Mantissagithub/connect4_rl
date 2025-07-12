import math
from typing import Dict, Any, Tuple

class HyperparameterScheduler: 
    def __init__(self, initial_params: Dict[str, Any]):
        self.initial_params = initial_params.copy()
        self.current_params = initial_params.copy()
        self.history = []
        self.iteration = 0
    
    def update_iteration(self, iteration: int):
        self.iteration = iteration
    
    def get_current_params(self) -> Dict[str, Any]:
        return self.current_params.copy()
    
    def log_metrics(self, metrics: Dict[str, Any]):
        self.history.append({
            'iteration': self.iteration,
            'metrics': metrics.copy(),
            'params': self.current_params.copy()
        })

def adjust_hyperparameters(scheduler: HyperparameterScheduler, metrics: Dict[str, Any], 
                         iteration: int, verbose: bool = True):   
    scheduler.update_iteration(iteration)
    scheduler.log_metrics(metrics)
    
    params = scheduler.get_current_params()
    
    params = apply_learning_rate_schedule(params, iteration, metrics, verbose)
    params = apply_exploration_schedule(params, iteration, metrics, verbose)
    params = apply_batch_size_schedule(params, iteration, metrics, verbose)
    params = apply_simulation_schedule(params, iteration, metrics, verbose)
    params = apply_adaptive_adjustments(params, scheduler.history, verbose)
    
    scheduler.current_params = params
    
    if verbose:
        print_hyperparameter_changes(scheduler.initial_params, params, iteration)
    
    return params

def apply_learning_rate_schedule(params: Dict[str, Any], iteration: int, 
                               metrics: Dict[str, Any], verbose: bool = True):
    initial_lr = params.get('learning_rate', 0.001)
    
    if 'total_iterations' in params:
        total_iterations = params['total_iterations']
        cosine_factor = 0.5 * (1 + math.cos(math.pi * iteration / total_iterations))
        min_lr = initial_lr * 0.01  
        new_lr = min_lr + (initial_lr - min_lr) * cosine_factor
    else:
        decay_rate = params.get('lr_decay_rate', 0.95)
        decay_frequency = params.get('lr_decay_frequency', 50)
        if iteration % decay_frequency == 0 and iteration > 0:
            new_lr = initial_lr * (decay_rate ** (iteration // decay_frequency))
        else:
            new_lr = params.get('learning_rate', initial_lr)
    
    if iteration > 10 and len(metrics) > 0:
        final_loss = metrics.get('final_loss', float('inf'))
        
        #loss is exploding, reducing the learning rate more
        if final_loss > 10.0:
            new_lr *= 0.5
            if verbose:
                print(f"learning rate reduced due to high loss: {new_lr:.2e}")
        
        #loss so small, increase the learning rate more
        elif final_loss < 0.01 and new_lr < initial_lr * 0.1:
            new_lr *= 1.1
            if verbose:
                print(f" Learning rate increased due to low loss: {new_lr:.2e}")
    
    params['learning_rate'] = max(new_lr, 1e-6)  #just like vanishing gradients prob, we dont want lr to become too small
    return params

def apply_exploration_schedule(params: Dict[str, Any], iteration: int, 
                             metrics: Dict[str, Any], verbose: bool = True):
    #exploration -> higher eraly temp
    #exploitation -> lower later temp
    initial_temperature = params.get('initial_temperature', 1.0)
    final_temperature = params.get('final_temperature', 0.1)
    temperature_decay_iterations = params.get('temperature_decay_iterations', 200)
    
    if iteration < temperature_decay_iterations:
        progress = iteration / temperature_decay_iterations
        new_temperature = initial_temperature - (initial_temperature - final_temperature) * progress
    else:
        new_temperature = final_temperature
    
    params['temperature'] = new_temperature
    
    #leanred about dirichlet noise weight scheduling
    initial_noise = params.get('initial_dirichlet_noise', 0.25)
    final_noise = params.get('final_dirichlet_noise', 0.05)
    noise_decay_iterations = params.get('noise_decay_iterations', 300)
    
    if iteration < noise_decay_iterations:
        progress = iteration / noise_decay_iterations
        new_noise = initial_noise - (initial_noise - final_noise) * progress
    else:
        new_noise = final_noise
    
    params['dirichlet_noise_weight'] = new_noise
    
    #for mcts exploration constant, that c_puct
    initial_c_puct = params.get('initial_c_puct', 1.0)
    final_c_puct = params.get('final_c_puct', 2.0)
    c_puct_increase_iterations = params.get('c_puct_increase_iterations', 150)
    
    if iteration < c_puct_increase_iterations:
        progress = iteration / c_puct_increase_iterations
        new_c_puct = initial_c_puct + (final_c_puct - initial_c_puct) * progress
    else:
        new_c_puct = final_c_puct
    
    params['c_puct'] = new_c_puct
    
    return params

def apply_batch_size_schedule(params: Dict[str, Any], iteration: int, 
                            metrics: Dict[str, Any], verbose: bool = True):
    initial_batch_size = params.get('initial_batch_size', 32)
    max_batch_size = params.get('max_batch_size', 128)
    batch_size_increase_frequency = params.get('batch_size_increase_frequency', 100)
    
    if iteration > 0 and iteration % batch_size_increase_frequency == 0:
        current_batch_size = params.get('batch_size', initial_batch_size)
        new_batch_size = min(current_batch_size * 1.5, max_batch_size)
        
        if new_batch_size != current_batch_size:
            params['batch_size'] = int(new_batch_size)
            if verbose:
                print(f"  Batch size increased to {int(new_batch_size)}")
    
    return params

def apply_simulation_schedule(params: Dict[str, Any], iteration: int, 
                            metrics: Dict[str, Any], verbose: bool = True):
    initial_simulations = params.get('initial_simulations', 100)
    max_simulations = params.get('max_simulations', 400)
    simulation_increase_frequency = params.get('simulation_increase_frequency', 75)
    
    #gracefully increasing tho no. of simulations for better learning and move quality
    if iteration > 0 and iteration % simulation_increase_frequency == 0:
        current_simulations = params.get('num_simulations', initial_simulations)
        new_simulations = min(current_simulations + 25, max_simulations)
        
        if new_simulations != current_simulations:
            params['num_simulations'] = new_simulations
            if verbose:
                print(f"  MCTS simulations increased to {new_simulations}")
    
    return params

def apply_adaptive_adjustments(params: Dict[str, Any], history: list, 
                             verbose: bool = True):
    if len(history) < 5:
        return params
    
    recent_losses = [h['metrics'].get('final_loss', float('inf')) for h in history[-5:]]
    recent_losses = [loss for loss in recent_losses if loss != float('inf')]
    
    if len(recent_losses) >= 3:
        loss_std = calculate_std(recent_losses)
        loss_mean = sum(recent_losses) / len(recent_losses)
        
        if loss_std < 0.001 and loss_mean > 0.1:  #low variance over here, bit higher loss, so stagnation, increasing the lr is the solution
            current_lr = params.get('learning_rate', 0.001)
            new_lr = current_lr * 1.2
            params['learning_rate'] = min(new_lr, 0.01)  #this is the max cap, to solve like the problem of exploding gradients
            
            if verbose:
                print(f"Loss stagnation detected, learning rate increased to {new_lr:.2e}")
        
        elif loss_std > loss_mean * 0.5:  #high variance, so if low variance is a stagnation then this is know as oscilation, decrease the learning rate
            current_lr = params.get('learning_rate', 0.001)
            new_lr = current_lr * 0.8
            params['learning_rate'] = new_lr
            
            if verbose:
                print(f"  Loss oscillation detected, learning rate reduced to {new_lr:.2e}")
    
    return params

def calculate_std(values: list):
    if len(values) < 2:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def print_hyperparameter_changes(initial_params: Dict[str, Any], current_params: Dict[str, Any], 
                                iteration: int):
    #summary of all the hyperparameter chnages
    print(f"\nHyperparameter Status (Iteration {iteration}):")
    print("-" * 40)
    
    key_params = ['learning_rate', 'temperature', 'dirichlet_noise_weight', 
                  'c_puct', 'batch_size', 'num_simulations']
    
    for param in key_params:
        if param in current_params:
            initial_val = initial_params.get(param, 'N/A')
            current_val = current_params[param]
            
            if isinstance(current_val, float):
                if isinstance(initial_val, (int, float)):
                    change_pct = ((current_val - initial_val) / initial_val * 100) if initial_val != 0 else 0
                    print(f"  {param}: {current_val:.4f} ({change_pct:+.1f}% from initial)")
                else:
                    print(f"  {param}: {current_val:.4f}")
            else:
                print(f"  {param}: {current_val}")

def create_default_scheduler():
    default_params = {
        'learning_rate': 0.001,
        'lr_decay_rate': 0.95,
        'lr_decay_frequency': 50,
        'total_iterations': 1000,
        
        #this is exclusively for exploration and exploitation from alphazero
        'initial_temperature': 1.0,
        'final_temperature': 0.1,
        'temperature_decay_iterations': 200,
        'initial_dirichlet_noise': 0.25,
        'final_dirichlet_noise': 0.05,
        'noise_decay_iterations': 300,
        'initial_c_puct': 1.0,
        'final_c_puct': 2.0,
        'c_puct_increase_iterations': 150,
        
        #the batch size params
        'initial_batch_size': 32,
        'batch_size': 32,
        'max_batch_size': 128,
        'batch_size_increase_frequency': 100,
        
        #the simulations params
        'initial_simulations': 100,
        'num_simulations': 100,
        'max_simulations': 400,
        'simulation_increase_frequency': 75,
    }
    
    return HyperparameterScheduler(default_params)
