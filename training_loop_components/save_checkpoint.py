import torch
import os
import time
from typing import Dict, Any, Optional

def save_checkpoint(neural_net, optimizer, iteration, metrics, checkpoint_dir="checkpoints", save_frequency=10):  
    #just a small check to avoid saving too frequenly, just skipping the save if not needed
    if iteration % save_frequency != 0:
        return None
    
    try:
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"checkpoint_iter_{iteration:04d}_{timestamp}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        
        checkpoint_data = {
            'iteration': iteration,
            'model_state_dict': neural_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': timestamp,
            'model_architecture': {
                'type': type(neural_net).__name__,
                'parameters': sum(p.numel() for p in neural_net.parameters()),
                'trainable_parameters': sum(p.numel() for p in neural_net.parameters() if p.requires_grad)
            }
        }

        torch.save(checkpoint_data, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        print(f"  Iteration: {iteration}")
        print(f"  Model parameters: {checkpoint_data['model_architecture']['parameters']:,}")
        print(f"  Trainable parameters: {checkpoint_data['model_architecture']['trainable_parameters']:,}")
        
        #keeping only the latest checkpoints to save disk space, already im running low on disk space 😁, so lets keep only 5 for now, for this emoji just go to this website and get thecpoy of the emote you want: https://emojicopy.com/
        cleanup_old_checkpoints(checkpoint_dir, keep_latest=5)
        
        return checkpoint_path
        
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return None

def load_checkpoint(checkpoint_path, neural_net, optimizer=None):
    try:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        neural_net.load_state_dict(checkpoint_data['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Iteration: {checkpoint_data.get('iteration', 'Unknown')}")
        print(f"Timestamp: {checkpoint_data.get('timestamp', 'Unknown')}")
        
        return {
            'iteration': checkpoint_data.get('iteration', 0),
            'metrics': checkpoint_data.get('metrics', {}),
            'timestamp': checkpoint_data.get('timestamp', ''),
            'model_architecture': checkpoint_data.get('model_architecture', {})
        }
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def cleanup_old_checkpoints(checkpoint_dir, keep_latest=5):
    try:
        if not os.path.exists(checkpoint_dir):
            return
        
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                          if f.startswith('checkpoint_') and f.endswith('.pt')]
        
        if len(checkpoint_files) <= keep_latest:
            return
        
        #sorting by time saved/modified
        checkpoint_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)
        
        #removing the old files
        files_to_remove = checkpoint_files[keep_latest:]
        for filename in files_to_remove:
            file_path = os.path.join(checkpoint_dir, filename)
            os.remove(file_path)
            print(f"Removed old checkpoint: {filename}")
            
    except Exception as e:
        print(f"Error cleaning up checkpoints: {e}")

def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    try:
        if not os.path.exists(checkpoint_dir):
            return None
        
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                          if f.startswith('checkpoint_') and f.endswith('.pt')]
        
        if not checkpoint_files:
            return None
        
        checkpoint_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)
        
        return os.path.join(checkpoint_dir, checkpoint_files[0])
        
    except Exception as e:
        print(f"Error finding latest checkpoint: {e}")
        return None

def save_best_model(neural_net, metrics, best_models_dir="best_models", metric_name="final_loss", lower_is_better=True):
    try:
        os.makedirs(best_models_dir, exist_ok=True)
        
        best_metric_file = os.path.join(best_models_dir, f"best_{metric_name}.txt")
        best_model_file = os.path.join(best_models_dir, f"best_model_{metric_name}.pt")
        
        current_metric_value = metrics.get(metric_name)
        if current_metric_value is None:
            print(f"Metric '{metric_name}' not found in metrics")
            return False
        
        is_best = False
        if os.path.exists(best_metric_file):
            with open(best_metric_file, 'r') as f:
                best_metric_value = float(f.read().strip())
            
            if lower_is_better:
                is_best = current_metric_value < best_metric_value
            else:
                is_best = current_metric_value > best_metric_value
        else:
            is_best = True  #the first model is automatically the best, fallback!!
        
        if is_best:
            torch.save(neural_net.state_dict(), best_model_file)
            
            with open(best_metric_file, 'w') as f:
                f.write(str(current_metric_value))
            
            print(f"New best model saved! {metric_name}: {current_metric_value:.6f}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error saving best model: {e}")
        return False
