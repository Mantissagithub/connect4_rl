import torch
import torch.nn as nn
from typing import Dict, Any
import time

def network_training(neural_net, optimizer, batch_size=32, num_epochs=10, verbose=True):
    from training_data_components.batch_sampling import batch_sampling
    from neural_network_components.calculate_loss import calculate_loss
    
    neural_net.train()
    training_start_time = time.time()
    
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_batches = 0
    
    metrics = {}
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_batches = 0
        
        try:
            batches = batch_sampling(batch_size=batch_size)
            
            if not batches:
                if verbose:
                    print(f"No training data available for epoch {epoch + 1}")
                continue
            
            for batch_idx, batch in enumerate(batches):
                states, policies, values = batch
                
                if not isinstance(states, torch.Tensor):
                    states = torch.tensor(states, dtype=torch.float32)
                if not isinstance(policies, torch.Tensor):
                    policies = torch.tensor(policies, dtype=torch.float32)
                if not isinstance(values, torch.Tensor):
                    values = torch.tensor(values, dtype=torch.float32)
                
                optimizer.zero_grad()
                policy_pred, value_pred = neural_net(states)
                
                loss, policy_loss, value_loss = calculate_loss(
                    policy_pred, value_pred, policies, values
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(neural_net.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                batch_loss = loss.item()
                batch_policy_loss = policy_loss.item()
                batch_value_loss = value_loss.item()
                
                epoch_loss += batch_loss
                epoch_policy_loss += batch_policy_loss
                epoch_value_loss += batch_value_loss
                epoch_batches += 1
                
                total_loss += batch_loss
                total_policy_loss += batch_policy_loss
                total_value_loss += batch_value_loss
                total_batches += 1
                
                if verbose and (batch_idx + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}: "
                          f"Loss={batch_loss:.4f}, Policy={batch_policy_loss:.4f}, Value={batch_value_loss:.4f}")
        
        except Exception as e:
            if verbose:
                print(f"Error during epoch {epoch + 1}: {e}")
            continue
        
        #summary of this epoch
        if epoch_batches > 0:
            avg_epoch_loss = epoch_loss / epoch_batches
            avg_epoch_policy_loss = epoch_policy_loss / epoch_batches
            avg_epoch_value_loss = epoch_value_loss / epoch_batches
            epoch_time = time.time() - epoch_start_time
            
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s: "
                      f"Loss={avg_epoch_loss:.4f}, Policy={avg_epoch_policy_loss:.4f}, Value={avg_epoch_value_loss:.4f}")
    
    #the final metrics
    training_time = time.time() - training_start_time
    
    if total_batches > 0:
        avg_total_loss = total_loss / total_batches
        avg_total_policy_loss = total_policy_loss / total_batches
        avg_total_value_loss = total_value_loss / total_batches
    else:
        avg_total_loss = 0.0
        avg_total_policy_loss = 0.0
        avg_total_value_loss = 0.0
    
    metrics = {
        'training_time': training_time,
        'total_batches': total_batches,
        'final_loss': avg_total_loss,
        'final_policy_loss': avg_total_policy_loss,
        'final_value_loss': avg_total_value_loss,
        'epochs_completed': num_epochs
    }
    
    if verbose:
        print(f"Network training completed in {training_time:.2f} seconds")
        print(f"Average loss: {avg_total_loss:.4f} (Policy: {avg_total_policy_loss:.4f}, Value: {avg_total_value_loss:.4f})")
    
    neural_net.eval()
    return metrics
