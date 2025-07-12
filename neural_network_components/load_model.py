# Load Model
# Purpose
# Restore previously trained network from saved checkpoint files.

# Loading Process
# Architecture Recreation: Initialize network with same structure

# Weight Restoration: Load saved parameters into network layers

# Optimizer State: Restore optimizer for continued training

# Validation: Verify model loads correctly and produces expected outputs

# Use Cases
# Resume Training: Continue training from checkpoint

# Inference Only: Load for game playing without further training

# Transfer Learning: Use pre-trained weights as starting point

# Model Evaluation: Load different checkpoints for performance comparison

# Error Handling
# Architecture Mismatch: Verify network structure matches saved model

# Missing Files: Handle cases where checkpoint files are corrupted

# Device Compatibility: Ensure model loads on correct device (CPU/GPU)

# this is for my ref, don't mind!!

import torch
import os
import time
from .neural_network import NeuralNetwork

def load_model(file_path='model.pth', device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(file_path):
        print(f"Model file not found: {file_path}")
        return None, None, None, None, None
    
    try:
        checkpoint = torch.load(file_path, map_location=device)
        
        model = NeuralNetwork(device=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        
        epoch = checkpoint.get('epoch', 0)
        loss_history = checkpoint.get('loss_history', [])
        timestamp = checkpoint.get('timestamp', time.time())
        
        model.eval()
        
        print(f"model successfully loaded from {file_path}")
        print(f"epoch finished/running : {epoch}, device(mostly cuda): {device}")
        
        return model, optimizer_state, epoch, loss_history, timestamp
        
    except Exception as e:
        print(f"error loading model: {e}")
        return None, None, None, None, None

def load_model_for_inference(file_path='model.pth', device=None):
    model, _, _, _, _ = load_model(file_path, device)
    
    if model is not None:
        model.eval() 
        
    return model

def load_model_for_training(file_path='model.pth', optimizer_class=None, device=None):
    model, optimizer_state, epoch, loss_history, timestamp = load_model(file_path, device)
    
    if model is None:
        return None, None, None, None
    
    optimizer = None
    if optimizer_class is not None:
        optimizer = optimizer_class(model.parameters())
        
        if optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
                print("Optimizer state restored")
            except Exception as e:
                print(f"Warning: Could not restore optimizer state: {e}")
                print("Using fresh optimizer")
    
    model.train()
    
    return model, optimizer, epoch, loss_history

def verify_model_compatibility(file_path='model.pth'):
    if not os.path.exists(file_path):
        return False, {"error": "File not found"}
    
    try:
        checkpoint = torch.load(file_path, map_location='cpu')
        
        temp_model = NeuralNetwork()
        temp_model.load_state_dict(checkpoint['model_state_dict'])
        
        info = {
            "epoch": checkpoint.get('epoch', 'Unknown'),
            "timestamp": checkpoint.get('timestamp', 'Unknown'),
            "model_parameters": sum(p.numel() for p in temp_model.parameters()),
            "architecture": str(temp_model),
            "compatible": True
        }
        
        return True, info
        
    except Exception as e:
        return False, {"error": str(e)}
    
#i know this is a bit redundant and unlikely, if possible, can load multiple models na
def load_multiple_models(file_paths, device=None):
    models = []
    metadata = []
    
    for file_path in file_paths:
        model, _, epoch, loss_history, timestamp = load_model(file_path, device)
        
        if model is not None:
            models.append(model)
            metadata.append({
                'file_path': file_path,
                'epoch': epoch,
                'loss_history': loss_history,
                'timestamp': timestamp
            })
        else:
            print(f"failed to load model from {file_path}")
    
    return models, metadata
