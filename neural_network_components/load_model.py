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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(file_path='model.pth'):
    if not os.path.exists(file_path):
        print(f"model file is missing, cehck the path: {file_path}")
        return None
    #this is how i'm saving in save_model
    # torch.save({
    #         'model_state_dict': model.state_dict(), #network params
    #         'optimizer_state_dict': optimizer.state_dict(), #about optmizer
    #         'epoch': epoch, #about epoch count
    #         'loss_history': loss_history,
    #         'model_architecture': str(model), #model architecture as string,
    #         'timestamp':time.time()
    #     }, file_path)
    try:
        checkpoint = torch.load(file_path, map_location=device)
        model = checkpoint['model_architecture']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = checkpoint['optimizer_state_dict']
        epoch = checkpoint['epoch']
        loss_history = checkpoint['loss_history']
        timestamp = checkpoint['timestamp']
        print(f"model successfully loaded from {file_path}")
        return model, optimizer, epoch, loss_history, timestamp
    except Exception as e:
        print(f"error loading model: {e}")
        return None, None, None, None, None