# ave Model
# Purpose
# Persist trained network weights and training state to disk for later use or deployment.

# What Gets Saved
# Network Parameters: All weights and biases from conv and dense layers

# Optimizer State: Adam momentum and variance estimates

# Training Metadata: Epoch count, loss history, hyperparameters

# Model Architecture: Network structure for reconstruction

# File Format Considerations
# PyTorch Format: .pt or .pth files with torch.save()

# Checkpoint Structure: Dictionary containing all necessary components

# Version Compatibility: Ensure saved models work across PyTorch versions

# this is for my ref, don't min!!

import torch
import os
import time

def save_model(model, optimizer, epoch, loss_history, file_path='model.pth'):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(), #network params
            'optimizer_state_dict': optimizer.state_dict(), #about optmizer
            'epoch': epoch, #about epoch count
            'loss_history': loss_history,
            'model_architecture': str(model), #model architecture as string,
            'timestamp':time.time()
        }, file_path)

        print(f"Model saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False
    
# torch.save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True)[source][source]
# Saves an object to a disk file.

# See also: Saving and loading tensors

# Parameters
# obj (object) – saved object

# f (Union[str, PathLike[str], IO[bytes]]) – a file-like object (has to implement write and flush) or a string or os.PathLike object containing a file name

# pickle_module (Any) – module used for pickling metadata and objects

# pickle_protocol (int) – can be specified to override the default protocol

#this is from the officical pytorch docs, so don't mind this