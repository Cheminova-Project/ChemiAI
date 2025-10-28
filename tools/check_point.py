import torch
import os
import re

# Function to save the model state


def save_checkpoint(model, optimizer, epoch, best_f1_score, checkpoint_dir, ema_state_dict=None, scheduler=None):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Format the F1 score to a fixed number of decimal places
    f1_score_str = f"{best_f1_score:.4f}"
    checkpoint_filename = f'checkpoint_epoch{epoch}_f1_{f1_score_str}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1_score': best_f1_score
    }
    
    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add EMA state if provided
    if ema_state_dict is not None:
        checkpoint_data['ema_state_dict'] = ema_state_dict
    
    torch.save(checkpoint_data, checkpoint_path)



# Function to load the model state


def load_checkpoint_main(model, optimizer, checkpoint_dir):
    # Regular expression to match checkpoint filenames
    checkpoint_pattern = re.compile(r'checkpoint_epoch(\d+)_f1_(\d+\.\d+).pth')
    
    checkpoints = []
    
    # Scan the directory for checkpoint files
    for filename in os.listdir(checkpoint_dir):
        match = checkpoint_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            f1_score = float(match.group(2))
            checkpoints.append((epoch, f1_score, filename))
    
    if checkpoints:
        # Sort by epoch or f1_score if needed
        checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)  # Sorting by epoch and f1_score in descending order
        latest_checkpoint = checkpoints[0][2]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_f1_score = checkpoint['best_f1_score']
        
        print(f"Checkpoint loaded successfully from '{checkpoint_path}' at epoch {epoch} with best F1 score {best_f1_score}.")
        return model, optimizer, epoch, best_f1_score
    else:
        print(f"No checkpoint found in '{checkpoint_dir}'. Starting training from scratch.")
        return model, optimizer, 0, 0.0




def load_checkpoint_test(model, checkpoint_dir):
    # Regular expression to match checkpoint filenames
    checkpoint_pattern = re.compile(r'checkpoint_epoch(\d+)_f1_(\d+\.\d+).pth')
    
    checkpoints = []
    
    # Scan the directory for checkpoint files
    for filename in os.listdir(checkpoint_dir):
        match = checkpoint_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            f1_score = float(match.group(2))
            checkpoints.append((epoch, f1_score, filename))
    
    if checkpoints:
        # Sort by epoch and f1_score in descending order
        checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)
        latest_checkpoint = checkpoints[0][2]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        # Load the model state dict only
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model weights loaded successfully from '{checkpoint_path}'.")
        return model
    else:
        print(f"No checkpoint found in '{checkpoint_dir}'. Starting with an untrained model.")
        return model


def load_checkpoint_1(model, optimizer, checkpoint_dir):
    # Regular expression to match checkpoint filenames
    checkpoint_pattern = re.compile(r'checkpoint_epoch(\d+)_f1_(\d+\.\d+).pth')
    
    checkpoints = []
    
    # Scan the directory for checkpoint files
    for filename in os.listdir(checkpoint_dir):
        match = checkpoint_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            f1_score = float(match.group(2))
            checkpoints.append((epoch, f1_score, filename))
    
    if checkpoints:
        # Sort by epoch or f1_score if needed
        checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)  # Sorting by epoch and f1_score in descending order
        latest_checkpoint = checkpoints[0][2]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        checkpoint = torch.load(checkpoint_path)
        
        # Filter out mismatched layers
        model_state_dict = model.state_dict()
        filtered_state_dict = {
            k: v for k, v in checkpoint['model_state_dict'].items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict)

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_f1_score = checkpoint['best_f1_score']
        
        print(f"Checkpoint loaded successfully from '{checkpoint_path}' at epoch {epoch} with best F1 score {best_f1_score}.")
        return model, optimizer, epoch, best_f1_score
    else:
        print(f"No checkpoint found in '{checkpoint_dir}'. Starting training from scratch.")
        return model, optimizer, 0, 0.0


def load_checkpoint(model, optimizer, checkpoint_dir):
    # Regular expression to match checkpoint filenames
    checkpoint_pattern = re.compile(r'checkpoint_epoch(\d+)_f1_(\d+\.\d+).pth')
    
    checkpoints = []
    
    # Scan the directory for checkpoint files
    for filename in os.listdir(checkpoint_dir):
        match = checkpoint_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            f1_score = float(match.group(2))
            checkpoints.append((epoch, f1_score, filename))
    
    if checkpoints:
        # Sort by epoch and F1 score in descending order
        checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)
        latest_checkpoint = checkpoints[0][2]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Get model state dict
        model_state_dict = model.state_dict()

        # Count total and matched parameters
        checkpoint_state_dict = checkpoint['model_state_dict']
        matched_params = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
        mismatched_params = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict and model_state_dict[k].shape != v.shape}
        missing_params = [k for k in model_state_dict.keys() if k not in checkpoint_state_dict]

        # Update only matching parameters
        model_state_dict.update(matched_params)
        model.load_state_dict(model_state_dict, strict=False)

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_f1_score = checkpoint['best_f1_score']

        # Print details
        print(f"Checkpoint loaded from '{checkpoint_path}' at epoch {epoch} with best F1 score {best_f1_score}.")
        print(f"Total model parameters: {len(model_state_dict)}")
        print(f"Matched parameters loaded: {len(matched_params)}")
        print(f"Mismatched parameters (shape mismatch): {len(mismatched_params)}")
        print(f"Missing parameters in checkpoint: {len(missing_params)}")

        return model, optimizer, epoch, best_f1_score
    else:
        print(f"No checkpoint found in '{checkpoint_dir}'. Starting training from scratch.")
        return model, optimizer, 0, 0.0

def save_best_checkpoint(model, optimizer, epoch, best_f1_score, checkpoint_dir, ema_state_dict=None, scheduler=None):
    """
    Save only the best checkpoint to a fixed filename 'best_model.pth'.
    Overwrites the previous best to avoid accumulating multiple files.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, 'best_f1_score.pth')

    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1_score': best_f1_score
    }

    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

    if ema_state_dict is not None:
        checkpoint_data['ema_state_dict'] = ema_state_dict

    torch.save(checkpoint_data, checkpoint_path)

def load_best_checkpoint_if_exists(model, optimizer, checkpoint_dir):
    """
    Load 'best_model.pth' if present. Returns (model, optimizer, epoch, best_f1_score) or None if missing.
    """
    best_path = os.path.join(checkpoint_dir, 'best_f1_score.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_f1_score = checkpoint['best_f1_score']
        print(f"Best checkpoint loaded from '{best_path}' at epoch {epoch} with best F1 score {best_f1_score}.")
        return model, optimizer, epoch, best_f1_score
    return None

