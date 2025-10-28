import logging
import sys
import os
import torch
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
import os
import torch
import logging
import inspect

def init_opt(
    encoder,
    predictor,
    target_encoder=None,
    start_lr=None,
    ref_lr=None,
    warmup=None,
    num_epochs=None,
    iterations_per_epoch=None,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }
    ]
    if target_encoder is not None:
        param_groups.append({
            'params': (p for n, p in target_encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        })
    param_groups.extend([
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ])
    if target_encoder is not None:
        param_groups.append({
            'params': (p for n, p in target_encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        })
    print('int(ipe_scale*num_epochs*iterations_per_epoch))',int(ipe_scale*num_epochs*iterations_per_epoch))
    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler



logger = logging.getLogger(__name__)

def save_checkpoint(encoder, predictor, target_encoder, optimizer, scheduler, epoch, loss, checkpoint_dir, filename='checkpoint.pth', best_mode=False, best_dir=None):
    """
    Save model checkpoint and print details.
    If best_mode is True, saves to best_dir with epoch and loss in filename.
    """
    checkpoint = {
        'encoder': encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    if best_mode and best_dir is not None:
        os.makedirs(best_dir, exist_ok=True)
        # Format loss with 4 decimals, replace . with _ for filename safety
        loss_str = f"{loss:.4f}".replace('.', '_')
        best_filename = f"best_epoch_{epoch}_loss_{loss_str}.pth"
        checkpoint_path = os.path.join(best_dir, best_filename)
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"  - Saved epoch: {epoch}")
    print(f"  - Loss at save: {loss}")
    print("  - Stored keys: 'encoder', 'predictor', 'target_encoder', 'opt'")
    logger.info(f'Checkpoint saved to {checkpoint_path}')


def load_checkpoint(encoder, predictor, target_encoder, optimizer, scheduler, checkpoint_path, iterations_per_epoch):
    """
    Load model checkpoint and print details.
    """
    try:
        # Load the checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']  # Loaded epoch number
        loss = checkpoint['loss']    # Loaded loss value

        # -- loading encoder weights
        msg_enc = encoder.load_state_dict(checkpoint['encoder'])
        print(f"Loaded encoder weights from epoch {epoch}: {msg_enc}")

        # -- loading predictor weights
        msg_pred = predictor.load_state_dict(checkpoint['predictor'])
        print(f"Loaded predictor weights from epoch {epoch}: {msg_pred}")

        # -- loading target encoder weights
        msg_targ = target_encoder.load_state_dict(checkpoint['target_encoder'])
        print(f"Loaded target encoder weights from epoch {epoch}: {msg_targ}")

        # -- loading optimizer state
        optimizer.load_state_dict(checkpoint['opt'])
        print(f"Loaded optimizer state from epoch {epoch}")

        # -- update scheduler to the correct training step
        current_step = epoch * iterations_per_epoch
        step_sig = inspect.signature(scheduler.step)
        # If scheduler.step accepts an argument, pass current_step
        if len(step_sig.parameters) > 1:
            scheduler.step(current_step)
            print(f"Scheduler stepped to iteration: {current_step} (used step(current_step))")
        else:
            # Some schedulers only accept no args: call step() repeatedly
            for _ in range(current_step):
                scheduler.step()
            print(f"Scheduler stepped {current_step} times via repeated step() calls")

        print(f"Checkpoint loaded successfully from: {checkpoint_path}")
        logger.info(f'Checkpoint loaded from {checkpoint_path}')
        del checkpoint

    except Exception as e:
        print(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        logger.info(f'Encountered exception when loading checkpoint: {e}')
        epoch = 0
        loss = float('inf')

    return epoch, loss
