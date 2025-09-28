import torch
from transformers import TrainerCallback
import numpy as np


class TrainingHealthMonitor(TrainerCallback):
    """
    Monitor training health and stop if gradient explosion detected
    Add this to your trainer to prevent model corruption
    """

    def __init__(self, patience=3):
        self.patience = patience
        self.nan_count = 0
        self.last_loss = None
        self.loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Check training health on each log"""
        if logs:
            # Check for NaN/Inf in loss
            if 'loss' in logs:
                loss = logs['loss']
                self.loss_history.append(loss)

                # Check for NaN or Inf
                if np.isnan(loss) or np.isinf(loss):
                    self.nan_count += 1
                    print(f"\n⚠️ WARNING: NaN/Inf detected in loss! Count: {self.nan_count}/{self.patience}")

                    if self.nan_count >= self.patience:
                        print("\n🛑 STOPPING: Too many NaN/Inf values - model is corrupting!")
                        control.should_training_stop = True
                        return control

                # Check for zero loss (training collapsed)
                elif loss == 0.0 and len(self.loss_history) > 10:
                    print("\n🛑 STOPPING: Loss collapsed to zero - training failed!")
                    control.should_training_stop = True
                    return control

                # Check for exploding loss
                elif self.last_loss and loss > self.last_loss * 100:
                    print(f"\n⚠️ WARNING: Loss exploded from {self.last_loss:.4f} to {loss:.4f}")
                    self.nan_count += 1

                    if self.nan_count >= self.patience:
                        print("\n🛑 STOPPING: Loss explosion detected!")
                        control.should_training_stop = True
                        return control

                # Check gradient norm
                if 'grad_norm' in logs:
                    grad_norm = logs['grad_norm']
                    if np.isnan(grad_norm) or np.isinf(grad_norm):
                        print(f"\n⚠️ WARNING: NaN/Inf gradient norm: {grad_norm}")
                        self.nan_count += 1

                        if self.nan_count >= self.patience:
                            print("\n🛑 STOPPING: Gradient explosion detected!")
                            control.should_training_stop = True
                            return control
                    elif grad_norm > 100:
                        print(f"\n⚠️ WARNING: Very high gradient norm: {grad_norm}")

                # Reset counter if training is healthy
                if not (np.isnan(loss) or np.isinf(loss) or loss == 0.0):
                    self.nan_count = 0
                    self.last_loss = loss

                    # Provide feedback every 10 steps
                    if state.global_step % 10 == 0 and state.global_step > 0:
                        avg_loss = np.mean(self.loss_history[-10:])
                        print(f"\n✅ Training healthy - Avg loss: {avg_loss:.4f}")

        return control


# How to use this monitor in your training:
"""
# In your train_and_upload_model function, after creating the Trainer:

from training_monitor import TrainingHealthMonitor

# Create trainer with health monitor
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[TrainingHealthMonitor(patience=3)]  # Add this line
)
"""