import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, 
                 center_momentum=0.9, warmup_teacher_temp=0.04, 
                 warmup_teacher_temp_epochs=0, nepochs=100):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.out_dim = out_dim
        
        # We register the 'center' buffer. 
        # It is NOT a model parameter (no gradients), but part of the state.
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Teacher temperature schedule
        # We warm up the teacher temperature to avoid instability early on,
        # then decay it or keep it constant.
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        
        Args:
            student_output: (Batch * Num_Crops, Out_Dim)
            teacher_output: (Batch * 2, Out_Dim) -> Teacher only sees global crops
            epoch: Current training epoch (used for temperature schedule)
        """
        
        # 1. Get Student Softmax
        # Student output is smoothed by a higher temperature (0.1)
        student_out = student_output / self.student_temp
        
        # We keep the output in log-space for numerical stability in CrossEntropy
        # Total crops = 2 global + N local.
        # Student sees ALL crops.
        student_out = student_out.chunk(student_output.shape[0] // teacher_output.shape[0])
        
        # 2. Get Teacher Softmax (Centered & Sharpened)
        # Teacher output is centered (subtract mean) and sharpened (lower temp)
        temp = self.teacher_temp_schedule[epoch]
        
        # Apply Centering
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        
        # 3. Calculate Loss
        # We want the student's view of crop X to match the teacher's view of Global Crop Y.
        total_loss = 0
        n_loss_terms = 0
        
        # Iterate over both global views from the teacher
        for iq, q in enumerate(teacher_output.chunk(2)):
            for v in range(len(student_out)):
                # Skip if the student and teacher are looking at the exact same view
                # (Standard DINO implementation detail: we don't compute loss 
                #  when student and teacher effectively see the exact same image instance)
                if v == iq: 
                    continue
                
                # Cross Entropy: - sum( P_teacher * log(P_student) )
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
                
        total_loss /= n_loss_terms
        
        # 4. Update the Center (EMA)
        self.update_center(teacher_output)
        
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update the center used for teacher output normalization.
        Exponential Moving Average (EMA).
        """
        # Calculate mean of the current batch
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        
        # If distributed training, we would sync here (dist.all_reduce).
        # Assuming single GPU for this project scope:
        batch_center = batch_center / len(teacher_output)
        
        # Update running average
        # center = m * center + (1-m) * batch_center
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)