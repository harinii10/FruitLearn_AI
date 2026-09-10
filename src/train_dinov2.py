import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dinov2_backbone import load_dinov2_pretrained

class DINOLoss(nn.Module):
    """
    DINO Teacher-Student Cross-Entropy Loss with Centering and Softmax Sharpening.
    """
    def __init__(self, out_dim=65536, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        student_out = student_output / self.student_temp
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1).detach()
        
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True) / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

def train_dinov2_epoch():
    print("=== FruitLearn AI: DINOv2 Self-Distillation Engine ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = load_dinov2_pretrained().to(device)
    teacher = load_dinov2_pretrained().to(device)
    
    # Teacher parameters initialized same as student
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
        
    print(f"Device: {device} | Student Parameters: {sum(p.numel() for p in student.parameters()):,}")
    print("DINOv2 Self-Distillation Teacher-Student initialized successfully.")

if __name__ == "__main__":
    train_dinov2_epoch()
