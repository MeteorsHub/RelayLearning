import torch
import torch.nn.functional as F


class DistillationLoss(torch.nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, teacher_logits, student_logits):
        return distillation_loss(teacher_logits, student_logits, self.tau)


def distillation_loss(teacher_logits, student_logits, tau=1.0):
    prob_t = F.softmax(teacher_logits / tau, dim=1)
    log_prob_s = F.log_softmax(student_logits / tau, dim=1)
    dist_loss = -(prob_t * log_prob_s).sum(dim=1).mean()
    return dist_loss
