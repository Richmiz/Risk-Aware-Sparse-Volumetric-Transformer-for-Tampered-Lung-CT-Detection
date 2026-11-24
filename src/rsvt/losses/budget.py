"""Soft expected-token penalty; hard inference caps are enforced by the packer."""

import math
import torch
from rsvt.models.routing import validate_cap


def expected_token_budget_loss(expected_sequence_counts,sequence_cap):
    validate_cap(sequence_cap)
    if (expected_sequence_counts.ndim!=1 or not expected_sequence_counts.numel()
            or not torch.isfinite(expected_sequence_counts).all() or (expected_sequence_counts<1).any()):
        raise ValueError("Expected sequence counts must be a finite nonempty vector >=1")
    return ((expected_sequence_counts-sequence_cap).clamp_min(0)/sequence_cap).square().mean()


class ProjectedTokenDual:
    """Projected ascent on a max-expected-sequence surrogate, not measured runtime."""
    def __init__(self,initial=.1,learning_rate=.05,maximum=10.):
        values=(initial,learning_rate,maximum)
        if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values)
                or learning_rate<=0 or maximum<=0 or not 0<=initial<=maximum):
            raise ValueError("Invalid projected-dual parameters")
        self.multiplier=float(initial);self.learning_rate=float(learning_rate);self.maximum=float(maximum);self.updates=0

    def terms(self,expected_sequence_counts,sequence_cap):
        expected_token_budget_loss(expected_sequence_counts,sequence_cap)
        constraint=(expected_sequence_counts.max()-sequence_cap)/sequence_cap
        return {"loss":constraint*self.multiplier,"constraint":constraint,"multiplier":self.multiplier,
                "scope":"max_expected_tokens_surrogate_soft_training_remains_dense"}

    def update(self,constraint):
        value=float(constraint.detach()) if isinstance(constraint,torch.Tensor) else float(constraint)
        if not math.isfinite(value):raise ValueError("Dual update requires finite constraint")
        self.multiplier=min(self.maximum,max(0.,self.multiplier+self.learning_rate*value));self.updates+=1
        return self.multiplier

    def state_dict(self):
        return {"multiplier":self.multiplier,"learning_rate":self.learning_rate,"maximum":self.maximum,"updates":self.updates}

    def load_state_dict(self,state):
        if set(state)!={'multiplier','learning_rate','maximum','updates'}:raise ValueError("Invalid dual state fields")
        restored=ProjectedTokenDual(state['multiplier'],state['learning_rate'],state['maximum'])
        if isinstance(state['updates'],bool) or not isinstance(state['updates'],int) or state['updates']<0:raise ValueError("Invalid dual update counter")
        self.multiplier=restored.multiplier;self.learning_rate=restored.learning_rate;self.maximum=restored.maximum;self.updates=state['updates']
