import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssl import SSLObjective

class Fixmatch(SSLObjective):
    def __init__(self, pl_threshold, hierarchical_ssl=None, edge_matrix=None):
        super(Fixmatch, self).__init__(
            hierarchical_ssl=hierarchical_ssl,
            edge_matrix=edge_matrix
        )
        self.pl_threshold = pl_threshold
        assert pl_threshold != None
        
    def calc_pl_mask(self, probs):
        max_probs, _ = probs.max(1)
        return max_probs >= self.pl_threshold
    
    def forward(self, model, inputs, labels):
        inputs_w, inputs_s = inputs
        
        labels = self.put_on_device(labels, inputs_w.device)
        
        outputs_s = self.calc_outputs(model, inputs_s)
        probs_s = F.softmax(outputs_s, dim=1)
        
        ssl_stats = self.calc_ssl_stats(probs_s, labels)
        
        filter_mask = self.calc_filter_mask(probs_s, labels)
        
        conditioned_probs = self.condition_outputs_for_probs(outputs_s, labels)
        pl_mask = self.calc_pl_mask(conditioned_probs)
        final_mask = pl_mask & filter_mask
        
        conditioned_log_probs = self.condition_outputs_for_log_probs(outputs_s, labels)
        
        pl_loss = torch.nn.NLLLoss(reduction='none')(
            # torch.log(conditioned_probs) + 1e-20,
            conditioned_log_probs,
            self.calc_labels(model, inputs_w)
        )
        pl_loss = final_mask * pl_loss
        return ssl_stats, pl_loss.mean()