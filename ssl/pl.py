import torch.nn as nn
import torch.nn.functional as F
from .ssl import SSLObjective

class PseudoLabel(SSLObjective):
    def __init__(self, pl_threshold, hierarchical_supervision=None, edge_matrix=None):
        super(PseudoLabel, self).__init__(
            hierarchical_supervision=hierarchical_supervision,
            edge_matrix=edge_matrix
        )
        self.pl_threshold = pl_threshold
        assert pl_threshold != None
        
    def calc_pl_mask(self, probs):
        max_probs, _ = probs.max(1)
        return max_probs >= self.pl_threshold
    
    def forward(self, model, inputs, labels):
        labels = self.put_on_device(labels, inputs.device)
        
        outputs = self.calc_outputs(model, inputs)
        probs = F.softmax(outputs, dim=1)
        ssl_stats = self.calc_ssl_stats(probs, labels)
        
        filter_mask = self.calc_filter_mask(probs, labels)
        
        conditioned_probs = self.condition_outputs_for_probs(outputs, labels)
        pl_mask = self.calc_pl_mask(conditioned_probs)
        final_mask = pl_mask & filter_mask
        
        pl_loss = torch.nn.NLLLoss(reduction='none')(
            torch.log(conditioned_probs) + 1e-20,
            labels[1]
        )
        
        pl_loss = final_mask * pl_loss
        
        return ssl_stats, pl_loss.mean()