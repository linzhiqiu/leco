import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssl import SSLObjective

class DistillSoft(SSLObjective):
    """Different from the implementation in Maji's work, this only compute KLD on unlabeled data"""
    def __init__(self, model_T, T=1.0, hierarchical_ssl=None, edge_matrix=None):
        super(DistillSoft, self).__init__(
            hierarchical_ssl=hierarchical_ssl,
            edge_matrix=edge_matrix
        )
        self.model_T = model_T
        self.model_T.eval()
        self.T = T
    
    def masked_kl(self, model, outputs_T, inputs, labels):
        with torch.no_grad():
            # outputs_t = self.calc_outputs(self.model_T, inputs)
            log_p_t = self.condition_outputs_for_log_probs(outputs_T/self.T, labels)
        
        outputs_s = self.calc_outputs(model, inputs)
        log_p_s = F.log_softmax(outputs_s/self.T, dim=1)
        kld_loss = F.kl_div(log_p_s, log_p_t, reduction='none', log_target=True)
        kld_loss = kld_loss * (self.T**2) * self.edge_matrix.T[labels[0]]
        return kld_loss.sum(1)

    def forward(self, model, inputs, labels):
        labels = self.put_on_device(labels, inputs.device)
        with torch.no_grad():
            outputs_T = self.calc_outputs(self.model_T, inputs)
            # outputs = self.calc_outputs(model, inputs)
            probs_T = F.softmax(outputs_T, dim=1)
            ssl_stats = self.calc_ssl_stats(probs_T, labels)
        
        kld_loss = self.masked_kl(model, outputs_T, inputs, labels)
        filter_mask = self.calc_filter_mask(probs_T, labels)
        
        kld_loss = filter_mask * kld_loss
        return ssl_stats, kld_loss.mean()


class DistillHard(SSLObjective):
    """Different from the implementation in Maji's work, this only compute CrossEntropy on unlabeled data"""
    def __init__(self, model_T, hierarchical_ssl=None, edge_matrix=None):
        super(DistillHard, self).__init__(
            hierarchical_ssl=hierarchical_ssl,
            edge_matrix=edge_matrix
        )
        self.model_T = model_T
        self.model_T.eval()
        
    def calc_labels_T(self, inputs):
        with torch.no_grad():
            outputs_t = self.model_T(inputs)
            _, labels_T = outputs_t.max(1)
        return labels_T

    def forward(self, model, inputs, labels):
        labels = self.put_on_device(labels, inputs.device)
        with torch.no_grad():
            outputs_T = self.calc_outputs(self.model_T, inputs)
            # outputs = self.calc_outputs(model, inputs)
            probs_T = F.softmax(outputs_T, dim=1)
            ssl_stats = self.calc_ssl_stats(probs_T, labels)
        
        # outputs = self.calc_outputs(model, inputs)
        # probs = F.softmax(outputs, dim=1)
        # ssl_stats = self.calc_ssl_stats(probs, labels)
        
        filter_mask = self.calc_filter_mask(probs_T, labels)
        outputs = self.calc_outputs(model, inputs)
        log_probs = F.log_softmax(outputs, dim=1)
        
        # conditioned_log_probs = self.condition_outputs_for_log_probs(outputs, labels)
        # labels_T = self.calc_labels_T(inputs)
        ce_loss = torch.nn.NLLLoss(reduction='none')(
            log_probs,
            self.calc_conditioned_labels(self.model_T, inputs, labels)
        )

        ce_loss = filter_mask * ce_loss
        return ssl_stats, ce_loss.mean()