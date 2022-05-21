import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssl import SSLObjective

class PreconDistillSoft(SSLObjective):
    """Different from the implementation in Maji's work, this only compute KLD on unlabeled data"""
    def __init__(self, model_T, T=1.0, hierarchical_ssl=None, edge_matrix=None):
        super(PreconDistillSoft, self).__init__(
            hierarchical_ssl=hierarchical_ssl,
            edge_matrix=edge_matrix
        )
        self.model_T = model_T
        self.model_T.eval()
        self.T = T
    
    def masked_kl(self, model, outputs_T, inputs, labels):
        with torch.no_grad():
            # outputs_t = self.calc_outputs(self.model_T, inputs)
            # F.log_softmax(outputs, dim=1)
            log_p_t = self.condition_outputs_for_log_probs(outputs_T/self.T, labels)
        
        outputs_s = self.calc_outputs(model, inputs)
        log_p_s = F.log_softmax(outputs_s/self.T, dim=1)
        kld_loss = F.kl_div(log_p_s, log_p_t, reduction='none', log_target=True)
        kld_loss = kld_loss * (self.T**2) * self.edge_matrix.T[labels[0]]
        return kld_loss.sum(1)

    def forward(self, model, inputs, labels):
        labels = self.put_on_device(labels, inputs.device)
        with torch.no_grad():
            outputs_all_heads_T = self.model_T(inputs)
            outputs_T = torch.zeros((inputs.shape[0], self.edge_matrix.shape[0])).to(inputs.device)
            probs_T = torch.zeros((inputs.shape[0], self.edge_matrix.shape[0])).to(inputs.device)
            for idx, parent_idx in enumerate(labels[0]):
                child_probs = F.softmax(outputs_all_heads_T[parent_idx][idx], dim=0)
                for child_idx, child_prob in enumerate(child_probs):
                    leaf_idx = self.model_T.fc.parent_and_child_idx_to_leaf_idx[parent_idx][child_idx]
                    probs_T[idx][leaf_idx] = child_prob
                    outputs_T[idx][leaf_idx] = outputs_all_heads_T[parent_idx][idx][child_idx]
            ssl_stats = self.calc_ssl_stats(probs_T, labels)
        
        kld_loss = self.masked_kl(model, outputs_T, inputs, labels)
        filter_mask = self.calc_filter_mask(probs_T, labels)
        
        kld_loss = filter_mask * kld_loss
        return ssl_stats, kld_loss.mean()


class PreconDistillHard(SSLObjective):
    """Different from the implementation in Maji's work, this only compute CrossEntropy on unlabeled data"""
    def __init__(self, model_T, hierarchical_ssl=None, edge_matrix=None):
        super(PreconDistillHard, self).__init__(
            hierarchical_ssl=hierarchical_ssl,
            edge_matrix=edge_matrix
        )
        self.model_T = model_T
        self.model_T.eval()
        
    def forward(self, model, inputs, labels):
        labels = self.put_on_device(labels, inputs.device)
        with torch.no_grad():
            outputs_T = self.model_T(inputs)
            probs_T = torch.zeros((inputs.shape[0], self.edge_matrix.shape[0])).to(inputs.device)
            for idx, parent_idx in enumerate(labels[0]):
                child_probs = F.softmax(outputs_T[parent_idx][idx], dim=0)
                for child_idx, child_prob in enumerate(child_probs):
                    leaf_idx = self.model_T.fc.parent_and_child_idx_to_leaf_idx[parent_idx][child_idx]
                    probs_T[idx][leaf_idx] = child_prob
            ssl_stats = self.calc_ssl_stats(probs_T, labels)
        
        outputs = self.calc_outputs(model, inputs)
        log_probs = F.log_softmax(outputs, dim=1)
        
        ce_loss = torch.nn.NLLLoss(reduction='none')(
            log_probs,
            probs_T.argmax(1)
        )

        return ssl_stats, ce_loss.mean()