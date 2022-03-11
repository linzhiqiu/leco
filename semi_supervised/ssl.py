import torch
import torch.nn as nn
import torch.nn.functional as F

class SSLObjective(nn.Module):
    def __init__(self, hierarchical_ssl=None, edge_matrix=None):
        self.hierarchical_ssl = hierarchical_ssl
        self.edge_matrix = edge_matrix
        _, self.fine_to_coarse = torch.nonzero(edge_matrix == 1.0, as_tuple=True)
        assert self.fine_to_coarse.size(0) == edge_matrix.size(0)
        super(SSLObjective, self).__init__()
        
    def forward(self, model, inputs, labels):
        raise NotImplementedError()
    
    def masked_softmax(self, outputs, coarse_labels):
        assert outputs.size(0) == coarse_labels.size(0)
        mask = self.edge_matrix.T[coarse_labels]
        exps = torch.exp(outputs)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(1, keepdim=True)
        return (masked_exps/masked_sums)

    def masked_log_softmax(self, outputs, coarse_labels, epsilon=1e-20):
        assert outputs.size(0) == coarse_labels.size(0)
        mask = self.edge_matrix.T[coarse_labels]
        exps = torch.exp(outputs)
        masked_exps = exps * mask.float() + epsilon
        log_masked_exps = torch.log(masked_exps)
        log_sum_masked_exps = torch.log(masked_exps.sum(1, keepdim=True))
        res = log_masked_exps - log_sum_masked_exps
        return res
    
    def put_on_device(self, labels, device):
        return [l.to(device) for l in labels]

    def calc_filter_mask(self, probs, labels):
        if self.hierarchical_ssl in ['filtering', 'filtering_conditioning']:
            _, pred = probs.max(1)
            pred_labels = torch.Tensor([self.fine_to_coarse[label] for label in pred]).to(probs.device)
            gt_labels = labels[0]
            return pred_labels == gt_labels
        elif self.hierarchical_ssl in [None, 'conditioning']:
            return labels[0] == labels[0]
        else:
            raise NotImplementedError()
    
    def calc_outputs(self, model, inputs):
        # Calculate output in a numerically stable way
        outputs = model(inputs)
        outputs = outputs - outputs.max(1)[0].unsqueeze(1)
        return outputs
    
    def condition_outputs_for_probs(self, outputs, labels):
        if self.hierarchical_ssl in ['conditioning', 'filtering_conditioning']:
            return self.masked_softmax(outputs, labels[0])
        elif self.hierarchical_ssl in [None, 'filtering']:
            return F.softmax(outputs, dim=1)
        else:
            raise NotImplementedError()
    
    def condition_outputs_for_log_probs(self, outputs, labels):
        if self.hierarchical_ssl in ['conditioning', 'filtering_conditioning']:
            return self.masked_log_softmax(outputs, labels[0])
        elif self.hierarchical_ssl in [None, 'filtering']:
            return F.log_softmax(outputs, dim=1)
        else:
            raise NotImplementedError()
    
    def calc_ssl_stats(self, probs, labels):
        # probs is a N x NUM_OF_CLASS tensor
        N = probs.size(0)
        max_probs, pred = probs.max(1)
        pred_labels = torch.zeros((2, N))
        pred_labels[1] = pred.cpu()
        pred_labels[0] = torch.Tensor([self.fine_to_coarse[label] for label in pred])
        
        gt_labels = torch.zeros((2, N))
        for i in [0, 1]:
            gt_labels[i] = labels[i].cpu()
        
        ssl_stats = {
            # size 1xN array
            'max_probs': max_probs.unsqueeze(0).cpu(),
            # size 2xN array (0 is coarse, 1 is fine)
            'pred_labels': pred_labels,
            # size 2xN array (0 is coarse, 1 is fine)
            'gt_labels': gt_labels,
        }
        return ssl_stats


class NoSSL(SSLObjective):
    def __init__(self, hierarchical_ssl=None, edge_matrix=None):
        super(NoSSL, self).__init__(
            hierarchical_ssl=hierarchical_ssl,
            edge_matrix=edge_matrix
        )
        assert hierarchical_ssl == None

    def forward(self, model, inputs, labels):
        outputs = self.calc_outputs(model, inputs)
        probs = F.softmax(outputs, dim=1)
        ssl_stats = self.calc_ssl_stats(probs, labels)
        return ssl_stats, 0.0