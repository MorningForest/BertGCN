from torch import nn
import torch as th
from torch.nn.parameter import Parameter
from transformers import BertModel
import math

class GraphConvolution(nn.Module):
    """
    Simple pygGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, infeatn, adj):
        sp = th.matmul(infeatn, self.weight)
        output = th.matmul(adj, sp)
        # support = th.spmm(infeatn, self.weight)
        # output = th.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = th.relu(x)
        x = th.dropout(x, self.dropout, train=self.training)
        x = self.gc2(x, adj)
        return x

class BertGCN(nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, gcn_hidden_size=256,
                  m=0.3, dropout=0.5, graph_info=None):
        super(BertGCN, self).__init__()
        self.bert_model = BertModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.clssifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(self.feat_dim, gcn_hidden_size, nb_class, dropout=dropout)
        self.graph_info = graph_info
        self.m = m

    def forward(self, input_ids, doc_id):
        attention_mask = input_ids > 0
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_pred = self.clssifier(cls_feats)
        self.graph_info['feats'][doc_id] = cls_feats.detach()
        gcn_pred = self.gcn(self.graph_info['feats'],self.graph_info['adj'])[doc_id]
        pred = gcn_pred*self.m + (1-self.m)*cls_pred
        return {'pred': pred}