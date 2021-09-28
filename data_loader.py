import torch as th
import scipy.sparse as sp
import numpy as np
from transformers import AutoTokenizer

from fastNLP.io.loader import MRLoader, OhsumedLoader, NG20Loader, R52Loader, R8Loader
from fastNLP.io.pipe import MRPmiGraphPipe, OhsumedPmiGraphPipe, R8PmiGraphPipe, R52PmiGraphPipe, NG20PmiGraphPipe
from fastNLP.io.loader import loader
from fastNLP import Vocabulary


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def preprocess_adj(adj, is_sparse=False):
    """Preprocessing of adjacency matrix for simple pygGCN model and conversion to
    tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    if is_sparse:
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized
    else:
        return th.from_numpy(adj_normalized.A).float()

class PrepareData:
    def __init__(self, args):
        self.arg = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        if self.arg.dataset == 'mr':
            data_bundle, adj, target_vocab = self._get_input(MRLoader, MRPmiGraphPipe,  args.dev_ratio)
        elif self.arg.dataset == 'R8':
            data_bundle, adj, target_vocab = self._get_input(R8Loader, R8PmiGraphPipe,  args.dev_ratio)
        elif self.arg.dataset == 'R52':
            data_bundle, adj, target_vocab = self._get_input(R52Loader, R52PmiGraphPipe,  args.dev_ratio)
        elif self.arg.dataset == 'ohsumed':
            data_bundle, adj, target_vocab = self._get_input(OhsumedLoader, OhsumedPmiGraphPipe,  args.dev_ratio)
        elif self.arg.dataset == '20ng':
            data_bundle, adj, target_vocab = self._get_input(NG20Loader, NG20PmiGraphPipe,  args.dev_ratio)
        else:
            raise RuntimeError('输入数据集错误，请更改为["mr", "R8", "R52", "ohsumed", "20ng"]')

        self.data_bundle = data_bundle
        self.target_vocab = target_vocab
        ## 论文中的memory bank实现形式
        feats = th.FloatTensor(th.randn((adj.shape[0], args.embed_size)))
        self.graph_info = {"adj": adj, "feats": feats}

    def _get_input(self, loader:loader, buildGraph, dev_ratio=0.2):
        ##加载数据集
        load, bg = loader(), buildGraph()
        data_bundle = load.load(load.download(dev_ratio=dev_ratio))
        adj, index = bg.build_graph(data_bundle)
        ## 添加doc标签，以便于在图中定位文档的位置
        data_bundle.get_dataset('train').add_field('doc_id', index[0])
        data_bundle.get_dataset('dev').add_field('doc_id', index[1])
        data_bundle.get_dataset('test').add_field('doc_id', index[2])
        ## 使用bert的分词器对数据文本进行分词
        data_bundle.get_dataset('train').apply_field(lambda x: self.tokenizer(x.replace('\\', ''), truncation=True, max_length=self.arg.max_len,
                                                     padding='max_length').input_ids, 'raw_words', 'input_ids')
        data_bundle.get_dataset('dev').apply_field(
            lambda x: self.tokenizer(x.replace('\\', ''), truncation=True, max_length=self.arg.max_len,
                                     padding='max_length').input_ids, 'raw_words', 'input_ids')
        data_bundle.get_dataset('test').apply_field(lambda x: self.tokenizer(x.replace('\\', ''), truncation=True, max_length=self.arg.max_len,
                                                     padding='max_length').input_ids, 'raw_words', 'input_ids')
        #将标签映射成数值
        target_vocab = Vocabulary(padding=None, unknown=None)
        target_vocab.from_dataset(data_bundle.get_dataset('train'), field_name='target',
                                  no_create_entry_dataset=[data_bundle.get_dataset('dev'), data_bundle.get_dataset('test')])
        target_vocab.index_dataset(data_bundle.get_dataset('train'),
                                   data_bundle.get_dataset('dev'),
                                       data_bundle.get_dataset('test'), field_name='target')
        #将其设置为模型的输入和真实输出
        data_bundle.get_dataset('train').set_input('doc_id', 'input_ids')
        data_bundle.get_dataset('dev').set_input('doc_id', 'input_ids')
        data_bundle.get_dataset('test').set_input('doc_id', 'input_ids')
        data_bundle.get_dataset('train').set_target('target')
        data_bundle.get_dataset('dev').set_target('target')
        data_bundle.get_dataset('test').set_target('target')
        ##->>>>>>>>>>>>>>>>>>>>>>>>>>
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = preprocess_adj(adj)
        print(data_bundle.get_dataset('train'))
        return data_bundle, adj, target_vocab

if __name__ == '__main__':
    import argparse
    arg = argparse.ArgumentParser()
    arg.add_argument("--dataset", default="mr")
    arg.add_argument("--dev_ratio", default=0.2)
    arg = arg.parse_args()
    PrepareData(arg)
