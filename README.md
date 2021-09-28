## Introduction
本文使用fastNLP和pytorch复现了近期发表的一篇文本分类论文[BertGCN](https://arxiv.org/abs/2105.05727),
具体实现过程见[this]()。
## Results
|Model|MR|R8|R52|ohsumed|20ng|
|----|----|----|---|---|---|
|BertGCN(原文)|0.860|0.981|0.966|0.728|0.893|
|BertGCN(This repo)|0.862127|0.98127|0.963006|0.719515|0.866304

NOTE: 在训练集上运行10轮后，在测试集上的精确度。(未调参)
## Usage
+ 更改train.py中数据集的名称。
+ 运行```python trainer.py```即可。

## References
[1] [BertGCN: Transductive Text Classification by Combining GCN and BERT](https://arxiv.org/abs/2105.05727)