from fastNLP import Tester, Trainer, CrossEntropyLoss, AccuracyMetric,  EarlyStopCallback, LRScheduler
from data_loader import PrepareData
from model import BertGCN
import argparse
import torch as th

parse = argparse.ArgumentParser()

parse.add_argument('--dataset', default="mr", help="[mr, 20ng, R8, R52, ohsumed]")
parse.add_argument("--embed_size", default=768)
parse.add_argument("--gcn_hidden_size", default=256)
# parse.add_argument("--cls_type", default=2)
parse.add_argument("--devices_gpu", default=[0, 1, 2])
parse.add_argument("--lr", default=2e-5, help="learning rate")
parse.add_argument("--bert_lr", default=2e-5)
parse.add_argument("--gcn_lr", default=2e-3)
parse.add_argument("--batch_size", default=32)
parse.add_argument("--max_len", default=128)
parse.add_argument("--p", default=0.3)
parse.add_argument("--pretrained_model", default='bert-base-uncased', help="['bert-base-uncased', 'roberta-base']")
parse.add_argument("--nb_epoch", default=10)
parse.add_argument("--dropout", default=0.5)
parse.add_argument("--dev_ratio", default=0.2)
arg = parse.parse_args()
device = th.device("cuda:2")
## PrePareData
print("Data Loading")
pd = PrepareData(arg)
pd.graph_info['feats'] = pd.graph_info['feats'].to(device)
pd.graph_info['adj'] = pd.graph_info['adj'].to(device)
arg.cls_type = len(pd.target_vocab)

### Load Model
print("Load Model")
model = BertGCN(arg.pretrained_model, arg.cls_type, arg.gcn_hidden_size,
                arg.p, arg.dropout, pd.graph_info)

optim = th.optim.Adam([
                        {'params': model.gcn.parameters(), 'lr': arg.gcn_lr},
                        {'params': model.bert_model.parameters(), 'lr': arg.bert_lr},
                        {'params': model.clssifier.parameters(), 'lr': arg.bert_lr},
                      ], lr=arg.lr)

scheduler = th.optim.lr_scheduler.MultiStepLR(optim, milestones=[30], gamma=0.1)
# cost = th.nn.CrossEntropyLoss(ignore_index=True)
callback = [EarlyStopCallback(10), LRScheduler(scheduler)]

trainer = Trainer(pd.data_bundle.get_dataset('train'), model, loss=CrossEntropyLoss(target='target'),
                   optimizer=optim, n_epochs=arg.nb_epoch, device=device, callbacks=callback,
                   batch_size=arg.batch_size,
                   dev_data=pd.data_bundle.get_dataset('dev'), metrics=AccuracyMetric(target='target'))

trainer.train()

tester = Tester(pd.data_bundle.get_dataset('test'), model, metrics=AccuracyMetric(target='target'),
       device=device)

tester.test()