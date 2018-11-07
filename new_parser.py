import re
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data
from gensim import models
import numpy as np
import math
import argparse
from jexus import Clock, Loader
from ELMoForManyLangs import elmo
from random import shuffle

class Embedder():
    def __init__(self):
        self.embedder = elmo.Embedder()

    def __call__(self, sents):
        seq_lens = np.array([len(x) for x in sents])
        max_len = seq_lens.max()
        emb_list = self.embedder.sents2elmo(sents)
        for i in range(len(emb_list)):
            emb_list[i] = np.concatenate([emb_list[i], np.zeros((max_len - seq_lens[i], emb_list[i].shape[1]))])
        return np.array(emb_list), seq_lens

def par(sent):
    if '"' in sent:
        return ""
    spt = sent.split(',')
    label = spt[1]
    sent = spt[0]
    sent = sent.split(' ')
    gex = [re.compile(r'\[1\]\{(\S*)\}'), re.compile(r'\[2\]\{(\S*)\}')]
    pos = [0, 0]
    for i, word in enumerate(sent):
        for idx in [0, 1]:
            found = gex[idx].findall(word)
            if len(found) != 0:
                pos[idx] = i
                sent[i] = found[0]
    return pos, sent, int(label)

def get_label_data(with_label=False):
    f = open("NfN.csv", encoding='utf8').read().split('\n')[1:-1]
    li = []
    for i in f:
        parsed = par(i)
        if parsed != '':
            li.append(parsed[1] if not with_label else parsed)
    return li

def split_valid(X, v_size=0.05, rand=True):
    if rand == True:
        randomize = np.arange(len(X[0]))
        np.random.shuffle(randomize)
        X = [np.array(x)[randomize] for x in X]

    t_size = math.floor(len(X[0]) * (1 - v_size))
    X_v = []
    for i in range(len(X)):
        X_v.append(X[i][t_size:])
        X[i] = X[i][:t_size]
    
    return X, X_v

def split_valid_list(X, v_size=0.05, rand=True):
    if rand == True:
        shuffle(X)

    t_size = math.floor(len(X) * (1 - v_size))
    X_v = X[t_size:]
    X = X[:t_size]
    
    return X, X_v
# def split_valid(X, v_size=0.05, rand=True):
#     if rand == True:
#         randomize = np.arange(len(X[0]))
#         np.random.shuffle(randomize)
#         X = [np.array(x)[randomize] for x in X]

#     t_size = math.floor(len(X[0]) * (1 - v_size))
#     X_v = []
#     for i in range(len(X)):
#         X_v.append(torch.from_numpy(X[i][t_size:]))
#         X[i] = torch.from_numpy(X[i][:t_size])
    
#     return X, X_v

def sort_by(li, piv=2,unsort=False):
    li[piv], ind = torch.sort(li[piv], dim=0, descending=(not unsort))
    for i in range(len(li)):
        if i == piv:
            continue
        else:
            li[i] = li[i][ind]
    return li, ind

class Parser(nn.Module):
    def __init__(self, input_size=100, hidden_size=300, h_size=500, n_layers=3, dropout=0.33):
        super(Parser, self).__init__()
        self.gpu = torch.cuda.is_available()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = 32

        print("loading ELMo model ...")
        self.elmo = Embedder()
        print("ELMo model loaded!")
        # print("loading word2vec model ...")
        # self.word2vec = models.Word2Vec.load('../Dependency_Analyser/skip-gram/word2vec.model')
        # print("word2vec model loaded!")
        # syn0 = np.concatenate((np.zeros((1, self.word2vec.wv.syn0.shape[1])),self.word2vec.wv.syn0), axis=0)
        # self.embedding = nn.Embedding(syn0.shape[0], syn0.shape[1])
        # self.embedding.weight.data.copy_(torch.from_numpy(syn0))
        # self.embedding.weight.requires_grad = False

        self.gru = nn.LSTM(input_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True, batch_first=True)
        self.mlp_root = nn.Linear(hidden_size * 2, h_size)
        self.root_prob = nn.Linear(h_size,1)
        self.mlp_nf = nn.Linear(hidden_size*2,h_size)
        self.mlp_na = nn.Linear(hidden_size*2,h_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # output: torch.Size([sentence len, batch size, 600])

    def multi_target_loss(self, preds, labels, bs):
        t = labels.view(bs, -1)
        y = preds.view(bs, -1)
        return -torch.mean(torch.sum(t*torch.log(y) + (1-t)*torch.log(1-y), dim=1))

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        root_prob = nn.Sigmoid()(self.root_prob(nn.LeakyReLU(0.1)(self.mlp_root(outputs))))
        nf_vec = nn.Sigmoid()(nn.LeakyReLU(0.1)(self.mlp_nf(outputs)))
        na_vec = nn.Sigmoid()(nn.LeakyReLU(0.1)(self.mlp_na(outputs)))

        
        na_vec = na_vec.transpose(1,2)

        graph = nn.Softmax(dim=2)(torch.matmul(nf_vec, na_vec))
        
        # output: torch.Size([sentence len, batch size, 600])
        return root_prob, graph
    def glorot_weight_zero_bias(self):
        """
        Initalize parameters of all modules
        by initializing weights with glorot  uniform/xavier initialization,
        and setting biases to zero.
        Weights from batch norm layers are set to 1.
        
        Parameters
        ----------
        model: Module
        """
        for module in self.modules():
            if hasattr(module, 'weight'):
                if not ('BatchNorm' in module.__class__.__name__):
                    nn.init.xavier_uniform(module.weight, gain=1)
                else:
                    nn.init.constant(module.weight, 1)
            if hasattr(module, 'bias'):
                if module.bias is not None:
                    try:
                        nn.init.constant(module.bias, 0)
                    except:
                        print("init fail for",module.bias)
    def get_data(self):
        raw_data = get_label_data(1)
        x1 = np.array([i[0] for i in raw_data])
        x2 = [i[1] for i in raw_data]
        x3 = np.array([i[2] for i in raw_data])

        max_len = 0
        for i in x2:
            if len(i) > max_len:
                max_len = len(i)
        x2_lens = []
        for i in range(len(x2)):
            x2_i_len = len(x2[i])
            x2_lens.append(x2_i_len)
            for j in range(max_len):
                if j < x2_i_len:
                    try:
                        x2[i][j] = self.word2vec.wv.vocab[x2[i][j]].index + 1
                    except:
                        x2[i][j] = 0
                else:
                    x2[i].append(0)

        return [x1, np.array(x2), np.array(x2_lens), np.array(x3 == 1, dtype=int)]
        
    def get_data_for_elmo(self):
        raw_data = get_label_data(1)
        # x1 = np.array([i[0] for i in raw_data])
        # x2 = [i[1] for i in raw_data]
        # x3 = np.array([i[2] for i in raw_data])
        return raw_data#[x1, x2, np.array(x3==1,dtype=int)]
            
    # def get_dataloader(self, shuffle=False):
    #     X = self.get_data()
    #     X_train, X_valid = split_valid(X, rand=shuffle)
        
    #     if self.gpu:
    #         for i in range(len(X_train)):
    #             X_train[i].cuda()
    #         for i in range(len(X_valid)):
    #             X_valid[i].cuda()
    #     # .sort(key=lambda x: len(x[0].split(" ")), reverse=True)

    #     train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(*X_train),
    #                                             batch_size=self.batch_size, 
    #                                             shuffle=True)

    #     test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(*X_valid),
    #                                             batch_size=self.batch_size, 
    #                                             shuffle=False)

    #     return train_loader, test_loader
    def get_dataloader(self, shuffle=False):
        X = self.get_data_for_elmo()
        X_train, X_valid = split_valid_list(X, rand=shuffle)
        # .sort(key=lambda x: len(x[0].split(" ")), reverse=True)

        train_loader = Loader(X_train, batch_size=self.batch_size)

        test_loader = Loader(X_valid, batch_size=self.batch_size)

        return train_loader, test_loader

    def test(self, arcs, seqs, seq_len, label):
        total = 0
        root_corr = 0
        root, graph = self.forward(seqs, seq_len)
        nf = torch.argmax(root, dim=1).squeeze()
        na = torch.argmax(graph[range(len(arcs)), arcs[:, 0]], dim=1)
        
        total = (label==1).sum().item()
        nf_corr = (nf[(label==1).nonzero().squeeze(1)] == arcs[:, 0][(label==1).nonzero().squeeze(1)]).sum().item()
        na_corr = (na[(label==1).nonzero().squeeze(1)] == arcs[:, 1][(label==1).nonzero().squeeze(1)]).sum().item()
        return total, nf_corr, na_corr

    def load_model(self):
        self.load_state_dict(torch.load('model.ckpt'))
        print("model.ckpt load!")

        # root[(label==1).nonzero().squeeze(1)], arcs[:, 0][(label==1).nonzero().squeeze(1)].unsqueeze(1)
    def evaluate(self, raw_sents, print_out=True, dev=False, one=False):
        self.to(self.device)
        max_len = 0
        for i in raw_sents:
            if len(i) > max_len:
                max_len = len(i)
        raw_sents_lens = []
        # backup_sents = raw_sents[:]
        new_sents = []
        for i in range(len(raw_sents)):
            raw_sents_i_len = len(raw_sents[i])
            raw_sents_lens.append(raw_sents_i_len)
            new_sents.append([])
            for j in range(max_len):
                if j < raw_sents_i_len:
                    try:
                        new_sents[-1].append(self.word2vec.wv.vocab[raw_sents[i][j]].index + 1)
                    except:
                        new_sents[-1].append(0)
                else:
                    new_sents[-1].append(0)

        X, X_len = torch.from_numpy(np.array(new_sents)), torch.from_numpy(np.array(raw_sents_lens))
        if self.gpu:
            X.cuda()
            X_len.cuda()
        eval_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X, X_len),
                                                batch_size=512, 
                                                shuffle=False)
        nf_list = []
        na_list = []
        na_prob = []
        nf_prob = []
        with torch.no_grad():
            for i, li in enumerate(eval_loader):
                [seqs, seq_len], ind = sort_by(li, piv=1)
                [seqs, seq_len] = [seqs.to(self.device), seq_len.to(self.device)]
                this_bs = seqs.shape[0]
                # Forward pass
                root, graph = self.forward(seqs, seq_len)
                [seqs, seq_len, root, graph, ind], ind2 = sort_by([seqs, seq_len, root, graph, ind], 4, True)
                if dev:
                    return root, graph
                nf = torch.argmax(root, dim=1).squeeze()
                na = torch.argmax(graph[range(this_bs), nf], dim=1)
                nf_list.append(np.array(nf))
                na_list.append(np.array(na))
                root = np.array(root, dtype = float)
                graph = np.array(graph, dtype = float)
                nf_prob.append(root[range(len(root)),nf])
                na_prob.append(graph[range(len(graph)), nf, na])
                # return graph, root, graph[range(len(graph)), nf, na], root[range(len(root)),nf]
        nf_all = np.concatenate(nf_list,axis=0)
        na_all = np.concatenate(na_list, axis=0)
        nf_prob = np.concatenate(nf_prob, axis=0)
        na_prob = np.concatenate(na_prob, axis=0)
        nfna = np.concatenate((nf_all[:,np.newaxis], na_all[:,np.newaxis]), axis=1)
        if print_out:
            idx = 0
            for [f, a], sent in zip(nfna, raw_sents):
                li = []
                for i, word in enumerate(sent):
                    if i == f:
                        li.append("[1]{%s}(%.2f)"%(word, nf_prob[idx]))
                    elif i == a:
                        li.append("[2]{%s}(%.2f)"%(word, na_prob[idx]))
                    else:
                        li.append(word)
                print(' '.join(li))
                idx += 1
                if one:
                    break
        else:
            return nfna




    def train(self, num_epochs=50, dev=False, rand=False):
        self.to(self.device)
        train_loader, test_loader = self.get_dataloader()
        # Train the model
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            ct = Clock(len(train_loader),title="Epoch %d/%d"%(epoch, num_epochs))
            ac_loss = 0
            num = 0
            nf_correct = 0
            na_correct = 0
            test_total = 0
            for i, li in enumerate(train_loader):
                [arcs, seqs, seq_len, label], ind = sort_by(li, piv=2)
                [arcs, seqs, seq_len, label] = [arcs.to(self.device), seqs.to(self.device), seq_len.to(self.device), label.to(self.device)]
                this_bs = arcs.shape[0]
                # Forward pass
                root, graph = self.forward(seqs, seq_len)
                ans = torch.zeros_like(graph)
                for i in range(len(arcs)):
                    ans[i,arcs[i][0],arcs[i][1]] = label[i]
                loss_1 = self.criterion(root[(label==1).nonzero().squeeze(1)], arcs[:, 0][(label==1).nonzero().squeeze(1)].unsqueeze(1))
                loss_2 = 0.1 * self.multi_target_loss(graph, ans, bs=this_bs)
                # return loss_1, loss_2
                # print(lossnum)
                loss = loss_1 + loss_2
                ac_loss += loss.item()
                num += 1
                if dev:
                    return root, graph, [arcs, seqs, seq_len, label], ans
                # Backward and optimize
                ct.flush(info={'loss':ac_loss/num})
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            
            with torch.no_grad():
                nf_correct = 0
                na_correct = 0
                test_total = 0
                for i, li in enumerate(test_loader):
                    [arcs, seqs, seq_len, label], ind = sort_by(li, piv=2)
                    [arcs, seqs, seq_len, label] = [arcs.to(self.device), seqs.to(self.device), seq_len.to(self.device), label.to(self.device)]
                    t, f, a = self.test(arcs, seqs, seq_len, label)
                    test_total += t
                    nf_correct += f
                    na_correct += a
                ct.flush(info={'loss':ac_loss/num, 'accNf':nf_correct/test_total, 'accNa':na_correct/test_total})
                

        # Test the model

        #     print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

        # Save the model checkpoint
        torch.save(self.state_dict(), 'model.ckpt')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pos1", help="positional argument 1")
    args = parser.parse_args()
    p = Parser()
    if args.pos1=='dev':
        a = torch.from_numpy(np.ones((32, 10), dtype=np.int64))
        seq_len = np.full((32,),10,dtype=np.int64)
        r = p.forward(a, seq_len)
        root, graph, [arcs, seqs, seq_len, label], ans = p.train(dev=True)
        # loss_1, loss_2 = p.train()
    elif args.pos1 == 'train':
        p.train()
    elif args.pos1 == 'eval':
        raw_data = get_label_data()
        t_size = math.floor(len(raw_data)*0.95)
        p.load_model()
        p.evaluate(raw_data[t_size:])
    elif args.pos1 == 'demo':
        from CKIP import PyWordSeg
        s = PyWordSeg()
        p.load_model()
        while True:
            e = input("sent> ")
            l = s.Segment(e, 'wordlist')
            print(l)
            p.evaluate([l,l], one=True)

        
