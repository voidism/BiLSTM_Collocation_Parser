import re, collections
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data
import numpy as np
import math
import argparse
from jexus import Clock, Loader, History
from ELMoForManyLangs import elmo
from random import shuffle
from draw_plot import *
from data_utils import load_data
import math
import sys


class Embedder():
    def __init__(self):
        self.embedder = elmo.Embedder(batch_size=512)

    def __call__(self, sents):
        seq_lens = np.array([len(x) for x in sents], dtype=np.int64)
        sents = [[self.sub_unk(x) for x in sent] for sent in sents]
        max_len = seq_lens.max()
        emb_list = self.embedder.sents2elmo(sents)
        for i in range(len(emb_list)):
            emb_list[i] = np.concatenate([emb_list[i], np.zeros((max_len - seq_lens[i], emb_list[i].shape[1]))])
        return np.array(emb_list, dtype=np.float32), seq_lens

    def sub_unk(self, e):
        e = e.replace('，',',')
        e = e.replace('：',':')
        e = e.replace('；',';')
        e = e.replace('？','?')
        e = e.replace('！', '!')
        return e

def calc_overlap(a, b):  #a:True Class, b:Pred Class
    # if a.shape == torch.Size([]):
    #     a = [a.item()]
    # if b.shape == torch.Size([]):
    #     b = [b.item()]
    try:
        a = list(a)
        b = list(b)
    except:
        print("a: ",a)
        print("b: ",b)
    a_multiset = collections.Counter(a)
    b_multiset = collections.Counter(b)
    overlap = list((a_multiset & b_multiset).elements())
    a_remainder = list((a_multiset - b_multiset).elements())
    b_remainder = list((b_multiset - a_multiset).elements())
    o, a, b = len(overlap), len(a_remainder), len(b_remainder)
    if a == 0 and b == 0:
        return 1, 1
    elif a == 0 and b != 0:
        return 1, 1 / (1 + b)
    elif a != 0 and b == 0:
        return 1 / (1 + a), 1
    precision = o / (o + b)
    recall = o / (o + a)
    return precision, recall


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
    # return [pos], sent, int(int(label)==1)
    ret = ([pos], sent, 1) if (int(label)==1) else([], sent, 0)
    return ret

def get_label_data(filename="NfN.csv", with_label=False):
    f = open(filename).read().split('\n')[1:-1]
    li = []
    for i in f:
        parsed = par(i)
        if parsed != '':
            li.append(parsed[1] if not with_label else parsed)
    return li

def get_multi_data(filename="double_pairs_ys.csv", with_label=True):
    f = open(filename)
    f = f.read().split('\n')[:-1]
    f = [x.replace('{', '') for x in f]
    f = [x.replace('}', '') for x in f]
    f = [x.split(' ') for x in f]
    gex = re.compile(r"([^nf]+)((?:\w\d)*)")
    pic = re.compile(r"([nf][0-9])")
    collection = []
    for sent in f:
        arc = {}
        ret = []
        for i, word in enumerate(sent):
            if word == '':
                sent.pop(i)
                continue
            try:
                gexf = gex.findall(word)[0]
            except:
                if word in prefix:
                    gexf = (word, '')
                else:
                    print(sent)
                    break
            if gexf[1] == '':
                continue
            else:
                sent[i] = gexf[0]
                points = pic.findall(gexf[1])
                for point in points:
                    arc.setdefault(point[1], {})[point[0]] = i
        for num in arc.keys():
            if not len(arc[num].keys()) < 2:
                ret.append([arc[num]['f'], arc[num]['n']])
        collection.append([ret, sent, 1] if ret!=[] else [ret, sent, 0])
    return collection[:2000] if with_label else [x[1] for x in collection[:2000]]


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


def tag_mat(mat):
    target = np.zeros_like(mat)
    if mat.shape[0] == 1:
        return target
    for i in range(mat.shape[0]):
        rank = np.argsort(-mat[i])
        top = rank[0] if rank[0] != i else rank[1]
        for x in range(mat.shape[1]):
            boo = int(x == top)
            if boo:
                target[i][x] = 1
    return target
        
def tag_array(mat):
    target = np.zeros_like(mat)
    if mat.shape[0] == 1:
        return target
    top = np.argsort(-mat[:,0])[0]
    target[top][0] = 1
    return target


def f1_score(p, r):
    return 2*p*r/(p+r) if p+r != 0 else 0


def sort_by(li, piv=2,unsort=False):
    li[piv], ind = torch.sort(li[piv], dim=0, descending=(not unsort))
    for i in range(len(li)):
        if i == piv:
            continue
        else:
            li[i] = li[i][ind]
    return li, ind

def sort_list(li, piv=2,unsort_ind=None):
    # li[piv], ind = torch.sort(li[piv], dim=0, descending=(not unsort))
    ind = []
    if unsort_ind == None:
        ind = sorted(range(len(li[piv])), key=(lambda k: li[piv][k]))
    else:
        ind = unsort_ind
    
    for i in range(len(li)):
        li[i] = [li[i][j] for j in ind]
    return li, ind

class Parser(nn.Module):
    def __init__(self, input_size=1024, hidden_size=300, h_size=500, n_layers=3, dropout=0.33, batch_size=32, cpu_only=False):
        super(Parser, self).__init__()
        self.gpu = torch.cuda.is_available()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        print("loading ELMo model ...", file=sys.stderr)
        self.elmo = Embedder()
        print("ELMo model loaded!", file=sys.stderr)
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
        self.bce = nn.BCELoss()
        # output: torch.Size([sentence len, batch size, 600])

    def multi_target_loss(self, preds, labels):
        bs = preds.shape[0]
        t = labels.view(bs, -1)
        y = preds.view(bs, -1)
        return nn.BCELoss()(y, t)
        # return - torch.mean(torch.sum(t * torch.log(y) + (1 - t) * torch.log(1 - y), dim=1))
        
    def multi_target_loss_root(self, preds, labels):
        bs = preds.shape[0]
        t = labels.view(bs, -1)
        y = preds.view(bs, -1)
        return nn.BCELoss()(y, t)

    def forward(self, input_seq, input_lengths, hidden=None):
        # embedded = self.embedding(input_seq)
        embedded = input_seq
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
            
    def get_dataloader(self, filename, shuffle=True, prefix=['a', 'b']):
        X = load_data(filename=filename, prefix=prefix)
        X_train, X_valid = split_valid_list(X, rand=shuffle)
        # .sort(key=lambda x: len(x[0].split(" ")), reverse=True)

        train_loader = Loader(X_train, batch_size=self.batch_size)

        test_loader = Loader(X_valid, batch_size=self.batch_size)

        return train_loader, test_loader

    def expand_data(self, label_data):
        # arcs = np.array([x[0] for x in label_data])
        arcs_array = []
        for group in label_data:
            if group[0] == []:
                z = np.zeros((10, 2), dtype=int)
                arcs_array.append(z)
            else:
                array = np.array(group[0])
                z = np.zeros((10, 2), dtype=int)
                z[:len(array)] = array
                arcs_array.append(z)
        arcs = np.array(arcs_array)
        labels = np.array([x[2] for x in label_data])
        seqs = [x[1] for x in label_data]
        embedded, seq_lens = self.elmo(seqs)
        return [torch.from_numpy(arcs), torch.from_numpy(embedded), torch.from_numpy(seq_lens), torch.from_numpy(labels)]

    def expand_unlabelled(self, sents):
        embedded, seq_lens = self.elmo(sents)
        return [torch.from_numpy(embedded), torch.from_numpy(seq_lens)]

    def multi_acc(self, arcs, root, graph, label, threshold=0.5):
        na_r = 0
        na_p = 0
        na_total = 0
        nf_r = 0
        nf_p = 0
        nf_total = 0
        for a, ro, g in zip(arcs, root, graph):
            root_ans = []
            nf_na_di = {}
            for row in a:
                if row[0] == row[1]:
                    break
                else:
                    root_ans.append(row[0].item())
                    nf_na_di.setdefault(row[0].item(), [])
                    nf_na_di[row[0].item()].append(row[1].item())
            pred_nf = (ro > threshold).nonzero().squeeze()
            pred_nf = [pred_nf.item()] if pred_nf.dim()==0 else pred_nf
            p, r = calc_overlap(root_ans, pred_nf)
            nf_total += 1
            nf_p += p
            nf_r += r
            for nf in nf_na_di:
                na = nf_na_di[nf]
                pred_na = (g[nf] > threshold).nonzero().squeeze()
                pred_na = [pred_na.item()] if pred_na.dim()==0 else pred_na
                p, r = calc_overlap(na, pred_na)
                na_total += 1
                na_p += p
                na_r += r
        # return: nf_precision, nf_recall, na_precision, na_recall
        return {"a":nf_p/nf_total, "b":nf_r/nf_total, "c":na_p/na_total, "d":na_r/na_total}
        # return {"nf_p":nf_p/nf_total, "nf_r":nf_r/nf_total, "na_p":na_p/na_total, "na_r":na_r/na_total}
        # return nf_p/nf_total, nf_r/nf_total, na_p/na_total, na_r/na_total


    def test(self, arcs, seqs, seq_len, label):
        # self.eval()
        total = 0
        root_corr = 0
        root, graph = self.forward(seqs, seq_len)
        nf = torch.argmax(root, dim=1).squeeze()
        na = torch.argmax(graph[range(len(arcs)), arcs[:, 0]], dim=1)
        
        total = (label==1).sum().item()
        nf_corr = (nf[(label==1).nonzero().squeeze(1)] == arcs[:, 0][(label==1).nonzero().squeeze(1)]).sum().item()
        na_corr = (na[(label==1).nonzero().squeeze(1)] == arcs[:, 1][(label==1).nonzero().squeeze(1)]).sum().item()
        return total, nf_corr, na_corr

    def acc(self, arcs, root, graph, label):
        total = 0
        root_corr = 0
        nf = torch.argmax(root, dim=1).squeeze()
        na = torch.argmax(graph[range(len(arcs)), arcs[:, 0]], dim=1)
        
        total = (label == 1).sum().item()
        wanted_idxs = (label==1).nonzero().squeeze(1)
        nf_corr = (nf[wanted_idxs] == arcs[:, 0][wanted_idxs]).sum().item()
        na_corr = (na[wanted_idxs] == arcs[:, 1][wanted_idxs]).sum().item()
        return total, nf_corr, na_corr

    def load_model(self, filename='model.ckpt'):
        self.load_state_dict(torch.load(filename))
        print("%s load!"%filename)

        # root[(label==1).nonzero().squeeze(1)], arcs[:, 0][(label==1).nonzero().squeeze(1)].unsqueeze(1)
    def evaluate(self, raw_sents, threshold=0.4, print_out=True, dev=False, prefix=["f", "n"], file=None, show_score=True):
        self.to(self.device)
        self.eval()
        max_len = 0
        for i in raw_sents:
            if len(i) > max_len:
                max_len = len(i)
        # backup_sents = raw_sents[:]
        embedded, seq_lens = self.elmo(raw_sents)

        X, X_len = torch.from_numpy(embedded), torch.from_numpy(seq_lens)
        eval_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X, X_len),
                                                batch_size=512, 
                                                shuffle=False)
        nf_list = []
        na_list = []
        na_prob = []
        nf_prob = []
        if type(threshold) is list:
            assert len(threshold) == 2
        else:
            assert type(threshold) is float
            threshold = [threshold, threshold]

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
                nf = (root>threshold[0]).squeeze()
                na = torch.argmax(graph, dim=-1)
                nf_list.append(np.array(nf.cpu()))
                na_list.append(np.array(na.cpu()))
                root = np.array(root.cpu(), dtype = float)
                graph = np.array(graph.cpu(), dtype = float)
                nf_prob.append(root)
                na_prob.append(graph)
        if len(nf_list)>1:
            nf_list = np.concatenate(nf_list,axis=0)
            na_list = np.concatenate(na_list, axis=0)
            nf_prob = np.concatenate(nf_prob, axis=0)
            na_prob = np.concatenate(na_prob, axis=0)
        else:
            nf_list = nf_list[0]
            na_list = na_list[0]
            nf_prob = nf_prob[0]
            na_prob = na_prob[0]
        if print_out:
            idx = 0
            # seg = CKIP.PyWordSeg()
            gex = re.compile(r"[\u4E00-\u9FFF]+")
            for idx, (nf, na, sent) in enumerate(zip(nf_list, na_list, raw_sents)):
                nf_id = 1
                for i, (word, f, a) in enumerate(zip(sent, nf, na)):
                    if f > 0:
                        # assert len(pos_tag) == len(sent)
                        assert sent == raw_sents[idx]
                        if i == a:
                            continue
                        try:
                            if len(gex.findall(sent[i])) == 0 or len(gex.findall(sent[a])) == 0:
                                continue
                            if na_prob[idx][i][a] < threshold[1]:
                                continue
                            if not show_score:
                                raw_sents[idx][i] = "%s%s%d" % (word, prefix[0], nf_id)
                                raw_sents[idx][a] = "%s%s%d" % (raw_sents[idx][a], prefix[1], nf_id)
                            else:
                                raw_sents[idx][i] = "%s%s%d(%.2f)" % (word, prefix[0], nf_id, nf_prob[idx][i])
                                raw_sents[idx][a] = "%s%s%d(%.2f)" % (raw_sents[idx][a], prefix[1], nf_id, na_prob[idx][i][a])
                            nf_id += 1
                        except Exception as ex:
                            pass
                if file!=None:
                    print(' '.join(raw_sents[idx]), file=file)
                else:
                    print(' '.join(raw_sents[idx]))
        else:
            return nf_list, na_list

    def visualize(self, raw_sents, out_dir, ct=None):
        self.to(self.device)
        self.eval()
        X, X_len = self.elmo(raw_sents)
        X = torch.from_numpy(X)
        X_len = torch.from_numpy(X_len)
        if self.gpu:
            X.cuda()
            X_len.cuda()
        eval_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X, X_len),
                                                batch_size=512, 
                                                shuffle=False)
        nf_list = []
        na_list = []
        root_list = []
        graph_list = []
        with torch.no_grad():
            self.eval()
            for i, li in enumerate(eval_loader):
                [seqs, seq_len] = li
                [seqs, seq_len], ind = sort_by([seqs, seq_len], piv=1)
                [seqs, seq_len] = [seqs.to(self.device), seq_len.to(self.device)]
                this_bs = seqs.shape[0]
                # Forward pass
                root, graph = self.forward(seqs, seq_len)
                [seqs, seq_len, root, graph, ind], ind2 = sort_by([seqs, seq_len, root, graph, ind], 4, True)
                # if dev:
                    # return root, graph
                for i, j, k in zip(root, graph, raw_sents):
                    s = len(k)
                    temp_root = np.array(i.cpu(), dtype = float)[:s]
                    temp_graph = np.array(j.cpu(), dtype = float)[:s, :s]
                    nf_list.append(tag_array(temp_root))
                    na_list.append(tag_mat(temp_graph))
                    root_list.append(temp_root)
                    graph_list.append(temp_graph)

        for sent, root, graph, root_tag, graph_tag in zip(raw_sents, root_list, graph_list, nf_list, na_list):
            if ct is not None:
                ct.flush()
            plot_confusion_matrix(
            graph, sent, graph_tag, root, root_tag, title='BiLSTM Collocation Parser', sv=True, save_dir=out_dir)

    def arcs_expand(self, arcs, graph_size, label):
        ans = torch.zeros((arcs.shape[0],graph_size,graph_size))
        for i, arc_array in enumerate(arcs):
            for pair in arc_array:
                if pair[0] == pair[1]:
                    break
                else:
                    ans[i, pair[0], pair[1]] = int(label[i] == 1)
        return ans.to(self.device)

    def train_model(self, train_file, save_model_name, num_epochs=10, prefix=['a', 'b']):
        self.to(self.device)
        train_loader, test_loader = self.get_dataloader(train_file, shuffle=True, prefix=prefix)
        # Train the model
        total_step = len(train_loader)
        His = History(title="TrainingCurve", xlabel="step", ylabel="f1-score", item_name=["train_Nf", "train_Na", "test_Nf", "test_Na"])
        step_idx = 0
        for epoch in range(num_epochs):
            self.train()
            ct = Clock(len(train_loader),title="Epoch %d/%d"%(epoch+1, num_epochs))
            ac_loss = 0
            num = 0
            f1ab = 0
            f1cd = 0

            for i, li in enumerate(train_loader):
                li = self.expand_data(li)
                [arcs, seqs, seq_len, label], ind = sort_by(li, piv=2)
                [arcs, seqs, seq_len, label] = [arcs.to(self.device), seqs.to(self.device), seq_len.to(self.device), label.to(self.device)]
                # this_bs = arcs.shape[0]
                # Forward pass
                root, graph = self.forward(seqs, seq_len)
                ans = self.arcs_expand(arcs, graph.shape[-1], label)
                root_ans = (ans.sum(-1) > 0).float()
                root = root.squeeze()
                # ans = torch.zeros_like(graph)
                # for i in range(len(arcs)):
                #     ans[i,arcs[i][0],arcs[i][1]] = int(label[i]==1)
                # loss_1 = self.criterion(root[(label==1).nonzero().squeeze(1)], arcs[:, 0][(label==1).nonzero().squeeze(1)].unsqueeze(1))
                if not ((root_ans >= 0).all() & (root_ans <= 1).all()).item():
                    print(root_ans)
                if not ((root >= 0).all() & (root <= 1).all()).item():
                    print(root)
                loss_1 = self.bce(root, root_ans)
                loss_2 = 0.3 * self.multi_target_loss(graph, ans)
                # return loss_1, loss_2
                # print(lossnum)
                loss = loss_1 + loss_2
                ac_loss += loss.item()
                num += 1
                # if dev:
                #     return root, graph, [arcs, seqs, seq_len, label], ans
                # total, nf_corr, na_corr = self.acc(arcs, root, graph, label)
                info_dict = self.multi_acc(arcs, root, graph, label)
                f1ab += f1_score(info_dict['a'], info_dict['b'])
                f1cd += f1_score(info_dict['c'], info_dict['d'])
                # info_dict = {'loss':ac_loss/num, 'accNf':train_nf_corr/train_total, 'accNa':train_na_corr/train_total}
                ct.flush(info=info_dict)
                step_idx += 1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            His.append_history(0, (step_idx, f1ab/num))
            His.append_history(1, (step_idx, f1cd/num))
                
            
            with torch.no_grad():
                self.eval()
                # nf_correct = 0
                # na_correct = 0
                # test_total = 0
                for i, li in enumerate(test_loader):
                    li = self.expand_data(li)
                    [arcs, seqs, seq_len, label], ind = sort_by(li, piv=2)
                    [arcs, seqs, seq_len, label] = [arcs.to(self.device), seqs.to(self.device), seq_len.to(self.device), label.to(self.device)]
                    root, graph = self.forward(seqs, seq_len)
                    root = root.squeeze()
                    info_dict = self.multi_acc(arcs, root, graph, label)
                    ct.flush(info=info_dict)
                    # t, f, a = self.test(arcs, seqs, seq_len, label)
                    # test_total += t
                    # nf_correct += f
                    # na_correct += a
                # info_dict = {'val_accNf':nf_correct/test_total, 'val_accNa':na_correct/test_total}
                # ct.flush(info={'loss':ac_loss/num, 'val_accNf':nf_correct/test_total, 'val_accNa':na_correct/test_total})
                
                His.append_history(2, (step_idx, f1_score(info_dict['a'], info_dict['b'])))
                His.append_history(3, (step_idx, f1_score(info_dict['c'], info_dict['d'])))

        # Save the model checkpoint
        torch.save(self.state_dict(), save_model_name)
        His.plot()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="execute mode")
    # For testing
    parser.add_argument("-in_file", default=None, required=False, help="input file name to parse.")
    parser.add_argument("-out_file", default=None, required=False, help="output file name to write.")
    parser.add_argument("-model_name", default=None, required=False, help="model file name to use.")
    parser.add_argument("-timing", type=str, default="True", help="show timing bar to monitor the executing time.")
    parser.add_argument("-prefix", type=str, default="A_N", help="phrase types wanted of the model, seperated by `_`. e.g. `Nf_N`")
    parser.add_argument("-show_score", type=str, default="True", help="show predicted score of each selected word in sentence.")
    parser.add_argument("-threshold", type=float, default=0.4, help="threshold to filt out low pairs with confidence")
    parser.add_argument("-threshold_2", type=float, default=None, required=False, help="threshold for the second word in a pair.")
    parser.add_argument("-batch_size", type=int, default=128, required=False, help="testing batch_size")
    parser.add_argument("-max_len", type=int, default=100, required=False, help="maximum sequence length.")
    # For visualize
    parser.add_argument("-out_folder", default=None, required=False, help="output folder to write imgs.")
    # For training
    parser.add_argument("-train_file", default=None, required=False, help="training+testing set file name.")
    parser.add_argument("-epochs", type=int, default=10, required=False, help="training number of epochs.")
    parser.add_argument("-save_model_name", type=str, default=None, required=False, help="save model file name.")
    parser.add_argument("-pretrain_name", type=str, default=None, required=False, help="load pretrain model file name.")


    args = parser.parse_args()
    p = Parser(batch_size=args.batch_size)
    thres = [args.threshold, args.threshold_2] if args.threshold_2 is not None else args.threshold
    if args.mode == 'print' or args.out_file is None:
        out_file = sys.stdout
    else:
        out_file = open(args.out_file, 'w')
    
    if args.mode in ['print', 'write']:
        prefix = args.prefix.split("_")
        assert len(prefix) == 2
        p.load_model(args.model_name)
        sents = []
        poss = []
        f = open(args.in_file)
        total_num = int(os.popen("wc -l %s" % args.in_file).read().split(' ')[0])
        ct = Clock(total_num, title="===> Parsing File %s with %d lines"%(args.in_file, total_num))
        for i in f:
            if args.timing == "True" and args.mode != 'print':
                ct.flush()
            sent_list = i.strip().split(' ')
            if len(sent_list) < 2:
                print("TOO SHORT ==>", ' '.join(sent_list), file=out_file)
                continue
            elif len(sent_list)>args.max_len:
                print("TOO LONG ==>", ' '.join(sent_list), file=out_file)
                continue
            sents.append(sent_list)
            if len(sents) >= p.batch_size:
                p.evaluate(sents, prefix=prefix, threshold=thres, file=out_file, print_out=(out_file is not None), show_score=(args.show_score=="True"))
                sents = []
        if len(sents)>0:
            p.evaluate(sents, prefix=prefix, threshold=thres, file=out_file, print_out=(out_file is not None), show_score=(args.show_score=="True"))

    if args.mode in ['draw', 'plot']:
        prefix = args.prefix.split("_")
        assert len(prefix) == 2
        p.load_model(args.model_name)
        sents = []
        poss = []
        f = open(args.in_file)
        total_num = int(os.popen("wc -l %s" % args.in_file).read().split(' ')[0])
        ct = Clock(total_num, title="===> Parsing File %s with %d lines"%(args.in_file, total_num))
        my_ct = None
        if args.timing == "True" and args.mode != 'print':
            my_ct = ct
        for i in f:
            sent_list = i.strip().split(' ')
            if len(sent_list) < 2:
                print("TOO SHORT ==>", ' '.join(sent_list), file=out_file)
                if my_ct is not None:
                    ct.flush()
                continue
            elif len(sent_list)>args.max_len:
                print("TOO LONG ==>", ' '.join(sent_list), file=out_file)
                if my_ct is not None:
                    ct.flush()
                continue
            sents.append(sent_list)
            if len(sents) >= p.batch_size:
                p.visualize(sents, out_dir=args.out_folder, ct=my_ct)
                sents = []
        if len(sents)>0:
            p.visualize(sents, out_dir=args.out_folder, ct=my_ct)

    if args.mode == "train":
        if args.pretrain_name is not None:
            p.load_model(args.pretrain_name)
        p.train_model(train_file=args.train_file, save_model_name=args.save_model_name, num_epochs=args.epochs, prefix=args.prefix.split("_"))



