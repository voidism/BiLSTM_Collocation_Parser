import time
import os
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import datetime

class Clock:
    def __init__(self, epoch, update_rate=50, flush_rate=1, title=""):
        self.title = title
        self.start_time = time.time()
        self.iteration = epoch
        self.rate = update_rate
        self.rem_time = 0
        self.pass_time = 0
        self.info_dict = {}
        self.last_txt_len = 0
        self.idx = 0
        self.flush_rate = flush_rate
        self.enter = os.fstat(0) != os.fstat(1)
        self.prior_dict = {"loss":0, "acc":1, "val_loss":2, "val_acc":3}

    def set_start(self):
        self.start_time = time.time()

    def set_total(self, epoch):
        self.iteration = epoch

    def info2str(self, info):
        #sort dict
        info_list = [(key, info[key]) for key in sorted(info.keys(), key=lambda d: self.prior_dict[d] if d in self.prior_dict else 4, reverse=False)]
        info_str = ''
        for i,j in info_list:
            txt = str(round(float(j), 4))# if type(info[i])==float else str(info[i])
            info_str += ' ' + str(i) + ': '
            info_str += txt
            info_str += ' '*(6-len(txt))
        return info_str

    def flush(self, info={}):
        print_txt = ""
        if self.idx != self.iteration - 1:
            print_txt = "\tETA: "+str(round(self.rem_time, 0)) + \
            " s" + self.info2str(info)
        else:
             print_txt = "\tALL: "+str(round(time.time() - self.start_time, 0)) + \
            " s" + self.info2str(info)
        if self.idx == 0 and self.title != "":
            self.last_txt_len = len(print_txt)
            print("\n [ "+self.title+" ] ")
        elif self.idx % self.rate == 1:
            self.pass_time = time.time() - self.start_time
            self.rem_time = self.pass_time * \
                (self.iteration - self.idx) / self.idx
        
        if self.idx % self.flush_rate != 0:
            pass
        else:
            print(
                chr(13) + "|" + "=" * (50 * self.idx // self.iteration
                                    ) + ">" + " " * (50 * (self.iteration - self.idx) // self.iteration
                                                        ) + "| " + str(
                    round(100 * self.idx / self.iteration, 1)) + "%",
                #"\tave cost: "+str(round(cost, 2)) if cost != 0 else "",
                print_txt+' '*(self.last_txt_len - len(print_txt)),
                sep=' ', end='', flush=True)
            self.last_txt_len = len(print_txt)
            if self.idx == self.iteration-1:
                print("")
            if self.enter:
                print("")
        self.idx += 1

class Loader():
    def __init__(self, li, batch_size=32):
        self.li = li
        self.batch_size = batch_size
        self.idx = 0
    def __iter__(self):
        return self
    def __len__(self):
        return math.floor(len(self.li)/self.batch_size)
    def __next__(self):
        if self.idx >= len(self.li):
            self.idx = 0
            raise StopIteration
        old_idx = self.idx
        self.idx += self.batch_size
        return self.li[old_idx:self.idx]

class History():
    def __init__(self, title, xlabel, ylabel, item_name=["loss", "train_accNf", "train_accNa", "test_accNf", "test_accNa"]):
        self.history = [[[], []] for _ in range(len(item_name))]
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.names = item_name

    def append_history(self, item_idx, item):# item = (step_num, value)
        self.history[item_idx][0].append(item[0])
        self.history[item_idx][1].append(item[1])

    def plot(self):
        plt.title(self.title)
        for i,trace in enumerate(self.history):
            plt.plot(trace[0], trace[1], label=self.names[i])
        plt.legend()

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.show()
        string = self.title+'_'+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        plt.savefig(string+'.png')