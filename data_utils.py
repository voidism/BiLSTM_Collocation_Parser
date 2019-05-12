import re

def load_data(filename="double_pairs_ys.csv", with_label=True, prefix=['a', 'b']):
    f = open(filename, 'r')
    gex = re.compile(r"([^%s%s]+)((?:\w\d)*)"%(prefix[0], prefix[1]))
    pic = re.compile(r"([%s%s][0-9])"%(prefix[0], prefix[1]))
    collection = []
    for sent in f:
        sent = sent.strip().split(' ')
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
                ret.append([arc[num][prefix[0]], arc[num][prefix[1]]])
        collection.append([ret, sent, 1] if ret!=[] else [ret, sent, 0])
    return collection if with_label else [x[1] for x in collection]