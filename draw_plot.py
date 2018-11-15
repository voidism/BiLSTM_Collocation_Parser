import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
import numpy as np
import itertools
import os


def plot_confusion_matrix(cm, classes, target_mat, bar, bar_target,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="Blues", sv=False, sg=0):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fs = 15-int(0.5*len(classes))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    size_scale = float(len(classes)) / 20.0
    size = (16*size_scale, 8*size_scale) if size_scale>1 else (16,8)
    plt.figure(figsize=size) 
    font_path = './bkai00mp.ttf'
    prop = mfm.FontProperties(fname=font_path)
    plt.subplot(1,2,2)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,
               fontproperties=prop)
    plt.yticks(tick_marks, classes,
               fontproperties=prop)

    fmt = '.4f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        cor = "white" if cm[i, j] > thresh else "black"
        if target_mat[i][j] == 1:
            cor = 'red'
        elif i == j:
            cor = 'green'
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=fs,
                 color=cor)

    plt.tight_layout()
    plt.ylabel('Center word')
    plt.xlabel('Context word')
    plt.subplot(1,2,1)
    plt.imshow(bar, interpolation='nearest', cmap=cmap)
    for i, j in itertools.product(range(bar.shape[0]), range(bar.shape[1])):
        cor = "white" if bar[i, j] > thresh else "black"
        if bar_target[i][j] == 1:
            cor = 'red'
        plt.text(j, i, format(bar[i, j], fmt),
                 horizontalalignment="center", fontsize=fs,
                 color=cor)
    plt.yticks(tick_marks, classes,
               fontproperties=prop)
    if sv:
        emb = 'skip_gram' if sg else 'CBOW'
        if not os.path.exists('./dependency_graphs'):
            os.makedirs('./dependency_graphs')
        plt.savefig('./dependency_graphs/{}_{}.png'.format(''.join(classes), emb))
    else:
        plt.show()
    plt.clf()