import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
import numpy as np
import itertools
import os
import re


def plot_confusion_matrix(cm, classes, target_mat, bar, bar_target, save_dir,
                          normalize=False,
                          title='Confusion matrix',
                          cmap="Blues", sv=False, sub_list="<>:'\"/\\|?*"):
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
    substrings = sorted(list(sub_list), key=len, reverse=True)
    gex = re.compile('|'.join(map(re.escape, substrings)))
    img_name = gex.sub('_',''.join(classes))
    if sv:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, '{}.png'.format(img_name)))
    else:
        plt.show()
    plt.clf()