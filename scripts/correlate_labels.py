from graphnmr import *
import os, sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 4:
    print('write_labels.py [embeddings] [record_info] [records]')
    exit()

def plot_fit(fit_labels, fit_predict, title):
    plot_suffix = '.png'
    rmsd = np.sqrt(np.mean((fit_labels - fit_predict)**2))
    mae = np.mean(np.abs(fit_labels - fit_predict))
    corr = np.corrcoef(fit_labels, fit_predict)[0,1]
    N = len(fit_labels)
    plt.figure(figsize=(5,4))
    plt.scatter(fit_labels, fit_predict, marker='o', s=6, alpha=0.5, linewidth=0)
    # take top 1% for upper bound
    mmax = np.quantile(fit_labels, q=[0.99] )[0] * 1.2
    plt.plot([0,mmax], [0,mmax], '-', color='gray')
    plt.xlim(0, mmax)
    plt.ylim(0, mmax)
    plt.xlabel('Measured Shift [ppm]')
    plt.ylabel('Measured Shift [ppm]')
    plt.savefig(title + '-nostats' + plot_suffix, dpi=300)
    plt.title(title + ': RMSD = {:.4f}. MAE = {:.4f} R^2 = {:.4f}. N={}'.format(rmsd,mae, corr**2, N), fontdict={'fontsize': 8})
    plt.savefig(title + plot_suffix, dpi=300)
    plt.close()
    return {'corr-coeff': corr, 'R^2': corr**2, 'MAE': mae, 'RMSD': rmsd, 'N': N, 'title': title, 'plot': title + plot_suffix}


embedding_dicts = load_embeddings(sys.argv[1])
dup_labels = duplicate_labels(sys.argv[3], embedding_dicts, sys.argv[2])
a, b = [], []
for k,v in dup_labels.items():
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            a.append(v[i])
            b.append(v[j])

#TODO Add classes
print(plot_fit(np.array(a), np.array(b), 'self-correlation'))

