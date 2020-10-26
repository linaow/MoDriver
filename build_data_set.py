import argparse
import sys
import numpy as np

np.random.seed(1)
import os, time
import math
import re
import tempfile
from tempfile import mkdtemp
from subprocess import Popen, check_output
import pandas as pd
import gzip
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import mixture
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn import linear_model, preprocessing
from sklearn.ensemble import RandomForestClassifier
from numpy import interp
import matplotlib.pyplot as plt
from os.path import splitext, basename, exists, abspath, isfile, getsize


def file2data(feature_file, coding_file, non_coding_file):
    coding_all = pd.read_csv(coding_file, header=0, sep=',',
                             usecols=['ID', 'Post-filter q-value'])
    non_coding_all = pd.read_csv(non_coding_file, header=0, sep=',',
                                 usecols=['ID', 'Post-filter q-value'])
    df_genome = pd.concat([coding_all, non_coding_all], axis=0)
    df_genome = df_genome.sort_values(by=['Post-filter q-value'], ascending=[True])
    df_pos = df_genome[df_genome['Post-filter q-value'] < 0.1]
    df_neg = df_genome[df_genome['Post-filter q-value'] > 0.99]
    nb_line = df_pos.shape[0]
    df_neg = df_neg.sample(nb_line, random_state=1)
    pos_ids = df_pos['ID'].values.tolist()
    neg_ids = df_neg['ID'].values.tolist()
    df = pd.read_csv(feature_file, header=0, index_col=0, sep='\t')
    df = df.loc[::, ['mut', 'cna', 'rna']]
    df_pos = df.loc[pos_ids, ::]
    df_neg = df.loc[neg_ids, ::]
    pos_label = np.ones((df_pos.shape[0]))
    neg_label = np.zeros((df_neg.shape[0]))
    X = np.concatenate([df_pos.values.astype(float), df_neg.values.astype(float)])
    y = np.concatenate([pos_label, neg_label])
    print(X.shape, y.shape)
    return X, y


def fit_cv(X, y, k=10, method='SVM', b_plot=False):
    n = X.shape[0]
    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    assignments = assignments[:n]
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    all_roc = []
    for i in range(1, k + 1):
        ix = assignments == i
        y_test = y[ix]
        y_train = y[~ix]
        # X = X.reshape(X.size)
        X_train = X[~ix, :]
        X_test = X[ix, :]
        scaler = preprocessing.MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if method == 'SVM':
            model = SVC(gamma='auto', probability=True)
            model.fit(X_train, y_train)
            probas_ = model.predict_proba(X_test)[:, 1]
            del model
        elif method == 'RF':
            model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
            model.fit(X_train, y_train)
            probas_ = model.predict_proba(X_test)[:, 1]
            del model
        fpr, tpr, thresholds = roc_curve(y_test, probas_)
        # Compute ROC curve and area the curve
        mean_tpr += interp(mean_fpr, fpr, tpr)
        all_roc.append(auc(fpr, tpr))
        mean_tpr[0] = 0.0
    mean_tpr /= k
    mean_tpr[-1] = 1.0
    print(all_roc)
    mean_auc = auc(mean_fpr, mean_tpr)
    print(all_roc)
    print("Mean ROC (area = %0.4f)" % mean_auc)
    if b_plot:
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='Mean ROC (area = %0.4f)' % mean_auc, lw=2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Variants prediction (ROC) with 10-fold cross validation')
        plt.legend(loc="lower right")
        plt.show()


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='NCDriver v0.1.')
    parser.add_argument("-f", dest='feature', default="./Pancan/Pancan.fea", help="feature")
    parser.add_argument("-c", dest='coding', default="./coding_key.csv", help="coding file")
    parser.add_argument("-n", dest='non_coding', default="./non_coding_key.csv", help="non_coding file")
    parser.add_argument("-m", dest='method', default="RF", help="method")
    args = parser.parse_args()
    X, y = file2data(args.feature, args.coding, args.non_coding)
    fit_cv(X, y, 10, args.method, False)


if __name__ == "__main__":
    main()
