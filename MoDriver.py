import argparse
import sys
# numpy
import numpy as np
import random
import os, time
import math
import re
import tempfile
from tempfile import mkdtemp
from subprocess import Popen, check_output
import pandas as pd
import pickle
import gzip
import pysam
# sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import mixture
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import linear_model, preprocessing
from sklearn.ensemble import RandomForestClassifier
# tensorflow
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, BatchNormalization, Dropout, Activation, merge, Conv2D, \
    MaxPooling2D, Activation, LeakyReLU, concatenate
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam, RMSprop
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from numpy import interp
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.multitest as mt
from os.path import splitext, basename, exists, abspath, isfile, getsize

nb_seed = 1
random.seed(nb_seed)
np.random.seed(nb_seed)
tf.set_random_seed(nb_seed)


class GeLU(Activation):
    def __init__(self, activation, **kwargs):
        super(GeLU, self).__init__(activation, **kwargs)
        self.__name__ = 'gelu'


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.1
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


get_custom_objects().update({'gelu': GeLU(gelu)})
get_custom_objects().update({'focal_loss': focal_loss})


class MoDriver():
    def __init__(self, datasets, model_path='./MoDriver.h5', epochs=30, batch_size=16):
        self.latent_dim = 128
        self.n = len(datasets)
        self.epochs = epochs
        self.batch_size = batch_size
        data_size = 0
        if self.n >= 1:
            data_size = datasets[0].shape[0]
        print(data_size)
        self.shape = []
        self.model_path = model_path
        input = []
        output = []
        for i in range(self.n):
            self.shape.append(datasets[i].shape[1])
        self.disc = self.build_disc()
        self.encoder = self.build_encoder()
        for i in range(self.n):
            input.append(Input(shape=(self.shape[i],)))
        z = self.encoder(input)
        output = self.disc(z)
        self.model = Model(input, output)
        self.model.compile(loss=focal_loss, optimizer=Adam(), metrics=['accuracy'])
        print(self.model.summary())
        return

    def build_encoder(self):
        encoding_dim = self.latent_dim
        X = []
        denses = []
        for i in range(self.n):
            X.append(Input(shape=(self.shape[i],)))
        for i in range(self.n):
            denses.append(Dense(self.shape[i] * 2, kernel_initializer="glorot_normal")(X[i]))
        if self.n > 1:
            merged_dense = concatenate(denses, axis=-1)
        else:
            merged_dense = denses[0]
        model = BatchNormalization()(merged_dense)
        model = Activation('gelu')(model)
        z = Dense(encoding_dim, kernel_initializer="glorot_normal")(model)
        return Model(X, z)

    def build_disc(self):
        X = Input(shape=(self.latent_dim,))
        dec = Dense(1, activation='sigmoid')(X)
        m_decoder = Model(X, dec)
        return m_decoder


def check_feature(df):
    ids = df[df['sample_count_mean'] < 1e-6].index.tolist()
    return ids


def file2data(cancer_type, train_pos, train_neg):
    mode_all = ['mut', 'cna', 'rna']
    tumors_file = './tumors.txt'
    tumors_set = {'Pancan': 'Pancan'}
    for line in open(tumors_file, 'rt'):
        txt = line.rstrip().split('\t')
        tumors_set[txt[0]] = txt[1]
    X_train = []
    X_sim = []
    X = []
    for mode in mode_all:
        fea_one = './%s/%s.fea' % (tumors_set[cancer_type], mode)
        df_one = pd.read_csv(fea_one, header=0, index_col=0, sep='\t')
        fea_sim_one = './sim/%s/%s.fea' % (tumors_set[cancer_type], mode)
        df_sim_one = pd.read_csv(fea_sim_one, header=0, index_col=0, sep='\t')
        ids = list(df_one.index)
        mat_train_pos = df_one.loc[train_pos, ::].values.astype(float)
        mat_train_neg = df_one.loc[train_neg, ::].values.astype(float)
        X_train.append(np.concatenate([mat_train_pos, mat_train_neg]))
        X.append(df_one.values.astype(float))
        X_sim.append(df_sim_one.values.astype(float))
    y_train = np.concatenate([np.ones((len(train_pos))), np.zeros((len(train_neg)))])
    return X_train, y_train, X, X_sim, ids


def fit(Xs, y, type):
    model_path = './model/%s.model' % type
    X = []
    for j in range(len(Xs)):
        scaler_path = './model/%s_%d.scaler' % (type, j)
        scaler = preprocessing.MinMaxScaler()
        X_one = scaler.fit_transform(Xs[j])
        X.append(X_one)
        fp = open(scaler_path, 'wb')
        pickle.dump(scaler, fp)
        fp.close()
        del scaler
    dcd = MoDriver(X)
    dcd.model.fit(X, y, batch_size=16, epochs=20, verbose=0)
    dcd.model.save(model_path)
    del dcd


# 10-fold cross validation
def fit_cv(Xs, y, k=10):
    n = Xs[0].shape[0]
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
        Xs_train = []
        Xs_test = []
        for j in range(len(Xs)):
            X_train = Xs[j][~ix, :]
            X_test = Xs[j][ix, :]
            scaler = preprocessing.MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            Xs_train.append(X_train)
            Xs_test.append(X_test)
        X_all_train = np.concatenate(Xs_train, axis=1)
        X_all_test = np.concatenate(Xs_test, axis=1)
        dcd = MoDriver(Xs_train)
        dcd.model.fit(Xs_train, y_train, batch_size=16, epochs=40, verbose=0)
        probas_ = dcd.model.predict(Xs_test, verbose=0)
        del dcd
        fpr, tpr, thresholds = roc_curve(y_test, probas_)
        # Compute ROC curve and area the curve
        mean_tpr += interp(mean_fpr, fpr, tpr)
        all_roc.append(auc(fpr, tpr))
        mean_tpr[0] = 0.0
    mean_tpr /= k
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print(all_roc)
    print("Mean ROC (area = %0.4f)" % mean_auc)


# annotation with Oncotator
def anno(file, cols=[]):
    bin_path = '/soft/a/envs/Oncotator/bin/Oncotator'  # path of Oncotator
    anno_path = '/data/oncotator_v1_ds_April052016'  # data pf Oncotator
    tmp_dir = '/data/tmp/'
    out_tmp = '%s/out_tmp.maf' % tmp_dir
    cmd = '%s -v --db-dir %s %s %s hg19 --output_format=TCGAMAF --tx-mode=EFFECT' % (bin_path, anno_path, file, out_tmp)
    check_output(cmd, shell=True)
    cmd = 'rm -f oncotator.log'
    check_output(cmd, shell=True)
    df = pd.read_csv(out_tmp, sep='\t', header=0, comment='#', usecols=cols)
    return df


def predict(Xs, type):
    model_path = './model/%s.model' % type
    X = []
    for j in range(len(Xs)):
        scaler_path = './model/%s_%d.scaler' % (type, j)
        scaler = pickle.load(open(scaler_path, 'rb'))
        X.append(scaler.transform(Xs[j]))
    dcd = MoDriver(X)
    dcd.model = load_model(model_path)
    return dcd.model.predict(X, verbose=0)


def eval(Xs, y, type):
    model_path = './model/%s.model' % type
    X = []
    for j in range(len(Xs)):
        scaler_path = './model/%s_%d.scaler' % (type, j)
        scaler = pickle.load(open(scaler_path, 'rb'))
        X.append(scaler.transform(Xs[j]))
    dcd = MoDriver(X)
    dcd.model = load_model(model_path)
    y_p = dcd.model.predict(X, verbose=0)
    fpr, tpr, thresholds = roc_curve(y, y_p)
    return auc(fpr, tpr)


def build_set(pos_key, neg_key, all_list, nb_imb=20, genome='a'):
    pos_ids = []
    neg_ids = []
    rand_dis = []
    for id in all_list:
        tmps = re.split('::', id)
        gene = tmps[2]
        reg = tmps[0]
        if 'cds' in reg and genome == 'n':
            continue
        elif 'cds' not in reg and genome == 'c':
            continue
        if gene in pos_key:
            pos_ids.append(id)
        elif gene in neg_key:
            neg_ids.append(id)
        else:
            rand_dis.append(id)
    rand_dis = random.sample(rand_dis, len(pos_ids) * nb_imb)
    neg_ids = list(set(rand_dis) | set(neg_ids))
    pos_ids.sort()
    neg_ids.sort()
    print(len(pos_ids), len(neg_ids))
    return pos_ids, neg_ids


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='MODriver v1.0')
    parser.add_argument("-c", dest='coding', default="./coding_key.csv", help="coding file")
    parser.add_argument("-n", dest='non_coding', default="./non_coding_key.csv", help="non_coding file")
    parser.add_argument("-s", dest='pos', default="./pos_2018.txt", help="coding file")
    parser.add_argument("-g", dest='neg', default="./neg_2018.txt", help="non_coding file")
    parser.add_argument("-m", dest='mode', default="sort", help="mode")
    parser.add_argument("-t", dest='type', default="Pancan", help="cancer type")
    parser.add_argument("-o", dest='out', default="./score/", help="coding file")
    parser.add_argument("-p", dest='threads_num', type=int, default=1, help="threads num")
    args = parser.parse_args()
    df_tmp = pd.read_csv('./chr_id.txt', header=0, index_col=3, sep='\t', usecols=[0, 1, 2, 3])
    all_list = df_tmp.index.tolist()
    key_2018 = './key_2018.txt'
    # if args.type  != 'Pancan':
    #     key_2018 = "./input/%s.key" % args.type
    pd_key = pd.read_csv(key_2018, header=None, sep='\t')
    pd_neg = pd.read_csv('./neg_2018.txt', header=None, sep='\t')
    pd_neg.columns = ['gene']
    pd_key.columns = ['gene', 'type']
    pd_key = pd_key.drop_duplicates(subset=['gene'], keep='first')
    pd_neg = pd_neg.drop_duplicates(subset=['gene'], keep='first')
    key_18 = pd_key['gene'].values.tolist()
    neg_18 = pd_neg['gene'].values.tolist()
    known_key = ['TERT']
    neg_key = ['CACNA1E', 'COL11A1', 'DST', 'TTN']
    key_18 = list(set(key_18) | set(known_key))
    # neg_key = list(set(neg_18) | set(neg_key))
    pos, neg = build_set(key_18, neg_key, all_list, nb_imb=20)
    # pos, neg = pickle.load(open('pos.neg', 'rb'))
    X_train, y_train, X, X_sim, ids = file2data(args.type, pos, neg)
    print(X_train[0].shape[0], X[0].shape[0], X_sim[0].shape[0])

    if args.mode == 'train':
        fit(X_train, y_train, args.type)

    elif args.mode == 'cv':
        fit_cv(X_train, y_train, 10, False)

    elif args.mode == 'score':
        y_p = predict(X, args.type)
        null_dist_path = '%s%s.null' % (args.out, args.type)
        f = open(null_dist_path, 'rb')
        null_dist = pickle.load(f)
        f.close()
        df_all = pd.DataFrame(data=y_p, index=ids, columns=['score'])
        ge_type = {}
        for id in ids:
            tmp = re.split('::', id)[0]
            tmp = str(tmp).replace("gc19_pc.", "")
            if tmp not in ge_type:
                ge_type[tmp] = [id]
            else:
                ge_type[tmp].append(id)
        nb_coding_drivers = 0
        nb_noncoding_drivers = 0
        dfs = []
        for key in ge_type.keys():
            df_score = df_all.loc[ge_type[key], ::]
            out_path = '%s%s.%s.score' % (args.out, args.type, key)
            pvals = 1 - null_dist(df_score['score'].values.tolist())
            df_score['p'] = pvals
            p_min = 1e-6
            df_score.loc[df_score['p'] < p_min, 'p'] = p_min
            _, qvals, _, _ = mt.multipletests(pvals=pvals, alpha=0.1, method='fdr_bh')
            df_score['q'] = qvals
            df_show = df_score[df_score['q'] < 0.1]
            dfs.append(df_show)
            if key == 'cds':
                nb_coding_drivers += df_show.shape[0]
            else:
                nb_noncoding_drivers += df_show.shape[0]
            df_score = df_score.sort_values(by=['score'], ascending=[False])
            df_score.to_csv(out_path, header=True)
        out_path = "%s%s.%s.score" % ("./", args.type, 'all')
        df = pd.concat(dfs, axis=0)
        df = df.sort_values(by=['score'], ascending=[False])
        df.to_csv(out_path, header=True)
        print(nb_coding_drivers + nb_noncoding_drivers, nb_coding_drivers, nb_noncoding_drivers)

    elif args.mode == 'null':
        y_sim = predict(X_sim, args.type, method=args.learn, b_null=True)
        df_sim = pd.DataFrame(data=y_sim, columns=['score'])
        out_path = '%s%s.null' % (args.out, args.type)
        null_dist = sm.distributions.ECDF(df_sim['score'].values.tolist())
        fp = open(out_path, 'wb')
        pickle.dump(null_dist, fp)
        fp.close()

    elif args.mode == 'simulation':
        tmp_dir = '/data/tmp/'
        sim_file = 'simulation.txt.gz'
        # based on the ori maf file
        ori_input = '../data/ICGC/final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz'
        col0 = ['Chromosome', 'Start_position', 'End_position', 'Reference_Allele',
                'Tumor_Seq_Allele2', 'Tumor_Sample_Barcode', 'Matched_Norm_Sample_Barcode']
        promoter_set = ['TERT', 'MALAT1', 'NEAT1']
        df = pd.read_csv(ori_input, header=0, sep='\t', usecols=col0 + ['Hugo_Symbol'])
        # remove the mutations in the TERT promoter, MALAT1, or NEAT1
        df_anno = df.loc[~df['Hugo_Symbol'].isin(promoter_set), col0]
        all_input_file = '%s/all_input.txt' % tmp_dir
        all_out_file = '%s/all_out.txt' % tmp_dir
        df_anno.to_csv(all_input_file, header=False, index=False, sep='\t')
        cmd = "python parallel_do.py -c 'python simulation.py -i %s -o %s' -t %d --r" % (
            all_input_file, all_out_file, args.threads_num)
        print(cmd)
        check_output(cmd, shell=True)
        df = pd.read_csv(all_out_file, header=None, sep='\t')
        df.columns = ['Chromosome', 'Start_position', 'End_position', 'Variant_Classification', 'Variant_Type',
                      'Reference_Allele',
                      'Tumor_Seq_Allele2', 'Tumor_Sample_Barcode', 'Matched_Norm_Sample_Barcode', 'gc_content']
        df.to_csv(sim_file, header=True, index=False, sep='\t', compression='gzip', float_format='%.3f')
        print("random mutations: " + str(df.shape[0]))


if __name__ == "__main__":
    main()
