import argparse
import sys
import numpy as np
import random

nb_seed = 1
random.seed(nb_seed)
np.random.seed(nb_seed)
import os, time
import math
import re
import tempfile
from tempfile import mkdtemp
from subprocess import Popen, check_output
import pandas as pd
import gzip
import pickle
from sklearn.ensemble import RandomForestClassifier
from os.path import splitext, basename, exists, abspath, isfile, isdir, getsize
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import glob
from scipy.stats import fisher_exact
import statsmodels.api as sm
import statsmodels.stats.multitest as mt


def pancan_q(method):
    score_file = './Pancan.all.score'
    df0 = pd.read_csv(score_file, header=0, index_col=0, sep=',')
    df0 = df0.loc[::, ['q']]
    # df0['q'] = df0['q'].apply(lambda x: -math.log10(max(1e-6, x)))
    df0['CGC'] = 1
    df0.columns = ['MODriver', 'CGC']
    tumors_set = {'Pancan': 'Pancan-no-skin-melanoma-lymph'}
    type = 'Pancan'
    if method in ['DriverPower', 'ncdDetect', 'oncodriveFML_cadd', 'NBR']:
        base_dir = '../data/ICGC/p-values/observed/%s/%s' % (tumors_set[type], method)
        file_list = glob.glob("%s/*.observed.*" % (base_dir))
    else:
        base_dir = '../data/ICGC/p-values/observed/%s/%s' % (type, method)
        file_list = glob.glob("%s/*.observed.*" % (base_dir))
    if len(file_list) == 0:
        file_list = glob.glob("%s/%s*" % (base_dir, type))
    if len(file_list) == 0:
        print("scores of %s on %s do not exist!" % (method, type))
    dfs = []
    if method == 'CGC':
        cgc_file = './cgc_v91.csv'
        df_cgc = pd.read_csv(cgc_file, header=0, sep=',')
        cgc_genes = df_cgc.loc[df_cgc['Somatic'] == 'yes', 'Gene Symbol'].values.tolist()
        for id in df0.index:
            tmps = re.split('::', id)
            gene = tmps[2]
            if gene in cgc_genes:
                df0.at[id, 'CGC'] = 1e-6
    for line in file_list:
        file_name = line
        if method == 'DriverPower':
            df = pd.read_csv(file_name, header=0, index_col=0, sep='\t')
            df = df.loc[::, ['p-value']]
        elif method == 'oncodriveFML_cadd' or method == 'ExInAtor':
            df = pd.read_csv(file_name, header=0, index_col=3, sep='\t')
            df = df.loc[::, ['p-value']]
        elif method == 'NBR' or method == 'ncDriver_combined' or method == 'ncdDetect':
            df = pd.read_csv(file_name, header=0, index_col=0, sep='\t')
            df = df.iloc[::, [-1]]
        df.columns = ['p']
        df = df.fillna(1.0)
        df['p'] = df['p'].apply(lambda x: max(1e-16, x))
        pvals = df['p'].values.astype(float).tolist()
        _, qvals, _, _ = mt.multipletests(pvals=pvals, alpha=0.1, method='fdr_bh')
        df['q'] = qvals
        df = df.loc[::, ['q']]
        # df = df.loc[df['q'] < 0.1]
        dfs.append(df)
    if len(dfs) > 0:
        df_all = pd.concat(dfs, axis=0)
        df_all = df_all.loc[~df_all.index.duplicated(keep='first')]
        ids = []
        for id in df_all.index:
            if id in df0.index:
                ids.append(id)
        df0[method] = 1
        df0.loc[ids, method] = df_all.loc[ids, 'q'].values.astype(float)
    df0 = df0.loc[::, method].apply(lambda x: -math.log10(max(1e-6, x)))
    return df0


def other_q(type, method):
    all_ids_file = './chr_id.txt'
    df0 = pd.read_csv(all_ids_file, header=0, index_col=3, sep='\t')
    df0 = df0.loc[::, []]
    df0['score'] = 0.0
    print(df0.shape[0])
    tumors_file = './tumors.txt'
    tumors_set = {}
    for line in open(tumors_file, 'rt'):
        txt = line.rstrip().split('\t')
        tumors_set[txt[0]] = txt[1]
    base_dir = '../data/ICGC/p-values/observed/%s/%s' % (tumors_set[type], method)
    file_list = glob.glob("%s/*.observed.*" % (base_dir))
    if len(file_list) == 0:
        file_list = glob.glob("%s/%s*" % (base_dir, type))
    if len(file_list) == 0:
        file_list = glob.glob("%s/%s*" % (base_dir, tumors_set[type]))
    if len(file_list) == 0:
        print("scores of %s on %s do not exist!" % (method, type))
    dfs = []
    for line in file_list:
        file_name = line
        if method == 'DriverPower':
            df = pd.read_csv(file_name, header=0, index_col=0, sep='\t')
            df = df.loc[::, ['p-value']]
        elif method == 'oncodriveFML_cadd' or method == 'ExInAtor':
            df = pd.read_csv(file_name, header=0, index_col=3, sep='\t')
            df = df.loc[::, ['p-value']]
        elif method == 'ncdDetect':
            df = pd.read_csv(file_name, header=0, index_col=0, sep='\t')
            df = df.iloc[::, [-1]]
        elif method == 'NBR' or method == 'ncdDetect':
            df = pd.read_csv(file_name, header=0, index_col=0, sep='\t')
            df = df.iloc[::, [-1]]
        df.columns = ['p']
        df = df.fillna(1.0)
        pvals = df['p'].values.astype(float).tolist()
        _, qvals, _, _ = mt.multipletests(pvals=pvals, alpha=0.1, method='fdr_bh')
        df['q'] = qvals
        df = df.loc[df['q'] < 0.1]
        dfs.append(df)
    if len(dfs) > 0:
        df_all = pd.concat(dfs, axis=0)
        df_all = df_all.loc[~df_all.index.duplicated(keep='first')]
        out_path = './tmp/%s_%s.score' % (method, type)
        df_all.to_csv(out_path, header=True, index=True, sep=',')
    return


def read_score(type, method, cgc_set):
    all_ids_file = './chr_id.txt'
    df0 = pd.read_csv(all_ids_file, header=0, index_col=3, sep='\t')
    df0 = df0.loc[::, []]
    df0['score'] = 0.0
    print(df0.shape[0])
    tumors_file = './tumors.txt'
    tumors_set = {'Pancan': 'Pancan-no-skin-melanoma-lymph'}
    for line in open(tumors_file, 'rt'):
        txt = line.rstrip().split('\t')
        tumors_set[txt[0]] = txt[1]
    if method == 'MODriver':
        base_dir = './score/'
        file_list = glob.glob("%s/%s*.score" % (base_dir, type))
        base_dir = '../data/ICGC/p-values/observed/%s/%s' % (tumors_set[type], method)
    elif method in ['DriverPower', 'ncdDetect', 'oncodriveFML_cadd']:
        base_dir = '../data/ICGC/p-values/observed/%s/%s' % (tumors_set[type], method)
        file_list = glob.glob("%s/*.observed.*" % (base_dir))
    else:
        base_dir = '../data/ICGC/p-values/observed/%s/%s' % (type, method)
        file_list = glob.glob("%s/*.observed.*" % (base_dir))
    if len(file_list) == 0:
        file_list = glob.glob("%s/%s*" % (base_dir, type))
    if len(file_list) == 0:
        print("scores of %s on %s do not exist!" % (method, type))
    dfs = []
    if method == 'MODriver':
        score_dir = './%s.all.score' % (type)
        df_q = pd.read_csv(score_dir, header=0, index_col=0, sep=',')
        ids_all = df_q.index.tolist()
        c_ids = []
        nc_ids = []
        c_cgc = []
        nc_cgc = []
        for id in ids_all:
            tmps = re.split('::', id)
            reg = tmps[0]
            if 'cds' in reg:
                c_ids.append(id)
                if tmps[2] in cgc_set:
                    c_cgc.append(id)
            else:
                nc_ids.append(id)
                if tmps[2] in cgc_set:
                    nc_cgc.append(id)
        print(len(c_ids), len(nc_ids))
        print(float(len(c_cgc)) / len(c_ids), float(len(nc_cgc)) / len(nc_ids))
    for line in file_list:
        file_name = line
        if method == 'DriverPower':
            df = pd.read_csv(file_name, header=0, index_col=0, sep='\t')
            df = df.loc[::, ['p-value']]
        elif method == 'MODriver':
            df = pd.read_csv(file_name, header=0, index_col=0, sep=',')
            df = df.loc[::, ['p']]
        elif method == 'oncodriveFML_cadd' or method == 'ExInAtor':
            df = pd.read_csv(file_name, header=0, index_col=3, sep='\t')
            df = df.loc[::, ['p-value']]
        elif method == method == 'NBR' or method == 'ncdDetect':
            df = pd.read_csv(file_name, header=0, index_col=0, sep='\t')
            df = df.iloc[::, [-1]]
        df.columns = ['score']
        df = df.fillna(1.0)
        df['score'] = df['score'].apply(lambda x: max(1e-16, x))
        df['score'] = df['score'].apply(lambda x: - math.log10(x))
        dfs.append(df)
    if len(dfs) > 0:
        df_all = pd.concat(dfs, axis=0)
        df_all = df_all.loc[~df_all.index.duplicated(keep='first')]
        ids = []
        for id in df_all.index:
            if id in df0.index:
                ids.append(id)
        df0.loc[ids, 'score'] = df_all.loc[ids, 'score'].values.astype(float)
    return df0


def set2res(cancer_type, pos1, neg1, pos2, neg2, pos3, neg3, pos4, neg4, cgc_genes):
    methods = ['DriverPower', 'ncdDetect', 'oncodriveFML_cadd', 'ExInAtor', 'MODriver', 'NBR']
    res1 = {}
    pr_auc2 = []
    pr_auc3 = []
    pr_auc1 = []
    pr_auc4 = []
    top_n = 50
    for m in methods:
        df = read_score(cancer_type, m, set(cgc_genes))
        p_pos1 = df.loc[pos1, 'score'].values.tolist()
        p_neg1 = df.loc[neg1, 'score'].values.tolist()
        p_pos2 = df.loc[pos2, 'score'].values.tolist()
        p_neg2 = df.loc[neg2, 'score'].values.tolist()
        p_pos3 = df.loc[pos3, 'score'].values.tolist()
        p_neg3 = df.loc[neg3, 'score'].values.tolist()
        p_pos4 = df.loc[pos4, 'score'].values.tolist()
        p_neg4 = df.loc[neg4, 'score'].values.tolist()
        p_1 = np.concatenate([p_pos1, p_neg1])
        p_2 = np.concatenate([p_pos2, p_neg2])
        p_3 = np.concatenate([p_pos3, p_neg3])
        p_4 = np.concatenate([p_pos4, p_neg4])
        y_1 = np.concatenate([np.ones((len(pos1))), np.zeros((len(neg1)))])
        y_2 = np.concatenate([np.ones((len(pos2))), np.zeros((len(neg2)))])
        y_3 = np.concatenate([np.ones((len(pos3))), np.zeros((len(neg3)))])
        y_4 = np.concatenate([np.ones((len(pos4))), np.zeros((len(neg4)))])
        pr_auc1.append(average_precision_score(y_1, p_1))
        pr_auc2.append(average_precision_score(y_2, p_2))
        pr_auc3.append(average_precision_score(y_3, p_3))
        pr_auc4.append(average_precision_score(y_4, p_4))
    return pr_auc1, pr_auc2, pr_auc3, pr_auc4


def build_set1(pos_genes, all_ids, nb_imb=1, genome='a'):
    pos = []
    neg = []
    for id in all_ids:
        tmps = re.split('::', id)
        reg = tmps[0]
        if 'cds' in reg and genome == 'n':
            continue
        elif 'cds' not in reg and genome == 'c':
            continue
        if tmps[2] in pos_genes:
            pos.append(id)
        else:
            neg.append(id)
    neg = random.sample(neg, len(pos) * nb_imb)
    return pos, neg


def build_set2(pos_ids, all_ids, nb_imb=1, genome='a'):
    pos = []
    neg = []
    for id in all_ids:
        tmps = re.split('::', id)
        reg = tmps[0]
        if 'cds' in reg and genome == 'n':
            continue
        elif 'cds' not in reg and genome == 'c':
            continue
        if id in pos_ids:
            pos.append(id)
        else:
            neg.append(id)
    neg = np.random.choice(neg, len(pos) * nb_imb).tolist()
    return pos, neg


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='fea v0.01.')
    parser.add_argument("-p", dest='path',
                        default="all.list",
                        help="meth input")
    parser.add_argument("-i", dest='input', default="./tumors.txt", help="fea")
    parser.add_argument("-m", dest='mode', default="compare", help="fea")
    args = parser.parse_args()
    coding_l = ['cds']
    plist = ['Kidney-RCC', 'Head-SCC', 'Panc-Endocrine', 'Prost-AdenoCA', 'Liver-HCC', 'Breast-AdenoCA', 'Panc-AdenoCA',
             'Myeloid-MPN', 'Bone-Osteosarc', 'CNS-PiloAstro', 'Stomach-AdenoCA', 'Eso-AdenoCA', 'CNS-Medullo',
             'Biliary-AdenoCA', 'Ovary-AdenoCA']
    noncoding_l = ['enhancers', '5utr', 'smallrna.ncrna', 'lncrna.promCore', 'promCore', 'lncrna.ncrna',
                   'mirna.prom', '3utr', 'lncrna.ss', 'mirna_pre', 'mirna_mat']

    # scoring all the cancer sets
    if args.mode == 'score':
        fea_type = ['mut', 'cna', 'rna']
        mode = ['train', 'null', 'score']
        for p in plist:
            for m in fea_type:
                cmd = 'python fea.py -m %s -t %s; cd ./sim/; python fea_sim.py -m %s -t %s; cd ..' % (m, p, m, p)
                print(cmd)
                check_output(cmd, shell=True)
        for p in plist:
            for m in mode:
                cmd = 'python MoDriver.py -l MODNN -m %s -t %s' % (m, p)
                print(cmd)
                check_output(cmd, shell=True)

    # get the weight of features
    elif args.mode == 'fea_weight':
        plist = ['Pancan', 'Kidney-RCC', 'Head-SCC', 'Panc-Endocrine', 'Prost-AdenoCA', 'Liver-HCC', 'Breast-AdenoCA',
                 'Panc-AdenoCA',
                 'Myeloid-MPN', 'Bone-Osteosarc', 'CNS-PiloAstro', 'Stomach-AdenoCA', 'Eso-AdenoCA', 'CNS-Medullo',
                 'Biliary-AdenoCA', 'Ovary-AdenoCA']
        tumors_file = './tumors.txt'
        tumors_set = {'Pancan': 'Pancan'}
        for line in open(tumors_file, 'rt'):
            txt = line.rstrip().split('\t')
            tumors_set[txt[0]] = txt[1]
        dfs = []
        for p in plist:
            file_out = './results/' + p + '.weight'
            df = pd.read_csv(file_out, header=0, index_col=0, sep=',')
            df.columns = [p]
            dfs.append(df)
        df_all = pd.concat(dfs, axis=1)
        df_all.to_csv("tmp/df6.txt", header=True, index=True, sep=',')

    elif args.mode == 'RF':
        plist = ['Pancan', 'Kidney-RCC', 'Head-SCC', 'Panc-Endocrine', 'Prost-AdenoCA', 'Liver-HCC', 'Breast-AdenoCA',
                 'Panc-AdenoCA',
                 'Myeloid-MPN', 'Bone-Osteosarc', 'CNS-PiloAstro', 'Stomach-AdenoCA', 'Eso-AdenoCA', 'CNS-Medullo',
                 'Biliary-AdenoCA', 'Ovary-AdenoCA']
        tumors_file = './tumors.txt'
        tumors_set = {'Pancan': 'Pancan'}
        for line in open(tumors_file, 'rt'):
            txt = line.rstrip().split('\t')
            tumors_set[txt[0]] = txt[1]
        top_n = 3
        df_list = []
        nb_line = 0
        for p in plist:
            nb_line += 1
            cancer_type = p
            fea_type = ['mut', 'cna', 'rna']
            dfs = []
            bases = []
            base_dims = []
            nb_line = 0
            nb_ids = 0
            for f in fea_type:
                nb_line += 1
                fea_tmp_file = './' + tumors_set[cancer_type] + '/' + f + '.fea'
                df = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t')
                if nb_line == 1:
                    nb_ids = df.shape[0]
                    ids = df.index.tolist()
                base_dims.append(df.shape[1])
                dfs.append(df)
            X = pd.concat(dfs, axis=1)
            feas = list(X)
            res_file = "./%s.all.score" % p
            df_res = pd.read_csv(res_file, header=0, index_col=0, sep=',')
            df0 = pd.DataFrame(data=np.zeros((nb_ids, 1)), index=ids, columns=['score'])
            df0.loc[df_res.index.tolist(), 'score'] = 1
            y = df0.iloc[:, 0].values.astype(int)
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, oob_score=True)
            model.fit(X.values, y)
            ranks = model.feature_importances_
            i_top = np.argsort(ranks)[::-1][:top_n]
            cols = [feas[i] for i in i_top]
            scores = [ranks[i] for i in i_top]
            df = pd.DataFrame(data=((np.array(scores))), columns=['Gini importance'])
            df['type'] = p
            df['feature'] = cols
            df = df.loc[::, ['type', 'feature', 'Gini importance']]
            df_list.append(df)
        print(len(df_list))
        df_all = pd.concat(df_list, axis=0)
        df_all.to_csv("tmp/df7.txt", header=True, index=False, sep='\t')


    elif args.mode == 'other':
        methods = ['DriverPower', 'ncdDetect', 'oncodriveFML_cadd', 'ExInAtor', 'NBR']
        for m in methods:
            for p in plist:
                other_q(p, m)

    # get p-values of Enrichment analyzes with MoDriver-mut
    elif args.mode == 'mut_p':
        plist = ['Pancan', 'Kidney-RCC', 'Head-SCC', 'Panc-Endocrine', 'Prost-AdenoCA', 'Liver-HCC', 'Breast-AdenoCA',
                 'Panc-AdenoCA',
                 'Myeloid-MPN', 'Bone-Osteosarc', 'CNS-PiloAstro', 'Stomach-AdenoCA', 'Eso-AdenoCA', 'CNS-Medullo',
                 'Biliary-AdenoCA', 'Ovary-AdenoCA']
        cgc_file = './cgc_v91.csv'
        df_cgc = pd.read_csv(cgc_file, header=0, sep=',')
        cgc_genes = set(df_cgc.loc[df_cgc['Somatic'] == 'yes', 'Gene Symbol'].values.tolist())
        tumors_file = './tumors.txt'
        tumors_set = {'Pancan': 'Pancan'}
        for line in open(tumors_file, 'rt'):
            txt = line.rstrip().split('\t')
            tumors_set[txt[0]] = txt[1]
        PCAWG_file_coding = './coding_key.csv'
        PCAWG_file_noncoding = './non_coding_key.csv'
        coding_all = pd.read_csv(PCAWG_file_coding, header=0, sep=',',
                                 usecols=['ID', 'tissue', 'Pre-filter q-value', 'Post-filter q-value'])
        #coding_all = coding_all[coding_all['tissue'] == 'Pancan-no-skin-melanoma-lymph']
        non_coding_all = pd.read_csv(PCAWG_file_noncoding, header=0, sep=',',
                                     usecols=['ID', 'tissue', 'Pre-filter q-value', 'Post-filter q-value'])
        #non_coding_all = non_coding_all[non_coding_all['tissue'] == 'Pancan-no-skin-melanoma-lymph']
        pos2_ids = set(coding_all.loc[coding_all['Post-filter q-value'] < 0.1, 'ID'].tolist())
        pos3_ids = set(non_coding_all.loc[non_coding_all['Post-filter q-value'] < 0.1, 'ID'].tolist())
        nb_coding = 0
        nb_noncoding = 0
        nb_cgc_coding = 0
        nb_cgc_noncoding = 0
        nb_PCAWG_coding = len(pos2_ids)
        nb_PCAWG_noncoding = len(pos3_ids)
        df_tmp = pd.read_csv('./chr_id.txt', header=0, index_col=3, sep='\t', usecols=[0, 1, 2, 3])
        for e in df_tmp.index:
            if str(e).startswith("gc19_pc.cds"):
                nb_coding += 1
                tmp = re.split('::', e)[2]
                if tmp in cgc_genes:
                    nb_cgc_coding += 1
            else:
                nb_noncoding += 1
                tmp = re.split('::', e)[2]
                if tmp in cgc_genes:
                    nb_cgc_noncoding += 1
        methods = ['Mutation (CGC)', 'Mutation (PCAWG)', 'Multi-omics (CGC)', 'Multi-omics (PCAWG)']
        df = pd.DataFrame(data=np.zeros((len(plist), len(methods))), index=plist, columns=methods)
        for m in range(2):
            logs_p1 = []
            logs_p2 = []
            p1s = {}
            p2s = {}
            for p in plist:
                if m == 1:
                    score_file = "./%s.all.score" % p
                elif m == 0:
                    score_file = "../mut/%s.all.score" % p
                df_score = pd.read_csv(score_file, header=0, sep=',', index_col=0)
                sig_elements = df_score.index
                coding_elements = []
                noncoding_elements = []
                coding_cgcs = []
                noncoding_cgcs = []
                coding_PCAWG = []
                noncoding_PCAWG = []
                for e in sig_elements:
                    if str(e).startswith("gc19_pc.cds"):
                        coding_elements.append(e)
                    else:
                        noncoding_elements.append(e)
                for e in coding_elements:
                    try:
                        tmp = re.split('::', e)[2]
                    except:
                        continue
                    if tmp in cgc_genes:
                        coding_cgcs.append(e)
                    if e in pos2_ids:
                        coding_PCAWG.append(e)
                for e in noncoding_elements:
                    try:
                        tmp = re.split('::', e)[2]
                    except:
                        continue
                    if tmp in cgc_genes:
                        noncoding_cgcs.append(e)
                    if e in pos3_ids:
                        noncoding_PCAWG.append(e)
                nb_all = nb_coding + nb_noncoding
                nb_cgc = nb_cgc_coding + nb_cgc_noncoding
                nb_PCAWG = nb_PCAWG_coding + nb_PCAWG_noncoding
                nb_elements = len(coding_elements) + len(noncoding_elements)
                nb_cgcs = len(coding_cgcs) + len(noncoding_cgcs)
                nb_PCAWGs = len(coding_PCAWG) + len(noncoding_PCAWG)
                _, p1 = fisher_exact([[nb_cgc, nb_cgcs], [nb_all - nb_cgc, nb_elements - nb_cgcs]], "less")
                _, p2 = fisher_exact([[nb_PCAWG, nb_PCAWGs], [nb_all - nb_PCAWG, nb_elements - nb_PCAWGs]], "less")
                logs_p1.append(-math.log10(p1))
                p1s[p]= p1
                logs_p2.append(-math.log10(p2))
                p2s[p]= p1
                df.at[p, methods[2 * m]] = -math.log10(p1)
                df.at[p, methods[2 * m + 1]] = -math.log10(p2)
            p1s = sorted(p1s.items(), key=lambda x: x[1])
            p2s = sorted(p2s.items(), key=lambda x: x[1])
            print(m, np.median(logs_p1), np.median(logs_p2))
            print(m, p1s)
            print(m, p2s)
        df.to_csv('./tmp/df5.txt', sep=',', index=True, header=True)

    # get p-values of Enrichment analyzes with MoDriver-mo
    elif args.mode == 'p':
        cgc_file = './cgc_v91.csv'
        df_cgc = pd.read_csv(cgc_file, header=0, sep=',')
        cgc_genes = set(df_cgc.loc[df_cgc['Somatic'] == 'yes', 'Gene Symbol'].values.tolist())
        tumors_file = './tumors.txt'
        tumors_set = {'Pancan': 'Pancan'}
        for line in open(tumors_file, 'rt'):
            txt = line.rstrip().split('\t')
            tumors_set[txt[0]] = txt[1]
        PCAWG_file_coding = './coding_key.csv'
        PCAWG_file_noncoding = './non_coding_key.csv'
        coding_all = pd.read_csv(PCAWG_file_coding, header=0, sep=',',
                                 usecols=['ID', 'tissue', 'Pre-filter q-value', 'Post-filter q-value'])
        #coding_all = coding_all[coding_all['tissue'] == 'Pancan-no-skin-melanoma-lymph']
        non_coding_all = pd.read_csv(PCAWG_file_noncoding, header=0, sep=',',
                                     usecols=['ID', 'tissue', 'Pre-filter q-value', 'Post-filter q-value'])
        # non_coding_all = non_coding_all[non_coding_all['tissue'] == 'Pancan-no-skin-melanoma-lymph']
        pos2_ids = set(coding_all.loc[coding_all['Post-filter q-value'] < 0.1, 'ID'].tolist())
        pos3_ids = set(non_coding_all.loc[non_coding_all['Post-filter q-value'] < 0.1, 'ID'].tolist())
        df_tmp = pd.read_csv('./chr_id.txt', header=0, index_col=3, sep='\t', usecols=[0, 1, 2, 3])
        nb_coding = 0
        nb_noncoding = 0
        nb_cgc_coding = 0
        nb_cgc_noncoding = 0
        nb_PCAWG_coding = len(pos2_ids)
        nb_PCAWG_noncoding = len(pos3_ids)
        for e in df_tmp.index:
            if str(e).startswith("gc19_pc.cds"):
                nb_coding += 1
                tmp = re.split('::', e)[2]
                if tmp in cgc_genes:
                    nb_cgc_coding += 1
            else:
                nb_noncoding += 1
                tmp = re.split('::', e)[2]
                if tmp in cgc_genes:
                    nb_cgc_noncoding += 1

        methods = ['MODriver', 'DriverPower', 'ncdDetect', 'oncodriveFML_cadd', 'ExInAtor', 'NBR']
        plist = ['Kidney-RCC', 'Head-SCC', 'Panc-Endocrine', 'Prost-AdenoCA', 'Liver-HCC', 'Breast-AdenoCA',
                 'Panc-AdenoCA',
                 'Myeloid-MPN', 'Bone-Osteosarc', 'CNS-PiloAstro', 'Stomach-AdenoCA', 'Eso-AdenoCA', 'CNS-Medullo',
                 'Biliary-AdenoCA', 'Ovary-AdenoCA']
        df1 = pd.DataFrame(data=np.zeros((len(plist), len(methods))), index=plist, columns=methods)
        df2 = pd.DataFrame(data=np.zeros((len(plist), len(methods))), index=plist, columns=methods)
        for m in methods:
            logs_p1 = []
            logs_p2 = []
            p1s = {}
            p2s = {}
            nb_driver = []
            c_driver = []
            nc_driver = []
            for p in plist:
                if m == 'MODriver':
                    score_file = "./%s.all.score" % p
                else:
                    score_file = "./tmp/%s_%s.score" % (m, p)
                df_score = pd.read_csv(score_file, header=0, sep=',', index_col=0)
                sig_elements = df_score.index
                coding_elements = []
                noncoding_elements = []
                coding_cgcs = []
                noncoding_cgcs = []
                coding_PCAWG = []
                noncoding_PCAWG = []
                nb_driver.append(len(sig_elements))
                for e in sig_elements:
                    if str(e).startswith("gc19_pc.cds"):
                        coding_elements.append(e)
                    else:
                        noncoding_elements.append(e)
                for e in coding_elements:
                    try:
                        tmp = re.split('::', e)[2]
                    except:
                        continue
                    if tmp in cgc_genes:
                        coding_cgcs.append(e)
                    if e in pos2_ids:
                        coding_PCAWG.append(e)
                for e in noncoding_elements:
                    try:
                        tmp = re.split('::', e)[2]
                    except:
                        continue
                    if tmp in cgc_genes:
                        noncoding_cgcs.append(e)
                    if e in pos3_ids:
                        noncoding_PCAWG.append(e)

                nb_all = nb_coding + nb_noncoding
                nb_cgc = nb_cgc_coding + nb_cgc_noncoding
                nb_PCAWG = nb_PCAWG_coding + nb_PCAWG_noncoding
                nb_elements = len(coding_elements) + len(noncoding_elements)
                nb_cgcs = len(coding_cgcs) + len(noncoding_cgcs)
                nb_PCAWGs = len(coding_PCAWG) + len(noncoding_PCAWG)
                _, p1 = fisher_exact([[nb_cgc, nb_cgcs], [nb_all - nb_cgc, nb_elements - nb_cgcs]], "less")
                _, p2 = fisher_exact([[nb_PCAWG, nb_PCAWGs], [nb_all - nb_PCAWG, nb_elements - nb_PCAWGs]], "less")
                logs_p1.append(-math.log10(p1))
                logs_p2.append(-math.log10(p2))
                p1s[p] = p1
                p2s[p] = p2
                df1.at[p, m] = -math.log10(p1)
                df2.at[p, m] = -math.log10(p2)
            p1s = sorted(p1s.items(), key=lambda x: x[1])
            p2s = sorted(p2s.items(), key=lambda x: x[1])
            print(m, p1s)
            print(m, p2s)
            print(m, nb_driver, np.sum(nb_driver))
            print(m, np.median(logs_p1), np.median(logs_p2))
        ms_show = ['MODriver', 'DriverPower', 'ncdDetect', 'OncodriveFML', 'ExInAtor', 'NBR']
        df1.columns = ms_show
        df2.columns = ms_show
        df1.to_csv('./tmp/df3.txt', sep=',', index=True, header=True)
        df2.to_csv('./tmp/df4.txt', sep=',', index=True, header=True)


    elif args.mode == 'mo':
        cgc_file = './cgc_v91.csv'
        df_cgc = pd.read_csv(cgc_file, header=0, sep=',')
        cgc_genes = set(df_cgc.loc[df_cgc['Somatic'] == 'yes', 'Gene Symbol'].values.tolist())
        tumors_file = './tumors.txt'
        tumors_set = {'Pancan': 'Pancan'}
        for line in open(tumors_file, 'rt'):
            txt = line.rstrip().split('\t')
            tumors_set[txt[0]] = txt[1]
        PCAWG_file_coding = './coding_key.csv'
        PCAWG_file_noncoding = './non_coding_key.csv'
        coding_all = pd.read_csv(PCAWG_file_coding, header=0, sep=',',
                                 usecols=['ID', 'tissue', 'Pre-filter q-value', 'Post-filter q-value'])
        #coding_all = coding_all[coding_all['tissue'] == 'Pancan-no-skin-melanoma-lymph']
        non_coding_all = pd.read_csv(PCAWG_file_noncoding, header=0, sep=',',
                                     usecols=['ID', 'tissue', 'Pre-filter q-value', 'Post-filter q-value'])
        # non_coding_all = non_coding_all[non_coding_all['tissue'] == 'Pancan-no-skin-melanoma-lymph']
        pos2_ids = set(coding_all.loc[coding_all['Post-filter q-value'] < 0.1, 'ID'].tolist())
        pos3_ids = set(non_coding_all.loc[non_coding_all['Post-filter q-value'] < 0.1, 'ID'].tolist())
        df_tmp = pd.read_csv('./chr_id.txt', header=0, index_col=3, sep='\t', usecols=[0, 1, 2, 3])
        nb_coding = 0
        nb_noncoding = 0
        nb_cgc_coding = 0
        nb_cgc_noncoding = 0
        nb_PCAWG_coding = len(pos2_ids)
        nb_PCAWG_noncoding = len(pos3_ids)
        for e in df_tmp.index:
            if str(e).startswith("gc19_pc.cds"):
                nb_coding += 1
                tmp = re.split('::', e)[2]
                if tmp in cgc_genes:
                    nb_cgc_coding += 1
            else:
                nb_noncoding += 1
                tmp = re.split('::', e)[2]
                if tmp in cgc_genes:
                    nb_cgc_noncoding += 1

        methods = ['MODriver']
        plist = ['Pancan', 'Kidney-RCC', 'Head-SCC', 'Panc-Endocrine', 'Prost-AdenoCA', 'Liver-HCC', 'Breast-AdenoCA',
                 'Panc-AdenoCA',
                 'Myeloid-MPN', 'Bone-Osteosarc', 'CNS-PiloAstro', 'Stomach-AdenoCA', 'Eso-AdenoCA', 'CNS-Medullo',
                 'Biliary-AdenoCA', 'Ovary-AdenoCA', ]
        df1 = pd.DataFrame(data=np.zeros((len(plist), len(methods))), index=plist, columns=methods)
        df2 = pd.DataFrame(data=np.zeros((len(plist), len(methods))), index=plist, columns=methods)
        for m in methods:
            logs_p1 = []
            logs_p2 = []
            p1s = {}
            p2s = {}
            nb_driver = []
            c_driver = []
            nc_driver = []
            mo_cgc = []
            mo_PCAWG = []
            p_drivers = {}
            for p in plist:
                s_drivers = {}
                s_list = {}
                if m == 'MODriver':
                    score_file = "./%s.all.score" % p
                else:
                    score_file = "./tmp/%s_%s.score" % (m, p)
                df_score = pd.read_csv(score_file, header=0, sep=',', index_col=0)
                sig_elements = df_score.index
                coding_elements = []
                noncoding_elements = []
                coding_cgcs = []
                noncoding_cgcs = []
                coding_PCAWG = []
                noncoding_PCAWG = []
                nb_driver.append(len(sig_elements))
                if p == 'Pancan':
                    p_drivers = set(sig_elements.tolist())
                else:
                    s_drivers = set(sig_elements.tolist()) - p_drivers
                for e in s_drivers:
                    try:
                        tmp = re.split('::', e)[2]
                    except:
                        continue
                    if tmp in cgc_genes:
                        s_list[e] = 'CGC'
                    else:
                        s_list[e] = 'Non'
                nb_c = 0
                nb_nc = 0
                for e in sig_elements:
                    if str(e).startswith("gc19_pc.cds"):
                        coding_elements.append(e)
                        nb_c += 1
                    else:
                        noncoding_elements.append(e)
                        nb_nc += 1
                c_driver.append(nb_c)
                nc_driver.append(nb_nc)
                for e in coding_elements:
                    try:
                        tmp = re.split('::', e)[2]
                    except:
                        continue
                    if tmp in cgc_genes:
                        coding_cgcs.append(e)
                    if e in pos2_ids:
                        coding_PCAWG.append(e)
                for e in noncoding_elements:
                    try:
                        tmp = re.split('::', e)[2]
                    except:
                        continue
                    if tmp in cgc_genes:
                        noncoding_cgcs.append(e)
                    if e in pos3_ids:
                        noncoding_PCAWG.append(e)

                nb_all = nb_coding + nb_noncoding
                nb_cgc = nb_cgc_coding + nb_cgc_noncoding
                nb_PCAWG = nb_PCAWG_coding + nb_PCAWG_noncoding
                nb_elements = len(coding_elements) + len(noncoding_elements)
                nb_cgcs = len(coding_cgcs) + len(noncoding_cgcs)
                nb_PCAWGs = len(coding_PCAWG) + len(noncoding_PCAWG)

                _, p1 = fisher_exact([[nb_cgc, nb_cgcs], [nb_all - nb_cgc, nb_elements - nb_cgcs]], "less")
                _, p2 = fisher_exact([[nb_PCAWG, nb_PCAWGs], [nb_all - nb_PCAWG, nb_elements - nb_PCAWGs]], "less")
                mo_cgc.append(len(coding_cgcs))
                mo_PCAWG.append(len(coding_PCAWG))
                logs_p1.append(-math.log10(p1))
                logs_p2.append(-math.log10(p2))
                p1s[p] = p1
                p2s[p] = p2
                df1.at[p, m] = -math.log10(p1)
                df2.at[p, m] = -math.log10(p2)
                print(p, s_list)
            p1s = sorted(p1s.items(), key=lambda x: x[1])
            p2s = sorted(p2s.items(), key=lambda x: x[1])

            print(m, c_driver, np.sum(c_driver))
            print(m, nc_driver, np.sum(nc_driver))
            print(m, mo_cgc, np.sum(mo_cgc))
            print(m, mo_PCAWG, np.sum(mo_PCAWG))

    # compare the performance of all the methods
    elif args.mode == 'compare':
        plist = ['Pancan']
        cgc_file = './cgc_v91.csv'
        df_cgc = pd.read_csv(cgc_file, header=0, sep=',')
        cgc_genes = df_cgc.loc[df_cgc['Somatic'] == 'yes', 'Gene Symbol'].values.tolist()
        tumors_file = './tumors.txt'
        tumors_set = {'Pancan': 'Pancan'}
        for line in open(tumors_file, 'rt'):
            txt = line.rstrip().split('\t')
            tumors_set[txt[0]] = txt[1]
        df_tmp = pd.read_csv('./chr_id.txt', header=0, index_col=3, sep='\t', usecols=[0, 1, 2, 3])
        all_list = df_tmp.index.tolist()
        PCAWG_file_coding = './coding_key.csv'
        PCAWG_file_noncoding = './non_coding_key.csv'
        coding_all = pd.read_csv(PCAWG_file_coding, header=0, sep=',',
                                 usecols=['ID', 'tissue', 'Pre-filter q-value', 'Post-filter q-value'])
        coding_all = coding_all[coding_all['tissue'] == 'Pancan-no-skin-melanoma-lymph']

        non_coding_all = pd.read_csv(PCAWG_file_noncoding, header=0, sep=',',
                                     usecols=['ID', 'tissue', 'Pre-filter q-value', 'Post-filter q-value'])

        non_coding_all = non_coding_all[non_coding_all['tissue'] == 'Pancan-no-skin-melanoma-lymph']

        pos2_ids = coding_all.loc[coding_all['Post-filter q-value'] < 0.1, 'ID'].tolist()
        pos3_ids = non_coding_all.loc[non_coding_all['Post-filter q-value'] < 0.1, 'ID'].tolist()
        pos1, neg1 = build_set1(cgc_genes, all_list, nb_imb=1, genome='c')
        pos2, neg2 = build_set1(cgc_genes, all_list, nb_imb=1, genome='n')
        pos3, neg3 = build_set2(pos2_ids, all_list, nb_imb=1, genome='c')
        pos4, neg4 = build_set2(pos3_ids, all_list, nb_imb=1, genome='n')
        res1, res2, res3, res4 = set2res(plist[0], pos1, neg1, pos2, neg2, pos3, neg3, pos4, neg4, cgc_genes)
        methods = ['DriverPower', 'ncdDetect', 'OncodriveFML', 'ExInAtor', 'MODriver', 'NBR']
        df1 = pd.DataFrame(data=np.zeros((len(methods), 2)), index=methods, columns=['coding', 'noncoding'])
        df2 = pd.DataFrame(data=np.zeros((len(methods), 2)), index=methods, columns=['coding', 'noncoding'])
        df1['coding'] = res1
        df1['noncoding'] = res2
        df2['coding'] = res3
        df2['noncoding'] = res4
        df1.to_csv('./tmp/df1.txt', sep=',', index=True, header=True)
        df2.to_csv('./tmp/df2.txt', sep=',', index=True, header=True)


if __name__ == "__main__":
    main()
