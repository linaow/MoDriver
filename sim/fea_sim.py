import argparse
import sys
import numpy as np

np.random.seed(1)
import os, time
import math
import re
import pickle
import tempfile
from tempfile import mkdtemp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.linear_model import ElasticNet
from subprocess import Popen, check_output
import pandas as pd
import gzip
from os.path import splitext, basename, exists, abspath, isfile, getsize


def fea_mut_all(df_ori, str_col):
    df = df_ori.copy()
    types = ['Intron', 'IGR', 'RNA', 'Missense_Mutation', "3'UTR", 'lincRNA', "5'Flank", 'Silent', "5'UTR",
             'Splice_Site',
             'Nonsense_Mutation', 'De_novo_Start_OutOfFrame', 'Frame_Shift_Del', 'In_Frame_Del', 'Frame_Shift_Ins',
             'De_novo_Start_InFrame', 'Start_Codon_SNP', 'In_Frame_Ins', 'Nonstop_Mutation', 'Start_Codon_Del',
             'Stop_Codon_Del', 'Stop_Codon_Ins', 'Start_Codon_Ins']
    snps = ['SNP', 'DNP', 'TNP', 'DEL', 'INS', 'ONP']
    dfs = []
    cols = []
    for i in types:
        df_tmp = df.loc[df['type'] == i, ['group']]
        df_tmp = df_tmp.groupby(['group']).size().reset_index(name=i)
        df_tmp.index = df_tmp['group']
        df_tmp = df_tmp.loc[::, [i]]
        dfs.append(df_tmp)
        cols.append(i)
    for i in snps:
        df_tmp = df.loc[df['snp'] == i, ['group']]
        df_tmp = df_tmp.groupby(['group']).size().reset_index(name=i)
        df_tmp.index = df_tmp['group']
        df_tmp = df_tmp.loc[::, [i]]
        dfs.append(df_tmp)
        cols.append(i)
    df = df.groupby(['group']).size().reset_index(name=str_col + '_all')
    df.index = df['group']
    df = df.loc[::, []]
    for i in range(len(cols)):
        type = str_col + '_' + cols[i]
        df[type] = 0
        ids = dfs[i].index
        df.loc[ids, type] = dfs[i].loc[::, cols[i]].values.astype(int)
    return df


def fea_mut_mean(df_ori):
    df = df_ori.copy()
    df = df.groupby(['group', 'id']).size().reset_index(name='sample_count')
    df = df.drop_duplicates(subset=['group', 'id'], keep='first')
    df_mean = df.groupby('group').agg(q=('sample_count', 'mean')).reset_index()
    df_var = df.groupby('group').agg(q=('sample_count', 'var')).reset_index()
    df_mean.index = df_mean['group']
    df_var.index = df_var['group']
    df_mean = df_mean.loc[::, ['q']]
    df_var = df_var.loc[::, ['q']]
    df_mean.columns = ['sample_count_mean']
    df_var.columns = ['sample_count_var']
    df_var = df_var.fillna(0)
    df_all = pd.concat([df_mean, df_var], axis=1)
    return df_all


def fea_mut_gc(df_ori):
    df = df_ori.copy()
    df_mean = df.groupby('group').agg(q=('gc', 'mean')).reset_index()
    df_var = df.groupby('group').agg(q=('gc', 'var')).reset_index()
    df_mean.index = df_mean['group']
    df_var.index = df_var['group']
    df_mean = df_mean.loc[::, ['q']]
    df_var = df_var.loc[::, ['q']]
    df_mean.columns = ['gc_mean']
    df_var.columns = ['gc_var']
    df_var = df_var.fillna(0)
    df_all = pd.concat([df_mean, df_var], axis=1)
    return df_all


def fea_mut(df):
    tmp_dir = './tmp/'
    fea_bed = tmp_dir + '/fea.bed'
    tmp_bed = tmp_dir + '/out.bed'
    sample_bed = '../chr_id.bed'
    df.to_csv(fea_bed, header=False, index=False, sep='\t')
    cmd = "bedtools intersect -wb -a %s -b %s >%s" % (sample_bed, fea_bed, tmp_bed)
    check_output(cmd, shell=True)
    df = pd.read_csv(tmp_bed, header=None, sep='\t', usecols=[3, 7, 8, 9, 10])
    df.columns = ['group', 'type', 'snp', 'id', 'gc']
    df_freq = df.copy().loc[::, ['group', 'id']]
    df_gc = df.copy().loc[::, ['group', 'gc']]
    df_gc_out = fea_mut_gc(df_gc)
    # df = df.drop_duplicates(subset=['group', 'id'], keep='first')
    df = df.loc[::, ['group', 'type', 'snp']]
    df = fea_mut_all(df, 'freq')
    df_add = fea_mut_mean(df_freq)
    df = pd.concat([df, df_add, df_gc_out], axis=1)
    return df


def read_score(file_name, mode):
    if mode == 'DriverPower':
        df = pd.read_csv(file_name, header=0, index_col=0, sep='\t')
        df = df.loc[::, ['q-value']]
    elif mode == 'ExInAtor' or mode == 'compositeDriver' or mode == 'oncodriveFML_cadd' or mode == 'oncodriveFML_vest3' or mode == 'regDriver':
        df = pd.read_csv(file_name, header=0, index_col=3, sep='\t')
        df = df.iloc[::, [-1]]
    elif mode == 'dNdScv':
        df = pd.read_csv(file_name, header=0, index_col=3, sep='\t')
        df = df.loc[::, ['p-value']]
    elif mode == 'ActiveDriverWGS' or mode == 'Mutsig' or mode == 'LARVA' or mode == 'NBR' or mode == 'ncDriver_combined' or mode == 'ncdDetect':
        df = pd.read_csv(file_name, header=0, index_col=0, sep='\t')
        df = df.iloc[::, [-1]]
    df.columns = ['q']
    df['q'] = df['q'].apply(lambda x: max(1e-16, x))
    df['q'] = df['q'].apply(lambda x: - math.log10(x))
    df = df.fillna(0)
    return df


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='iVariant v0.01.')
    parser.add_argument("-i", dest='input', default="../../data/ICGC/p-values/observed/chr_id.txt", help="clinvar_pos")
    parser.add_argument("-m", dest='mode', default="sim_mut", help="mode")
    parser.add_argument("-t", dest='tumor', default="Pancan", help="clinvar_pos")
    args = parser.parse_args()
    chr_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                 '19',
                 '20', '21', '22', 'X', 'Y', ]
    fea_data = ['mut', 'cna', 'rna']
    tumors_file = '../tumors.txt'
    samples_file = '../../data/ICGC/samples_info.csv'
    tumors_set = {'Pancan': 'Pancan'}
    for line in open(tumors_file, 'rt'):
        txt = line.rstrip().split('\t')
        tumors_set[txt[0]] = txt[1]
    path = './%s/' % tumors_set[args.tumor]
    if not exists(path):
        os.makedirs(path)
    df_samples = pd.read_csv(samples_file, header=0, sep=',',
                             usecols=['tumour_specimen_aliquot_id', 'normal_specimen_aliquot_id',
                                      'histology_abbreviation', 'submitted_donor_id'])
    if args.tumor != 'Pancan':
        df_samples = df_samples[df_samples['histology_abbreviation'] == args.tumor]
    else:
        plist = ['Ovary-AdenoCA', 'CNS-Medullo', 'Biliary-AdenoCA', 'Breast-AdenoCA', 'CNS-PiloAstro', 'Bone-Osteosarc',
                 'Myeloid-AML', 'Bone-Epith', 'Liver-HCC', 'Prost-AdenoCA', 'Panc-Endocrine', 'Breast-LobularCA',
                 'Myeloid-MDS', 'Head-SCC', 'Eso-AdenoCA', 'Stomach-AdenoCA', 'Panc-AdenoCA', 'Bone-Cart',
                 'Breast-DCIS', 'Myeloid-MPN', 'Kidney-RCC']
        df_samples = df_samples[df_samples['histology_abbreviation'].isin(plist)]
    sample_ids = df_samples.loc[::, 'tumour_specimen_aliquot_id'].values.astype(str)
    rna_ids = df_samples.loc[::, 'submitted_donor_id'].values.astype(str)
    samples = {}
    rnas = {}
    for id in sample_ids:
        samples[id] = 1
    for i in range(len(sample_ids)):
        rnas[rna_ids[i]] = sample_ids[i]
    dfs = []
    df0 = pd.read_csv(args.input, header=0, index_col=3, sep='\t')
    df0 = df0.loc[::, []]
    id_tmp = list(pd.read_csv(args.input, header=0, index_col=3, sep='\t').index)
    ids = {}
    for id in id_tmp:
        ids[id] = 1
    outfile = './%s/%s.fea' % (path, args.mode)
    if args.mode == 'mut':
        input_file = './simulation.txt.gz'
        df = pd.read_csv(input_file, header=0, sep='\t',
                         usecols=['Chromosome', 'Start_position', 'End_position', 'Variant_Classification',
                                  'Variant_Type', 'Tumor_Sample_Barcode', 'gc_content'])
        df.columns = ['chr', 'start', 'end', 'type', 'snp', 'id', 'gc']
        df = df.loc[df['id'].isin(samples), :]
        print(df.shape[0])
        X = fea_mut(df)
        # X = pd.concat([X1, X2], axis=1)
        print(X.shape)
    elif args.mode == 'cna':
        input_file = '../%s/%s.fea' % (path, args.mode)
        outfile = './%s/%s.fea' % (path, args.mode)
        df = pd.read_csv(input_file, header=0, index_col=0, sep='\t')
        # mat = df.iloc[:, :].values.astype(float)
        # np.random.shuffle(mat)
        # df.iloc[:, :] = mat
        # col_names = list(df)
        # cols = []
        # for col in col_names:
        #     if '_var' in col:
        #         cols.append(col)
        # df.loc[::, cols] = df.loc[::, cols] / 10
        df.to_csv(outfile, header=True, index=True, sep='\t')
        return
    elif args.mode == 'rna':
        input_file = '../%s/%s.fea' % (path, args.mode)
        outfile = './%s/%s.fea' % (path, args.mode)
        df = pd.read_csv(input_file, header=0, index_col=0, sep='\t')
        # mat = df.iloc[:, :].values.astype(float)
        # np.random.shuffle(mat)
        # df.iloc[:, :] = mat
        # col_names = list(df)
        # cols = []
        # for col in col_names:
        #     if '_var' in col:
        #         cols.append(col)
        # df.loc[::, cols] = df.loc[::, cols] / 10
        df.to_csv(outfile, header=True, index=True, sep='\t')
        return
    x_ids = list(X.index)
    fea_ids = []
    for id in x_ids:
        if id in ids:
            fea_ids.append(id)
    cols = list(X)
    print(cols)
    for col in cols:
        df0[col] = 0.0
        df0.loc[fea_ids, col] = X.loc[fea_ids, col].values.astype(float)
    df0 = df0.fillna(0.0)
    # df_tmp = df0.sort_values(by=['q'], ascending=[False])
    # print(df_tmp.head(5))
    df0.to_csv(outfile, header=True, index=True, sep='\t')


if __name__ == "__main__":
    main()
