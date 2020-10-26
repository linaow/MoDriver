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
from os.path import splitext, basename, exists, abspath, isfile, getsize


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='iVariant v0.01.')
    parser.add_argument("-p", dest='path',
                        default="all.list",
                        help="meth input")
    parser.add_argument("-i", dest='input', default="./tumors.txt", help="clinvar_pos")
    parser.add_argument("-m", dest='mode', default="merge", help="clinvar_pos")
    args = parser.parse_args()
    tumors_file = args.input
    tumors_set = {'Pancan': 'Pancan'}
    for line in open(tumors_file, 'rt'):
        txt = line.rstrip().split('\t')
        tumors_set[txt[0]] = txt[1]
    # mode_all = ['mut', 'cna', 'rna', 'DriverPower', 'ExInAtor', 'compositeDriver', 'oncodriveFML_cadd',
    #             'oncodriveFML_vest3', 'regDriver', 'ActiveDriverWGS', 'Mutsig', 'LARVA', 'NBR', 'ncDriver_combined',
    #             'ncdDetect']
    mode_all = ['mut', 'cna', 'rna']
    if args.mode == 'merge':
        id_file = '../data/ICGC/p-values/observed/chr_id.txt'
        df0 = pd.read_csv(id_file, header=0, index_col=3, sep='\t')
        df0 = df0.loc[::, []]
        idx = list(df0.index)
        t_start = 'Pancan'
        for tumor in tumors_set.keys():
            if tumor != t_start:
                continue
            path = './%s/' % tumors_set[tumor]
            dfs = [df0]
            for mode in mode_all:
                fea_one = './%s/%s.fea' % (path, mode)
                dfs.append(pd.read_csv(fea_one, header=0, index_col=0, sep='\t'))
            df_all = pd.concat(dfs, axis=1)
            out_file = './%s/all.fea' % (path)
            print(df_all.head(5))
            df_all.to_csv(out_file, header=True, index=True, sep='\t')
            return
    else:
        t_start = 'Pancan'
        for tumor in tumors_set.keys():
            if tumor != t_start:
                continue
            for mode in mode_all:
                cmd = 'python fea.py -m %s -t %s' % (mode, tumor)
                print(cmd)
                check_output(cmd, shell=True)
            return


if __name__ == "__main__":
    main()
