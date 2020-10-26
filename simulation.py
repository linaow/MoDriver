import argparse
import sys
import numpy as np
import random
nb_seed = 2
random.seed(nb_seed)
np.random.seed(nb_seed)
import os, time
import re
import tempfile
from tempfile import mkdtemp
import pickle
from subprocess import Popen, check_output
import pandas as pd
import gzip
import pysam
from os.path import splitext, basename, exists, abspath, isfile, getsize


def anno(file_in, file_out, cols=[]):
    bin_path = '/soft/a/envs/Oncotator/bin/Oncotator'
    anno_path = '/data/oncotator_v1_ds_April052016'
    cmd = '%s -v --db-dir %s %s %s hg19 --output_format=TCGAMAF --tx-mode=EFFECT &> /dev/null' % (
        bin_path, anno_path, file_in, file_out)
    check_output(cmd, shell=True)
    df = pd.read_csv(file_out, sep='\t', header=0, comment='#', usecols=cols)
    return df


def sample(seq, str, pos, win_len, drop_len):
    start = pos - win_len
    stop = pos + win_len
    drop_start = pos - drop_len
    drop_end = pos + drop_len
    pos_all = []
    for i in range(start, stop - 2):
        if seq[i - start:i - start + 3] == str and (i < drop_start or i > drop_end):
            pos_all.append(i + 1)
    random_pos = random.choice(pos_all)
    return random_pos


def sample_fast(seq, str, pos, win_len, drop_len, start, stop):
    begin = max(pos - (win_len - drop_len), start + drop_len)
    end = min(pos + (win_len - drop_len), stop - drop_len)
    move = random.randrange(begin, end)
    # random pos
    if move > pos:
        move += drop_len
    else:
        move -= drop_len
    # move slightly to maintain the trinucleotide context of the mutation
    begin = pos - win_len
    end = pos + win_len
    # left search
    left_pos = -1
    for i in range(move, begin, -1):
        if seq[i - begin - 1:i - begin + 2] == str:
            left_pos = i
            break
    # right search
    right_pos = -1
    for i in range(move, end):
        x = i - begin
        if seq[i - begin - 1:i - begin + 2] == str:
            right_pos = i
            break
    random_pos = -1
    if left_pos != -1 and right_pos != -1:
        if abs(left_pos - move) < abs(right_pos - move):
            random_pos = left_pos
        else:
            random_pos = right_pos
    elif left_pos != -1:
        random_pos = left_pos
    elif right_pos != -1:
        random_pos = right_pos
    else:
        random_pos = move
    return random_pos


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='iVariant v0.01.')
    parser.add_argument("-i", dest='input', default="./input.maf",
                        help="annotation input")
    parser.add_argument("-o", dest='out_path', default="./output.maf", help="clinvar_pos")
    args = parser.parse_args()
    input_file = str(args.input)
    print(input_file)
    if "_" in input_file:
        id = int(input_file[input_file.rfind("_") + 1:len(input_file)])
        tmp_dir = '/data/tmp/%d/' % id
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
    else:
        tmp_dir = mkdtemp(dir='/data/tmp/')
    seqs = pysam.FastaFile("./hg19.fa")
    df_genome = pd.read_csv("./hg19.genome", header=None, index_col=0, sep='\t', dtype={0: str, 1: int})
    df_genome.columns = ['len']
    win_len = 50000
    drop_len = 50
    col0 = ['Chromosome', 'Start_position', 'End_position', 'Reference_Allele',
            'Tumor_Seq_Allele2', 'Tumor_Sample_Barcode', 'Matched_Norm_Sample_Barcode']
    df = pd.read_csv(args.input, header=None, sep='\t')
    df.columns = col0
    anno_input_file = '%s/anno_input.maf.txt' % tmp_dir
    anno_out_file = '%s/anno_out.maf.txt' % tmp_dir
    df.to_csv(anno_input_file, header=True, index=False, sep='\t')
    df = anno(anno_input_file, anno_out_file, col0 + ['Protein_Change'])
    df = df.astype(
        dtype={"Chromosome": str, "Start_position": int, "End_position": int,
               "Reference_Allele": str, "Tumor_Seq_Allele2": str, "Protein_Change": str})
    # remove all coding mutations
    df = df.loc[df['Protein_Change'] == 'nan', ::]
    # print("mutations: " + str(df.shape[0]))
    # set output base file
    df_out = df.copy().loc[::, col0]
    # randomly move
    df['chr'] = df['Chromosome'].apply(lambda x: 'chr' + str(x))
    df['start'] = df['Start_position'] - win_len
    df['end'] = df['Start_position'] + win_len
    for i, row in df.iterrows():
        chr = row['chr']
        start = row['start']
        end = row['end']
        pos = row['Start_position']
        pos2 = row['End_position']
        ref = row['Reference_Allele']
        alt = row['Tumor_Seq_Allele2']
        start = max(1, start)
        end = min(end, df_genome.at[chr, 'len'])
        # keep trinucleotide context
        # pysam 1-based fetch sequences
        seq = seqs.fetch(region="%s:%d-%d" % (chr, start, end)).upper()
        trinuc = seq[win_len - 1: win_len + 2]
        random_pos = sample_fast(seq, trinuc, pos, win_len, drop_len, start, end)
        random_pos2 = random_pos + (pos2 - pos)
        # insert
        if ref == '-':
            random_ref = ref
        # others
        else:
            random_ref = seqs.fetch(region="%s:%d-%d" % (chr, random_pos, random_pos + len(ref) - 1)).upper()
        random_alt = alt
        # update output df
        df_out.at[i, 'Start_position'] = random_pos
        df_out.at[i, 'End_position'] = random_pos2
        df_out.at[i, 'Reference_Allele'] = random_ref
        df_out.at[i, 'Tumor_Seq_Allele2'] = random_alt
    random_input_file = '%s/random_input.maf.txt' % tmp_dir
    random_out_file = '%s/random_out.maf.txt' % tmp_dir
    df_out.to_csv(random_input_file, header=True, index=False, sep='\t')
    del df_out, df
    # mutation annotation
    df = anno(random_input_file, random_out_file, col0 + ['Variant_Classification', 'Variant_Type', 'gc_content'])
    df.to_csv(args.out_path, header=False, index=False, sep='\t', float_format='%.3f')
    # clear
    cmd = 'rm -f %s %s %s %s oncotator.log' % (anno_input_file, anno_out_file, random_input_file, random_out_file)
    check_output(cmd, shell=True)


if __name__ == "__main__":
    main()
