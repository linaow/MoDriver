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
    parser.add_argument("-i", dest='input', default="samples_info.csv",
                        help="clinvar_pos")
    parser.add_argument("-o", dest='output', default="tumors.txt", help="clinvar_pos")
    args = parser.parse_args()
    df = pd.read_csv(args.input, header=0, sep=',', usecols=['histology_abbreviation'])
    df = df.drop_duplicates(subset=['histology_abbreviation'], keep='first')
    df = df.sort_values(by=['histology_abbreviation'], ascending=[True])
    print(df.shape)
    df.to_csv(args.output, header=False, index=False, sep='\t')

if __name__ == "__main__":
    main()
