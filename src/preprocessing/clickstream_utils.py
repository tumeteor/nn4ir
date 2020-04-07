from __future__ import print_function
from argparse import ArgumentParser


def read_tsv(file="2015_02_clickstream.tsv"):
    import csv
    with open(file, 'rb') as clk:
        clk = csv.reader(clk, delimiter='\t', quoting=csv.QUOTE_NONE)
        clk = list(clk)
        return clk


def rank_outlink(clk, srcEntity="87th_Academy_Awards"):
    from collections import defaultdict
    en_dict = defaultdict(int)
    for row in clk:
        if row[3] == srcEntity:
            en_dict[row[4]] = int(row[2])

    # sort reversely
    for w in sorted(en_dict, key=en_dict.get, reverse=True):
        print(w, en_dict[w])


if __name__ == '__main__':
    parser = ArgumentParser(description='Required arguments')
    parser.add_argument('-p', '--path', help='File path', required=False)
    parser.add_argument('-e', '--entity', help='Source entity', required=False)
    args = parser.parse_args()

    clk = read_tsv(file=args.path)
    rank_outlink(clk, srcEntity=args.entity)
