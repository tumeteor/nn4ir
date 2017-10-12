from argparse import ArgumentParser

class Util:
    clk = {}

    def readTsv(self, file="2015_02_clickstream.tsv"):
        import csv
        with open(file, 'rb') as clk:
            clk = csv.reader(clk, delimiter='\t',quoting=csv.QUOTE_NONE)
            self.clk = list(clk)


    def rankOutlink(self, srcEntity="87th_Academy_Awards"):
        from collections import defaultdict
        enDict = defaultdict(int)
        for row in self.clk:
            if row[3] == srcEntity:
                enDict[row[4]] = int(row[2])

        # sort reversely
        for w in sorted(enDict, key=enDict.get, reverse=True):
            print w, enDict[w]


if __name__ == '__main__':
    parser = ArgumentParser(description='Required arguments')
    parser.add_argument('-p', '--path', help='File path', required=False)
    parser.add_argument('-e', '--entity', help='Source entity', required=False)
    args = parser.parse_args()
    util = Util()


    util.readTsv(file=args.path)
    util.rankOutlink(srcEntity=args.entity)

