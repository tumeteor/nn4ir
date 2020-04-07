import warc
from pyspark import SparkContext, SparkConf


class WarcUtil:
    def __init__(self):
        self.conf = SparkConf()
        self.sc = SparkContext(conf=conf)

    def load_hdfs_files(self):
        hadoop = self.sc._jvm.org.apache.hadoop
        fs = hadoop.fs.FileSystem
        conf = hadoop.conf.Configuration()
        path = hadoop.fs.Path('tmp/subset-data/top1k/')
        return fs.get(conf).listStatus(path)

    def parse_warcs(self):
        files = self.load_hdfs_files()
        for wf in files:
            wf_path = wf.getPath().toString()
            if not "DATA-EXTRACTION-PART" in wf_path: continue
            warcText = self.sc.textFile(wf_path)
            f = warc.open(warcText)
            for record in f:
                print record['WARC-Target-URI']


if __name__ == '__main__':
    util = WarcUtil()

    util.load_hdfs_files()
    util.parse_warcs()
