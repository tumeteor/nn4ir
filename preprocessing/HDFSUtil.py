import warc
from pyspark import SparkContext, SparkConf
class WarcUtil:

    conf = SparkConf()
    sc = SparkContext(conf=conf)

    def loadHDFSFiles(self):
        hadoop = self.sc._jvm.org.apache.hadoop
        fs = hadoop.fs.FileSystem
        conf = hadoop.conf.Configuration()
        path = hadoop.fs.Path('tmp/subset-data/top1k/')
        return fs.get(conf).listStatus(path)


    def parseWarcs(self):
        files = self.loadHDFSFiles()
        for wf in files:
            wf_path = wf.getPath().toString()
            if not "DATA-EXTRACTION-PART" in wf_path: continue
            warcText = self.sc.textFile(wf)
            f = warc.open(warcText)
            for record in f:
                print record['WARC-Target-URI']

if __name__ == '__main__':
    util = WarcUtil()

    util.loadHDFSFiles()
    util.parseWarcs()








