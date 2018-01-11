from surt import surt
from warcio.archiveiterator import ArchiveIterator
import os
import io
import pandas as pd
import csv
import justext
import codecs
from os.path import basename
import bz2

class WarcParser:

    PATH_TO_WARCS = "/home/nguyen/nn4ir/data/top1k/"


    def combineUrls(self, dir):

        urls = []

        for subdir, dirs, files in os.walk(dir):
            for file in files:
                file.title()
                csv_reader = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
                for row in csv_reader:
                    urls.append(row[1])

        return urls

    def combineUrlLabels(self, dir):
        qIds = {}
        with open("", "r") as f:
            csv_reader = csv.reader(f, delimiter='\t', quouting=csv.QUOTE_NONE)
            for row in csv_reader:
                qIds[row[1]] = row[0]

        with open("","r") as f:
            for subdir, dirs, files in os.walk(dir):
                for file in files:
                    csv_reader = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
                    for row in csv_reader:
                        f.write('\t'.join((qIds[file.name], row[0], row[1])))
                        f.write('\n')




    def trimUrlsFromCompressedFiles(self, dir):

        for subdir, dirs, files in os.walk(dir):
            for file in files:
                file1000 = basename(file) + ".txt"
                output_file = codecs.open(file1000, 'w+', 'utf-8')
                source_file = bz2.BZ2File(file, "r")
                count = 0
                for line in source_file:
                    count += 1
                    if count <= 1000:
                        output_file.write(line)
                source_file.close()
                output_file.close()


    def getUrlContentsPerQuery(self):
        pass



    def extractContentFromWarc(self, dir):

        url_content_dict = {}

        for filename in os.listdir(dir):
            if not filename.endswith(suffix="arc.gz"): continue
            with open(os.path.join(dir, filename), 'rb') as stream:
                for record in ArchiveIterator(stream):
                    if record.rec_type == 'warcinfo':
                        print(record.raw_stream.read())

                    elif record.rec_type == 'response':
                        if record.http_headers is None:
                            continue
                        if record.http_headers.get_header('Content-Type') == 'text/html':
                            uri = record.rec_headers.get_header('WARC-Target-URI')
                            timestamp = record.rec_headers.get_header('')
                            ts = pd.Timestamp(timestamp)
                            surtUri = surt(uri)
                            html_content = record.content_stream().read()
                            if html_content is None: continue


                            if surtUri in url_content_dict.keys():
                                if url_content_dict[surtUri][0] >= ts:
                                    continue

                            try:
                                #extractor = Extractor(extractor='ArticleExtractor', html=html_content)
                                #content = extractor.getText()
                                content = ""
                                paragraphs = justext.justext(html_content, justext.get_stoplist("German"))
                                for paragraph in paragraphs:
                                    if not paragraph.is_boilerplate:
                                        content += paragraph
                                        content += "\n"

                                url_content_dict[surtUri] = tuple([ts, content])
                            except TypeError as terr:
                                print uri
                            except UnicodeDecodeError as unierr:
                                print uri



        with io.open('docs.top1k.tsv', 'w', encoding='utf8') as csv_file:
            writer = csv.writer(csv_file,delimiter="\t")
            for url, doc in url_content_dict.items():
                writer.writerow([url, doc])



if __name__ == '__main__':
    parser = WarcParser()
    parser.extractContentFromWarc(dir=parser.PATH_TO_WARCS)






