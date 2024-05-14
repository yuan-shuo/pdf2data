from svoUtils.triple_extraction import *

class SvoCn:
    def __init__(self):
        self.id = 'svo'
    def deal(self, text):
        self.extractor = TripleExtractor()
        content = text
        self.svos = self.extractor.triples_main(content)
        self.cur = []
        for i in self.svos:
            self.cur.append({'subject': i[0], 'relation': i[1], 'object': i[2]})
        return self.cur

if __name__=='__main__':
    a = SvoCn()
    print(a.deal('李克强总理今天来我家了,我感到非常荣幸'))