from pdfeUtils.pdfe import PdfE # PDF抽取文本工具
from svoUtils.svo_cn import SvoCn # 事件抽取工具
from reUtils.re_en import ReEn # 关系抽取工具
from l2neo import l2neo # 建库工具

class MainNeo:
    def __init__(self, lan='cn') -> None:
        self.pdfe = PdfE() # PDF抽取文本工具
        if lan == 'cn':     
            self.ex = SvoCn() # 事件抽取工具
        elif lan == 'en':
            self.ex = ReEn() # 关系抽取工具
        else:
            self.ex = ReEn() # 关系抽取工具
    def run(self, path):
        neo = l2neo(self.ex.deal(self.pdfe.deal(path)))
        neo.go()
        print("PDF Extraction finshed!")

if __name__ == '__main__':
    a=MainNeo(lan='en')
    a.run("pdfSQL/topdfen.pdf")