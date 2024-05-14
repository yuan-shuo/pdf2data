import torch
from ltp import LTP, StnSplit

ltp = LTP(r"E:\python_work\pyneo\pdfToData\myPDF\models\tiny")  # 默认加载 Small 模型
                        # 也可以传入模型的路径，ltp = LTP("/path/to/your/model")
                        # /path/to/your/model 应当存在 config.json 和其他模型文件

# 将模型移动到 GPU 上
if torch.cuda.is_available():
    # ltp.cuda()
    ltp.to("cuda")

# 自定义词表
# ltp.add_word("汤姆去", freq=2)
ltp.add_words(["隧道围岩分级", "yuan"], freq=2)
sents = StnSplit().split(
    "研究成果可以为冰水堆积体隧道围岩的分级、支护力的确定等提供参考。奥巴马是美国的总统。"
    )
print(sents)

output = ltp.pipeline(sents, tasks=["cws", "dep"])
print(f"分词：{output.cws}")
print(f"依存句法分析：{output.dep}")


# 分词 cws、词性 pos、命名实体标注 ner、语义角色标注 srl、依存句法分析 dep、语义依存分析树 sdp、语义依存分析图 sdpg
# def sanyuan(word, rel):
#     cur = []
#     for i in rel:
#         if i[2] != 'mPUNC' and i[1]!=0 and i[0]!=0:
#             cur.append({'subject': word[i[0]-1], 'relation': i[2], 'object': word[i[1]-1]})
#     return cur
# class LtpModel:
#     def __init__(self, text=None, wordList=None):
#         self.text = text
#         self.wordList = wordList

#     def re(self):
#         if self.text:
#             self.ltp = LTP(r"E:\python_work\pyneo\pdfToData\myPDF\models\tiny")

#             if torch.cuda.is_available():
#                 self.ltp.to("cuda")
#             if self.wordList:
#                 self.ltp.add_words(self.wordList, freq=2)

#             self.sents = StnSplit().split(self.text)
#             print(f"处理语句：{self.sents}")
#             self.output = ltp.pipeline(sents, tasks=["cws", "sdpg"])









