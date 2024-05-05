# import jiagu

# # 吻别是由张学友演唱的一首歌曲。
# # 《盗墓笔记》是2014年欢瑞世纪影视传媒股份有限公司出品的一部网络季播剧，改编自南派三叔所著的同名小说，由郑保瑞和罗永昌联合导演，李易峰、杨洋、唐嫣、刘天佐、张智尧、魏巍等主演。

# text = '1889年4月20日，希特勒出生于当时奥匈帝国的因河畔布劳瑙'
# knowledge = jiagu.knowledge(text)
# print(knowledge)
import torch
from ltp import LTP, StnSplit

# 默认 huggingface 下载，可能需要代理

ltp = LTP(r"E:\python_work\pyneo\pdfToData\myPDF\models\tiny")  # 默认加载 Small 模型
                        # 也可以传入模型的路径，ltp = LTP("/path/to/your/model")
                        # /path/to/your/model 应当存在 config.json 和其他模型文件

# 将模型移动到 GPU 上
if torch.cuda.is_available():
    # ltp.cuda()
    ltp.to("cuda")

# 自定义词表
# ltp.add_word("汤姆去", freq=2)
# ltp.add_words(["外套", "外衣"], freq=2)
# sents = StnSplit().split(
#     "Hawaii"
#     )
# print(sents)
sents = ["Barack Obama", "Hawaii"]

for text in sents:
    #  分词 cws、词性 pos、命名实体标注 ner、语义角色标注 srl、依存句法分析 dep、语义依存分析树 sdp、语义依存分析图 sdpg
    output = ltp.pipeline([text], tasks=["pos"])

    # 使用字典格式作为返回结果
    print(f"词性：{output.pos[0]}")
