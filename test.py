import torch
from ltp import LTP

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

#  分词 cws、词性 pos、命名实体标注 ner、语义角色标注 srl、依存句法分析 dep、语义依存分析树 sdp、语义依存分析图 sdpg
output = ltp.pipeline(["他叫汤姆去拿外衣。"], tasks=["cws", "pos", "ner", "srl", "dep", "sdp", "sdpg"])
# 使用字典格式作为返回结果
print(f"命名实体识别：{output.ner}")
print(f"分词：{output.cws}")
print(f"语义依存分析图：{output.sdpg}")