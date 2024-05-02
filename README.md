# PDF提取三元组 

* 使用语言 => (Python+Java（×）)

* 脚本语言 => (Python)

* 主要的库 => (OpenIE、LTP、Stanford CoreNLP（×）、PyPDF2)

***

###  OpenIE：

1. 处理纯英文文本

2. Github：https://github.com/philipperemy/stanford-openie-python/tree/fix1
3. code-EX：

```python
from openie import StanfordOpenIE

# https://stanfordnlp.github.io/CoreNLP/openie.html#api
# Default value of openie.affinity_probability_cap was 1/3.
properties = {
    'openie.affinity_probability_cap': 2 / 3,
}

with StanfordOpenIE(properties=properties) as client:
    text = 'Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'
    text = 'The CNN algorithm is mainly used to realize the recognition and feature extraction of target images'
    print('Text: %s.' % text)
    for triple in client.annotate(text):
        print('|-', triple)
        
# 输出：
|- {'subject': 'CNN algorithm', 'relation': 'is used', 'object': 'realize'}
|- {'subject': 'CNN algorithm', 'relation': 'is', 'object': 'used'}
|- {'subject': 'CNN algorithm', 'relation': 'realize', 'object': 'recognition'}      
|- {'subject': 'CNN algorithm', 'relation': 'is', 'object': 'mainly used'}
|- {'subject': 'CNN algorithm', 'relation': 'is mainly used', 'object': 'realize'}   
|- {'subject': 'CNN algorithm', 'relation': 'is mainly used', 'object': 'to realize'}
|- {'subject': 'CNN algorithm', 'relation': 'is used', 'object': 'to realize'} 
```

***

### LTP：

1. 中英文通用
2. Github：https://github.com/HIT-SCIR/ltp
3. code-EX：

```python
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
print(f"分词：{output.cws}") # 索引从0开始
print(f"语义依存分析图：{output.sdpg}") # 索引会因为虚节点+1，里面的1就是cws的0 => 1：他，2：叫，3：汤姆

# 输出
命名实体识别：[[('Nh', '汤姆')]]
分词：[['他', '叫', '汤姆', '去', '拿', '外衣', '。']]
语义依存分析图：[[(1, 2, 'AGT'), (2, 0, 'Root'), (3, 2, 'DATV'), (3, 4, 'AGT'), (4, 2, 'eSUCC'), (5, 2, 'eSUCC'), (5, 4, 'eSUCC'), (6, 5, 'PAT'), (7, 2, 'mPUNC')]]
```

***

###  × Stanford CoreNLP（废案）：

1. 处理中文(Main)、阿拉伯语、法语等
2. 中文教程：https://blog.csdn.net/weixin_46570668/article/details/116478666
3. Release：https://stanfordnlp.github.io/CoreNLP/history.html
4. code-EX：

```python
from stanfordcorenlp import StanfordCoreNLP

#如果要用其他语言，需要单独设置
nlp_ch= StanfordCoreNLP(r'E:\python_work\pyneo\pdfToData\myPDF\models\stanford-corenlp-full-2018-02-27', lang='zh')


sen='我是一名交大的学生'

print(nlp_ch.pos_tag(sen))
print(nlp_ch.parse(sen))

# 输出：
[('我', 'PN'), ('是', 'VC'), ('一', 'CD'), ('名', 'M'), ('交大', 'NR'), ('的', 'DEG'), ('学生', 'NN')]
(ROOT
  (IP
    (NP (PN 我))
    (VP (VC 是)
      (NP
        (QP (CD 一)
          (CLP (M 名)))
        (DNP
          (NP (NR 交大))
          (DEG 的))
        (NP (NN 学生))))))
```

***

### 对于使用斯坦福模型的说明

#### 1. 功能

   **Tokenize（分词）**：将输入的文本按照空格或其他规定的方式分割成单词或标记的过程。在这个例子中，输入的句子被分割成了一个个单词或短语。

   **Part of Speech（词性标注）**：为句子中的每个单词标注其词性（如名词、动词、形容词等）的过程。在这个例子中，每个单词都被标注了相应的词性。

   **Named Entities（命名实体识别）**：识别文本中具有特定意义的命名实体（如人名、地名、组织机构等）的过程。在这个例子中，被标记为命名实体的是"Guangdong University of Foreign Studies"和"Guangzhou"。

   **Constituency Parsing（组成句法分析）**：分析句子的结构，将句子分解成词组（短语）的过程。在这个例子中，句子被分解成了名词短语、介词短语和动词短语等。

   **Dependency Parsing（依存句法分析）**：分析句子中单词之间的依存关系，即单词之间的语法关系，如主谓关系、修饰关系等。在这个例子中，以树状结构表示了单词之间的依存关系。

- **五种功能对应的调用函数：**

```python
print 'Tokenize:', nlp.word_tokenize(sentence)
print 'Part of Speech:', nlp.pos_tag(sentence)
print 'Named Entities:', nlp.ner(sentence)
print 'Constituency Parsing:', nlp.parse(sentence)
print 'Dependency Parsing:', nlp.dependency_parse(sentence)
```

#### 2. parse代号

* (ROOT): 句子的根节点，表示整个句子的结构。
* (IP): 简单句或独立主格结构，表示一个完整的简单句。
* (NP): 名词短语，表示名词及其修饰词。
* (PN): 代词，指示特定的人或事物，如"我"、"他"等。
* (VP): 动词短语，表示动词及其补语。
* (VC): 系动词，用于表示主语的状态或性质，如"是"、"成为"等。
* (QP): 量词短语，表示数量词及其修饰。
* (CD): 数词，表示具体的数量，如"一"、"两"等。
* (CLP): 量词，用于表示数量的单位，如"个"、"位"等。
* (DNP): 冠词短语，表示名词前的修饰关系。
* (NR): 专有名词，表示特定的名词，如地名、人名等。
* (DEG): 的，表示所属关系，连接名词与其修饰成分。
* (NN): 名词，表示事物的名称，如"学生"、"教师"等。
