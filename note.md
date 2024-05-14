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

###  LTP（推荐）：

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

### × DeepKE（可能）：

及其拉垮的模型以及学术造假的可能，要自己训练自己的模型

***

### × jiagu（可能）：

乐色模型啥也跑不了，但是用着是真方便

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

### 对于LTP模型的说明

#### 1. 分词对照表

1. `a`：形容词
2. `b`：区别词
3. `c`：连词
4. `d`：副词
5. `e`：叹词
6. `g`：语素词
7. `h`：前接成分
8. `i`：成语
9. `j`：简称略语
10. `k`：后接成分
11. `m`：数词
12. `n`：名词
13. `nd`：方位名词
14. `nh`：人名
15. `ni`：机构名
16. `nl`：名词性惯用语
17. `ns`：地名
18. `nt`：时间名词
19. `nz`：其他专有名词
20. `o`：拟声词
21. `p`：介词
22. `q`：量词
23. `r`：代词
24. `u`：助词
25. `v`：动词
26. `wp`：标点符号
27. `ws`：其他

#### 2. 语义依存对照表

施事关系	Agt	Agent	我送她一束花 (我 <-- 送)
当事关系	Exp	Experiencer	我跑得快 (跑 --> 我)
感事关系	Aft	Affection	我思念家乡 (思念 --> 我)
领事关系	Poss	Possessor	他有一本好读 (他 <-- 有)
受事关系	Pat	Patient	他打了小明 (打 --> 小明)
客事关系	Cont	Content	他听到鞭炮声 (听 --> 鞭炮声)
成事关系	Prod	Product	他写了本小说 (写 --> 小说)
源事关系	Orig	Origin	我军缴获敌人四辆坦克 (缴获 --> 坦克)
涉事关系	Datv	Dative	他告诉我个秘密 ( 告诉 --> 我 )
比较角色	Comp	Comitative	他成绩比我好 (他 --> 我)
属事角色	Belg	Belongings	老赵有俩女儿 (老赵 <-- 有)
类事角色	Clas	Classification	他是中学生 (是 --> 中学生)
依据角色	Accd	According	本庭依法宣判 (依法 <-- 宣判)
缘故角色	Reas	Reason	他在愁女儿婚事 (愁 --> 婚事)
意图角色	Int	Intention	为了金牌他拼命努力 (金牌 <-- 努力)
结局角色	Cons	Consequence	他跑了满头大汗 (跑 --> 满头大汗)
方式角色	Mann	Manner	球慢慢滚进空门 (慢慢 <-- 滚)
工具角色	Tool	Tool	她用砂锅熬粥 (砂锅 <-- 熬粥)
材料角色	Malt	Material	她用小米熬粥 (小米 <-- 熬粥)
时间角色	Time	Time	唐朝有个李白 (唐朝 <-- 有)
空间角色	Loc	Location	这房子朝南 (朝 --> 南)
历程角色	Proc	Process	火车正在过长江大桥 (过 --> 大桥)
趋向角色	Dir	Direction	部队奔向南方 (奔 --> 南)
范围角色	Sco	Scope	产品应该比质量 (比 --> 质量)
数量角色	Quan	Quantity	一年有365天 (有 --> 天)
数量数组	Qp	Quantity-phrase	三本书 (三 --> 本)
频率角色	Freq	Frequency	他每天看书 (每天 <-- 看)
顺序角色	Seq	Sequence	他跑第一 (跑 --> 第一)
描写角色	Desc(Feat)	Description	他长得胖 (长 --> 胖)
宿主角色	Host	Host	住房面积 (住房 <-- 面积)
名字修饰角色	Nmod	Name-modifier	果戈里大街 (果戈里 <-- 大街)
时间修饰角色	Tmod	Time-modifier	星期一上午 (星期一 <-- 上午)
反角色	r + main role		打篮球的小姑娘 (打篮球 <-- 姑娘)
嵌套角色	d + main role		爷爷看见孙子在跑 (看见 --> 跑)
并列关系	eCoo	event Coordination	我喜欢唱歌和跳舞 (唱歌 --> 跳舞)
选择关系	eSelt	event Selection	您是喝茶还是喝咖啡 (茶 --> 咖啡)
等同关系	eEqu	event Equivalent	他们三个人一起走 (他们 --> 三个人)
先行关系	ePrec	event Precedent	首先，先
顺承关系	eSucc	event Successor	随后，然后
递进关系	eProg	event Progression	况且，并且
转折关系	eAdvt	event adversative	却，然而
原因关系	eCau	event Cause	因为，既然
结果关系	eResu	event Result	因此，以致
推论关系	eInf	event Inference	才，则
条件关系	eCond	event Condition	只要，除非
假设关系	eSupp	event Supposition	如果，要是
让步关系	eConc	event Concession	纵使，哪怕
手段关系	eMetd	event Method	
目的关系	ePurp	event Purpose	为了，以便
割舍关系	eAban	event Abandonment	与其，也不
选取关系	ePref	event Preference	不如，宁愿
总括关系	eSum	event Summary	总而言之
分叙关系	eRect	event Recount	例如，比方说
连词标记	mConj	Recount Marker	和，或
的字标记	mAux	Auxiliary	的，地，得
介词标记	mPrep	Preposition	把，被
语气标记	mTone	Tone	吗，呢
时间标记	mTime	Time	才，曾经
范围标记	mRang	Range	都，到处
程度标记	mDegr	Degree	很，稍微
频率标记	mFreq	Frequency Marker	再，常常
趋向标记	mDir	Direction Marker	上去，下来
插入语标记	mPars	Parenthesis Marker	总的来说，众所周知
否定标记	mNeg	Negation Marker	不，没，未
情态标记	mMod	Modal Marker	幸亏，会，能
标点标记	mPunc	Punctuation Marker	，。！
重复标记	mPept	Repetition Marker	走啊走 (走 --> 走)
多数标记	mMaj	Majority Marker	们，等
实词虚化标记	mVain	Vain Marker	
离合标记	mSepa	Seperation Marker	吃了个饭 (吃 --> 饭) 洗了个澡 (洗 --> 澡)
根节点	Root	Root	全句核心节点

# 研究日志

### 2024.5.3

直接通过LTP分词：

```python
命名实体识别：[[]]
分词：[['由于', '不同', '的', '岩体', '特征', '（', '软弱', '夹层', '、', '节理', '裂隙', '、', '地下水', '、', '岩体表观', '结构', '等', '）', '识别', '涉及', '方法', '有', '差别', '，', '故', '不', '对', '每', '一', '类', '岩体', '特征', '所', '采用', '的', 'CNN', '模型', '结
构', '进行', '详细', '分析', '。']]
```

可以看到像“节理裂隙”被分成了“节理”和“裂隙“

解决办法：

```python
ltp.add_words(["节理裂隙", "软弱夹层"], freq=2)
```

输出：

```python
分词：[['由于', '不同', '的', '岩体', '特征', '（', '软弱夹层', '、', '节理裂隙', '、', '地下水', '、', '岩体表观', '结构', '等', '）', '识
别', '涉及', '方法', '有', '差别', '，', '故', '不', '对', '每', '一', '类', '岩体', '特征', '所', '采用', '的', 'CNN', '模型', '结构', '进
行', '详细', '分析', '。']]
```

