# 使用说明

### 1. l2neo.py

手写封装模块，给予其三元组，执行主函数即可完成后续任务，只需将目光集中在三元组的获取方式上即可。

使用例：

```python
list_syz = [
        {'subject': 'Barack Obama', 'relation': 'was born in', 'object': 'Hawaii'},
        {'subject': 'Richard Manning', 'relation': 'wrote', 'object': 'sentence'},
        ]
a = l2neo(list_syz)
a.go()
```

### 2. from svoUtils.svo_cn import SvoCn

手写封装模块，给予其文本，返回事件三元组。

使用例：

```python
if __name__=='__main__':
    a = SvoCn()
    print(a.deal('李克强总理今天来我家了,我感到非常荣幸'))
    
# output
[{'subject': '李克强总理', 'relation': '来', 'object': '我家'}, {'subject': '我', 'relation': '感到', 'object': '荣幸'}]
```

### 3. from reUtils.re_en import ReEn

手写封装模块，给予其文本，返回关系三元组。

使用例：

```python
if __name__ == '__main__':
    a = ReEn()
    print(a.deal('Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'))
    
# output
[{'subject': 'Barack Obama', 'relation': 'was', 'object': 'born'}, {'subject': 'Barack Obama', 'relation': 'was born in', 'object': 'Hawaii'}, {'subject': 'Richard Manning', 'relation': 'wrote', 'object': 'sentence'}]
```

### 4.from pdfeUtils.pdfe import PdfE

手写封装模块，给予PDF路径，返回纯文本。

使用例：

```python
if __name__ == '__main__':
    a = PdfE()
    print(a.deal("pdfSQL/topdf.pdf"))
    
# output
卷积神经网络（ＣｏｎｖｏｌｕｔｉｏｎａｌＮｅｕｒａｌＮｅｔｗｏｒｋ，ＣＮＮ）是一类含卷积运算并具深度结构的前馈神经网络，是深度学习的代表算 法之一．传统的神经网络普遍采用的全连接方式会导致参数量巨大，网络训练时间长、耗能高，甚至难以训练等问题．而ＣＮＮ则通过积运算实现了神经元 的局部连接和权值共享，即它是一种不完全连接的网络，这极大地降低了网络的训练难度，也提升了模型的综合表现．故主要用ＣＮＮ算法实现目标图像的 识别和特征提取．在ＣＮＮ输入层与输出层之间常具有多个隐含层，这些隐含层主要包括卷积层、池化层、全连接层归一化层等，此外还可能有反卷积层． 另外，除初始化选择模型内部函数及求解方式外，还需进行参数的初始化设置．为区别模型训练完成后的重、偏置等参数，初始化设置参数被称为超参数， 其中主要包括学习率、批次、迭代次数、权重衰减、动量等．由于不同的岩体特征（软弱夹层、节裂隙、地下水、岩体表观结构等）识别涉及方法有差别， 故不对每一类岩体特征所采用的ＣＮＮ模型结构进行详细分析．
```

