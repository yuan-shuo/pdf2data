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

