from py2neo import Node, Graph, Relationship, NodeMatcher, Subgraph
import torch
from ltp import LTP, StnSplit

class l2neo:
    def __init__(self, syz, debug=False):
        self.syz = syz
        self.debug = debug

    # 启动！
    def go(self):
        # 实体抽取+去重
        self.list_ner = ner_extract(self.syz)
        # 实体分类
        self.list_type = ner_divide(self.list_ner)

        if self.debug:
            print("目前以debug模式运行ing，仅打印执行内容")
            print(f"实体列表：{self.list_ner}")
            print(f"实体类别：{self.list_type}")
        else:
            # 连接数据库并清空所有内容（要先启动数据库不然先注释掉）
            link = Graph("http://localhost:7474", auth=("neo4j", "174235"))
            self.graph = link

            warn = input('(__!慎重!__)是否删除原有数据库?(y/n)：')

            if warn == 'y':
                # 删库！慎用
                self.graph.delete_all()
                print("已清空原有数据")
            else:
                print("将保留原有数据库")

            # 绘制节点
            self.plot_point()
            # 绘制联系
            self.plot_relation()

            print("构建任务完成！")
            

    # 画点
    def plot_point(self):
        for i in range(len(self.list_ner)):
            node = Node(self.list_type[i], name=self.list_ner[i])
            # 绘制代码，debug时注释掉就行
            self.graph.create(node)

        print("节点绘制完成！")

    # 联系
    def plot_relation(self):
        for item in self.syz:
            subject = self.graph.nodes.match(name=item['subject']).first()
            object = self.graph.nodes.match(name=item['object']).first()
            rel = Relationship(subject, item['relation'], object)
            self.graph.create(rel)
        
        print("联系绘制完成！")

# 提取三元组实体 - 2个（sub+obj）
def ner_extract(list_syz):
    all_ner = []
    for item in list_syz:
        all_ner.append(item['subject'])
        all_ner.append(item['object'])
    # 用set去重省得占用LTP资源
    return list(set(all_ner))

# 实体类别提取 - LTP .GPU
def ner_divide(list_ner):
    all_ner_type = []
    ltp = LTP(r"E:\python_work\pyneo\pdfToData\myPDF\models\tiny")
    if torch.cuda.is_available():
        # ltp.cuda()
        ltp.to("cuda")
    for text in list_ner:
        output = ltp.pipeline([text], tasks=["pos"])
        ner_type = output.pos[0]
        # print(f"{text}的实体类别：{ner_type}")
        all_ner_type.append(ner_type)
    return all_ner_type

if __name__ == '__main__':
    list_syz = [
        {'subject': 'Barack Obama', 'relation': 'was born in', 'object': 'Hawaii'},
        {'subject': 'Richard Manning', 'relation': 'wrote', 'object': 'sentence'},
        ]
    a = l2neo(list_syz)
    a.go()




# # 联系节点的函数
# def link_point(point_name, level_list):
#     # 所有level 1=>0
#     node_a = graph.nodes.match(level='1').all()
#     node_b = graph.nodes.match(level='0').first()
#     for na in node_a:
#         rel_ab = Relationship(na, "评估技术", node_b)
#         graph.create(rel_ab)
#     node_a = graph.nodes.match(level='R').all()
#     node_b = graph.nodes.match(level='0').first()
#     for na in node_a:
#         rel_ab = Relationship(na, "专业关键词", node_b)
#         graph.create(rel_ab)
#     node_a = graph.nodes.match(level='F').all()
#     node_b = graph.nodes.match(level='0').first()
#     for na in node_a:
#         rel_ab = Relationship(na, "相关学科", node_b)
#         graph.create(rel_ab)

#     # 所有level 2=>1
#     for i in range(len(level_list)):
#         # 连接 0 1 父子节点
#         if level_list[i] == '1':
#             p_name = point_name[i]
#             node_a = graph.nodes.match(father=p_name).all()
#             node_b = graph.nodes.match(name=p_name).first()
#             if node_a:
#                 for na in node_a:
#                     rel_ab = Relationship(na, "评估参数", node_b)
#                     graph.create(rel_ab)

#         # 连接论文 主节点
#         if level_list[i] == 'J':
#             p_name = point_name[i]
#             node_a = graph.nodes.match(father=p_name).all()
#             node_b = graph.nodes.match(name=p_name).all()
#             if node_a:
#                 for nb in node_b:
#                     for na in node_a:
#                         rel_ab = Relationship(na, "领域内现有文献", nb)
#                         graph.create(rel_ab)
#         # 连接作者 论文
#         if level_list[i] == 'P':
#             p_name = point_name[i]
#             node_a = graph.nodes.match(father=p_name).all()
#             node_b = graph.nodes.match(name=p_name).all()
#             if node_a:
#                 for nb in node_b:
#                     for na in node_a:
#                         rel_ab = Relationship(na, "文献作者", nb)
#                         graph.create(rel_ab)

# def vtFunc(str1, str2, rel):
#     # 所有智能决策+学习算法+信息化=>人工智能
#     node_a = graph.nodes.match(valueType=str1).all()
#     node_b = graph.nodes.match(valueType=str2).all()
#     for nb in node_b:
#         for na in node_a:
#             rel_ab = Relationship(na, rel, nb)
#             graph.create(rel_ab)

# def vvFunc(str1, str2, rel):
#     # 所有智能决策+学习算法+信息化=>人工智能
#     node_a = graph.nodes.match(str1).all()
#     node_b = graph.nodes.match(valueType=str2).all()
#     for nb in node_b:
#         for na in node_a:
#             rel_ab = Relationship(na, rel, nb)
#             graph.create(rel_ab)

# def vrFunc(str1, str2, rel):
#     # 所有智能决策+学习算法+信息化=>人工智能
#     node_a = graph.nodes.match(str1).all()
#     node_b = graph.nodes.match(str2).all()
#     for nb in node_b:
#         for na in node_a:
#             rel_ab = Relationship(na, rel, nb)
#             graph.create(rel_ab)
# def vnFunc(str1, str2, rel):
#     # 变量名 => name
#     node_a = graph.nodes.match(str1).all()
#     node_b = graph.nodes.match(name=str2).first()
#     for na in node_a:
#         rel_ab = Relationship(na, rel, node_b)
#         graph.create(rel_ab)

# def fnFunc(str1, str2, rel):
#     # father => [inn] => name
#     node_a = graph.nodes.match(father=str1).all()
#     node_b = graph.nodes.match(name=str2).all()
#     for nb in node_b:
#         for na in node_a:
#             rel_ab = Relationship(na, rel, nb)
#             graph.create(rel_ab)

# point_name = []
# for i in range(0, len(point_data)):
#     point_name.append(str(point_data['节点'][i]))
# var_name = []
# for i in range(0, len(point_data)):
#    var_name.append(str(point_data['变量名'][i]))
# point_level = []
# for i in range(0, len(point_data)):
#    point_level.append(str(point_data['level'][i]))
# point_father = []
# for i in range(0, len(point_data)):
#    point_father.append(str(point_data['father'][i]))
# valueType = []
# for i in range(0, len(point_data)):
#    valueType.append(str(point_data['valueType'][i]))

# # 画点
# plot_point(point_name, var_name, point_level, point_father, valueType)
# # 连线
# link_point(point_name, point_level)
# vtFunc("智能决策", "人工智能", "技术构成")
# vtFunc("学习算法", "人工智能", "技术构成")
# vtFunc("信息化", "人工智能", "技术构成")
# vvFunc("使用参数", "信息化", "信息化内容")
# vnFunc("智能决策", "自动化技术", "学科领域内容")
# vnFunc("学习算法", "自动化技术", "学科领域内容")
# vnFunc("信息化", "自动化技术", "学科领域内容")
# vnFunc("研究人员", "智能围岩分级", "领域内研究人员")

# a_have=graph.nodes.match("分级方法",name="RMR").first()
# b_have=graph.nodes.match("使用参数", father="RMR").all()
# for node_b in b_have:
#     rel_a = Relationship(node_b, "评估参数", a_have)
#     graph.create(rel_a)

# print(point_name)
# print(var_name)