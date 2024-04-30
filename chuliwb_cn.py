from stanfordcorenlp import StanfordCoreNLP

#如果要用其他语言，需要单独设置
nlp_ch= StanfordCoreNLP(r'E:\python_work\pyneo\pdfToData\myPDF\models\stanford-corenlp-full-2018-02-27', lang='zh')


sen='我是一名交大的学生'

print(nlp_ch.pos_tag(sen))
print(nlp_ch.parse(sen))

nlp_ch.close()
