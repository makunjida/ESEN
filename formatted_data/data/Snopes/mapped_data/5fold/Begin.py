# ####该文件的目的：将原始origin的内容转换为该目录下的.json文件
#
#
# import json
# import re
# from stanfordcorenlp import StanfordCoreNLP
# nlp = StanfordCoreNLP(r'F:/AGGCN/Our-AGGCN_5.5/stanford-corenlp-full-2016-10-31', lang='en')
# from tqdm import tqdm
# from eda import eda
#
# # sen = "He was elected president in 2008."
# # print(nlp.parse(sen))#构建了语法树
# # print("**********")
# # print(nlp.dependency_parse(sen))
# # tokens = nlp.word_tokenize(sen)
# # res = nlp.dependency_parse(sen)
# #
# # print(res)
# #
# # head = [-1 for x in range(len(tokens))]
# #
# # print(head)
# #
# # for tuple in res:
# #     id = int(tuple[1])
# #     value = int(tuple[2])
# #     head[value-1] = id
# #
# # print(head)
#
# def clean_sen(string):#1.清理句子，删除无用的字符，保留暂时有用的字符
#     string = re.sub(r"[^A-Za-z0-9\'.,:\-]", " ", string)  # 余留小数点.的意思是为了保存出现的小数；保留逗号，的意思是保存300，000这样的大树；保留引号‘的意思是保留缩写
#     string = re.sub(r"\s{2,}", " ", string)  # 将多余的空格合并为一个
#     return string.strip()# return string.strip().lower()#全部转为小写
#
# def change_tag(tag):#二分类
#     if tag == 'pants-fire':
#         return "False"
#     elif tag == 'false':
#         return "False"
#     elif tag == 'barely-true':
#         return "True"
#     elif tag == 'half-true':
#         return "True"
#     elif tag == 'mostly-true':
#         return "True"
#     elif tag == 'true':
#         return "True"
#
# def process(content,nlp):
#
#     content = clean_sen(content)#清除无效字符
#
#     tokens = nlp.word_tokenize(content)#分词
#
#     res = nlp.dependency_parse(content)
#
#     head = [-1 for x in range(len(tokens))]
#
#     for tuple in res:
#         id = int(tuple[1])
#         value = int(tuple[2])
#         head[value-1] = id
#
#     return tokens, head
#
# def process_liar(nlp):
#
#     sum=0
#
#     f_r1 = open('F:/AGGCN/Our-AGGCN_5.5/dataset/Liar/origin/train.tsv', 'r', encoding='UTF-8')
#     f_r2 = open('F:/AGGCN/Our-AGGCN_5.5/dataset/Liar/origin/valid.tsv', 'r', encoding='UTF-8')
#     f_r3 = open('F:/AGGCN/Our-AGGCN_5.5/dataset/Liar/origin/test.tsv', 'r', encoding='UTF-8')
#
#     train_text = []
#     print("Train:")
#     for line in tqdm(f_r1.readlines()):
#
#         lin = line.strip()
#         if not lin:
#             continue
#         temp = lin.split('\t')
#
#         tag = change_tag(temp[1])
#         content = temp[2]
#
#         if tag == "False":
#             sentences = eda(content)
#             for content in sentences:
#                 sum += 1
#                 tokens, head = process(content, nlp)
#                 dict = {"id": str(sum), "relation": tag, "token": tokens, "stanford_head": head}
#                 train_text.append(dict)
#         else:
#             sum+=1
#
#             tokens, head = process(content, nlp)
#
#             dict = {"id": str(sum), "relation": tag, "token": tokens, "stanford_head": head}
#
#             train_text.append(dict)
#
#
#     with open('F:/AGGCN/Our-AGGCN_5.5/dataset/Liar/data/train.json', 'w', encoding='UTF-8') as f:
#         json.dump(train_text,f)
#
#     print(len(train_text))
#     print(sum)
#
#     dev_text = []
#     print("Dev")
#     for line in tqdm(f_r2.readlines()):
#
#
#         lin = line.strip()
#         if not lin:
#             continue
#         temp = lin.split('\t')
#
#         tag = change_tag(temp[1])
#         content = temp[2]
#
#         if tag == "False":
#             sentences = eda(content)
#             for content in sentences:
#                 sum += 1
#                 tokens, head = process(content, nlp)
#                 dict = {"id": str(sum), "relation": tag, "token": tokens, "stanford_head": head}
#                 dev_text.append(dict)
#         else:
#             sum += 1
#
#             tokens, head = process(content, nlp)
#
#             dict = {"id": str(sum), "relation": tag, "token": tokens, "stanford_head": head}
#
#             dev_text.append(dict)
#
#     print(len(dev_text))
#     with open('F:/AGGCN/Our-AGGCN_5.5/dataset/Liar/data/dev.json', 'w', encoding='UTF-8') as f:
#         json.dump(dev_text, f)
#     print(sum)
#
#     test_text = []
#     print("Test")
#     for line in tqdm(f_r3.readlines()):
#
#         lin = line.strip()
#         if not lin:
#             continue
#         temp = lin.split('\t')
#
#         tag = change_tag(temp[1])
#         content = temp[2]
#
#         if tag == "False":
#             sentences = eda(content)
#             for content in sentences:
#                 sum += 1
#                 tokens, head = process(content, nlp)
#                 dict = {"id": str(sum), "relation": tag, "token": tokens, "stanford_head": head}
#                 test_text.append(dict)
#         else:
#             sum += 1
#
#             tokens, head = process(content, nlp)
#
#             dict = {"id": str(sum), "relation": tag, "token": tokens, "stanford_head": head}
#
#             test_text.append(dict)
#
#     print(len(test_text))
#     with open('F:/AGGCN/Our-AGGCN_5.5/dataset/Liar/data/test.json', 'w', encoding='UTF-8') as f:
#         json.dump(test_text, f)
#
#     print(sum)
#
# if __name__ == '__main__':
#     process_liar(nlp)


####该文件的目的：将原始origin的内容转换为该目录下的.json文件


import json
import re
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'stanford-corenlp-full-2016-10-31', lang='en')
from tqdm import tqdm
#from eda import eda

# sen = "He was elected president in 2008."
# print(nlp.parse(sen))#构建了语法树
# print("**********")
# print(nlp.dependency_parse(sen))
# tokens = nlp.word_tokenize(sen)
# res = nlp.dependency_parse(sen)
#
# print(res)
#
# head = [-1 for x in range(len(tokens))]
#
# print(head)
#
# for tuple in res:
#     id = int(tuple[1])
#     value = int(tuple[2])
#     head[value-1] = id
#
# print(head)


import pandas as pd

# 读取原始数据

"""#增加一列
file_path = 'formatted_data/declare/Snopes/mapped_data/5fold/train_0.tsv'
df = pd.read_csv(file_path, sep='\t')
# 新建一列 "new_column"，可以根据需要替换为实际的列名
df['head_left'] = '1'  # 将 'your_values_here' 替换为实际的数值或计算逻辑
df['head_right'] = '2'
# 将修改后的数据保存回原始文件
df.to_csv(file_path, sep='\t', index=False)

file_path = 'formatted_data/declare/Snopes/mapped_data/5fold/train_1.tsv'
df = pd.read_csv(file_path, sep='\t')
# 新建一列 "new_column"，可以根据需要替换为实际的列名
df['head_left'] = '1'  # 将 'your_values_here' 替换为实际的数值或计算逻辑
df['head_right'] = '2'
# 将修改后的数据保存回原始文件
df.to_csv(file_path, sep='\t', index=False)


file_path = 'formatted_data/declare/Snopes/mapped_data/5fold/train_2.tsv'
df = pd.read_csv(file_path, sep='\t')
# 新建一列 "new_column"，可以根据需要替换为实际的列名
df['head_left'] = '1'  # 将 'your_values_here' 替换为实际的数值或计算逻辑
df['head_right'] = '2'
# 将修改后的数据保存回原始文件
df.to_csv(file_path, sep='\t', index=False)



file_path = 'formatted_data/declare/Snopes/mapped_data/5fold/train_3.tsv'
df = pd.read_csv(file_path, sep='\t')
# 新建一列 "new_column"，可以根据需要替换为实际的列名
df['head_left'] = '1'  # 将 'your_values_here' 替换为实际的数值或计算逻辑
df['head_right'] = '2'
# 将修改后的数据保存回原始文件
df.to_csv(file_path, sep='\t', index=False)


file_path = 'formatted_data/declare/Snopes/mapped_data/5fold/train_4.tsv'
df = pd.read_csv(file_path, sep='\t')
# 新建一列 "new_column"，可以根据需要替换为实际的列名
df['head_left'] = '1'  # 将 'your_values_here' 替换为实际的数值或计算逻辑
df['head_right'] = '2'
# 将修改后的数据保存回原始文件
df.to_csv(file_path, sep='\t', index=False)


"""


def process(content,nlp):

    
    tokens = nlp.word_tokenize(content)#分词   

    res = nlp.dependency_parse(content)

    head = [-1 for x in range(len(tokens))]

    for tuple in res:
        id = int(tuple[1])
        value = int(tuple[2])
        head[value-1] = id

    return  head


def process_liar(nlp):
#####0
    df = pd.read_csv('formatted_data/declare/Snopes/mapped_data/dev.tsv', sep='\t', encoding='UTF-8')
    print("新闻")
    head_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        
        content = row['claim_text']
        head = process(content, nlp)
        head_list.append(head)
    
    df['head_left'] = head_list    #claim_text   evidence
    df.to_csv('formatted_data/declare/Snopes/mapped_data/dev.tsv', sep='\t', index=False, encoding='UTF-8')

    df = pd.read_csv('formatted_data/declare/Snopes/mapped_data/dev.tsv', sep='\t', encoding='UTF-8')
    print("证据")
    head_list1 = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        content = row['evidence']
        #content = row['claim_text']
        head = process(content, nlp)
        head_list1.append(head)
    df['head_right'] = head_list1
    #df['head_left'] = head_list    #claim_text   evidence
    df.to_csv('formatted_data/declare/Snopes/mapped_data/dev.tsv', sep='\t', index=False, encoding='UTF-8')

"""
#####1
    print("1")
    df = pd.read_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_1.tsv', sep='\t', encoding='UTF-8')
    print("新闻")
    head_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        
        content = row['claim_text']
        head = process(content, nlp)
        head_list.append(head)
    
    df['head_left'] = head_list    #claim_text   evidence
    df.to_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_1.tsv', sep='\t', index=False, encoding='UTF-8')

    df = pd.read_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_1.tsv', sep='\t', encoding='UTF-8')
    print("证据")
    head_list1 = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        content = row['evidence']
        #content = row['claim_text']
        head = process(content, nlp)
        head_list1.append(head)
    df['head_right'] = head_list1
    #df['head_left'] = head_list    #claim_text   evidence
    df.to_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_1.tsv', sep='\t', index=False, encoding='UTF-8')


######2
    print("2")
    df = pd.read_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_2.tsv', sep='\t', encoding='UTF-8')
    print("新闻")
    head_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        
        content = row['claim_text']
        head = process(content, nlp)
        head_list.append(head)
    
    df['head_left'] = head_list    #claim_text   evidence
    df.to_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_2.tsv', sep='\t', index=False, encoding='UTF-8')

    df = pd.read_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_2.tsv', sep='\t', encoding='UTF-8')
    print("证据")
    head_list1 = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        content = row['evidence']
        #content = row['claim_text']
        head = process(content, nlp)
        head_list1.append(head)
    df['head_right'] = head_list1
    #df['head_left'] = head_list    #claim_text   evidence
    df.to_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_2.tsv', sep='\t', index=False, encoding='UTF-8')


######3
    print("3")
    df = pd.read_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_3.tsv', sep='\t', encoding='UTF-8')
    print("新闻")
    head_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        
        content = row['claim_text']
        head = process(content, nlp)
        head_list.append(head)
    
    df['head_left'] = head_list    #claim_text   evidence
    df.to_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_3.tsv', sep='\t', index=False, encoding='UTF-8')

    df = pd.read_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_3.tsv', sep='\t', encoding='UTF-8')
    print("证据")
    head_list1 = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        content = row['evidence']
        #content = row['claim_text']
        head = process(content, nlp)
        head_list1.append(head)
    df['head_right'] = head_list1
    #df['head_left'] = head_list    #claim_text   evidence
    df.to_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_3.tsv', sep='\t', index=False, encoding='UTF-8')


#####4
    print("4")
    df = pd.read_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_4.tsv', sep='\t', encoding='UTF-8')
    print("新闻")
    head_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        
        content = row['claim_text']
        head = process(content, nlp)
        head_list.append(head)
    
    df['head_left'] = head_list    #claim_text   evidence
    df.to_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_4.tsv', sep='\t', index=False, encoding='UTF-8')

    df = pd.read_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_4.tsv', sep='\t', encoding='UTF-8')
    print("证据")
    head_list1 = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        content = row['evidence']
        #content = row['claim_text']
        head = process(content, nlp)
        head_list1.append(head)
    df['head_right'] = head_list1
    #df['head_left'] = head_list    #claim_text   evidence
    df.to_csv('formatted_data/declare/Snopes/mapped_data/5fold/train_4.tsv', sep='\t', index=False, encoding='UTF-8')

"""
if __name__ == '__main__':
    process_liar(nlp)



###以下用不了
def process_liar1(nlp):

    

    f_r1 = open('formatted_data/declare/Snopes/mapped_data/5fold/c.tsv', 'r', encoding='UTF-8')
    
    print("Train:")
    head_list = []
   
    lines = []
    for line in tqdm(f_r1.readlines()):

        lin = line.strip()
        if not lin:
            continue
        temp = lin.split('\t')

        
        content = temp[3]
        print("content",content)
      
        head = process(content, nlp)
        print(head)
        temp[8] = head
        print("temp[8]",temp[8])
        temp[8] = str(head)
        head_list.append(temp[8])

        # 将修改后的行保存到 lines 列表
        lines.append('\t'.join(temp))
    with open('formatted_data/declare/Snopes/mapped_data/5fold/c.tsv', 'w', encoding='UTF-8') as f_w:
        f_w.writelines(lines)