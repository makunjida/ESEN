import typing
import csv
#加载数据
# import keras
import pandas as pd
import matchzoo
import os
from Models.FCWithEvidences.DeClare import pack
import torch
from typing import List, Dict
from tqdm import tqdm
import numpy as np



#WikiQA 是一个用于问答系统的数据集，由微软研究院创建。该数据集包含了来自维基百科的问题和答案，以及人工创建的一些问题和答案。它被用于评估自然语言处理和问答系统的性能，特别是在回答事实性问题方面。
_url = "https://download.microsoft.com/download/E/5/F/" \
       "E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip"


def load_data(
    data_root: str,
    stage: str = 'train',
    task: str = 'classification',
    filtered: bool = False,
    return_classes: bool = False,
    kfolds: int = 5,
    extend_claim: bool = False
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load WikiQA data.
    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param filtered: Whether remove the questions without correct answers.
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.
    :param kfolds: `int` the number of folds
    :param extend_claim: `bool` `True` to merge claim id and claim text as a way to extend text of claims是否将索赔 ID 和索赔文本合并，作为扩展索赔文本的一种方式。
    :return: A DataPack unless `task` is `classificiation` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    if stage not in ['dev'] + ["train_%s" % i for i in range(kfolds)] + ["test_%s" % i for i in range(kfolds)]:
        raise ValueError("%s is not a valid stage. Must be one of `train`, `dev`, and `test`." % stage)

    # data_root = _download_data()
    data_root = data_root
    file_path = os.path.join(data_root, '%s.tsv' % (stage))
    data_pack = _read_data(file_path, extend_claim)

    if task == 'ranking':#排序
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        # data_pack.one_hot_encode_label(task.num_classes, inplace=True)
        # I removed this due to PyTorch https://github.com/pytorch/pytorch/issues/5554
        if return_classes:
            return data_pack, [False, True]
        else:
            return data_pack
    else:
        raise ValueError("{%s} is not a valid task." % task + " " 
                         "Must be one of `Ranking` and `Classification`.")

#函数用于实际加载数据。它从指定的文件路径 path 中读取数据，这个文件是一个以制表符分隔的 TSV 文件。
# 然后，它创建一个新的 DataFrame 包含所需的列，这些列包括左侧文本、原始左侧文本、索赔 ID、索赔来源、证据、证据来源、字符索赔来源、原始索赔来源、扩展文本等。
def _read_data(path, extend_claim: bool = False):
    table = pd.read_csv(path, sep='\t', header=0, quoting=csv.QUOTE_NONE)#从指定路径读取数据表格，使用制表符作为分隔符，指定表头为第一行，引用方式为无引号
    # 将字符串标签转换为数字标签的函数
    def str2num(lab):
        # convert bool label to int# 将布尔型标签转换为整数
        lab = str(lab) if type(lab) == bool else lab
        assert lab == "TRUE" or lab == "True" or lab == "False" or lab == "FALSE" or lab == "MIXED"
        if lab == "TRUE" or lab == "True":
            return 1
        elif lab == "FALSE" or lab == "False":
            return 0
        else:
            return 2
    # 合并文本的函数
    def merge_text(a, b):
        a = a.replace(".json", " ")
        a = " ".join(a.split("_"))
        a = " ".join(a.split("-"))
        return a + " " + b
    # cred_label	claim_id	claim_text	claim_source	evidence	evidence_source
    # 创建一个新的DataFrame，包含所需的列
    df = pd.DataFrame({
        'text_left': table['claim_text'],#将数据表格中的 'claim_text' 列的值赋给了新创建的 DataFrame 中的 'text_left' 列
        'raw_text_left': table['claim_text'].copy(),# 将 'claim_text' 列的值复制给 'raw_text_left' 列
        'claim_id': table["claim_id"],
        "claim_source": table["claim_source"],
        "char_claim_source": table["claim_source"].copy(),#声明来源的字符表示
        "raw_claim_source": table["claim_source"].copy(),
        "extended_text": table.progress_apply(lambda x: merge_text(x.claim_id, x.claim_text), axis = 1),

        'text_right': table['evidence'],
        'raw_text_right': table['evidence'].copy(),
        "evidence_source": table["evidence_source"],
        "char_evidence_source": table["evidence_source"].copy(),
        "raw_evidence_source": table["evidence_source"].copy(),

        'id_left': table['id_left'],
        'id_right': table['id_right'],
        #新增的head列
        'head_left': table['head_left'],
        'head_right': table['head_right'],
        
        'label': table['cred_label'].progress_apply(str2num)
    
        


    })## 如果需要扩展索赔文本，将左侧文本替换为扩展后的文本
    if extend_claim:
        df["text_left"] = df["extended_text"]
        df["raw_text_left"] = df["extended_text"]
    # I decided to create a new datapack to avoid touching old code of learning to rank我决定创建一个新的数据包，以避免触及旧的学习排序代码
    return pack.pack(df, selected_columns_left = ['text_left', 'id_left',  'head_left', 'raw_text_left', 'claim_source', 'raw_claim_source', 'char_claim_source'],
                     selected_columns_right = ['text_right', 'id_right', 'head_right', 'raw_text_right', 'evidence_source', 'raw_evidence_source', 'char_evidence_source'])

