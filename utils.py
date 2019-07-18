# -*- coding:utf-8 -*-
# 工具函数

import pickle
from bert import tokenization
import numpy as np

# 向量长度
_maxlen = 50
# 文本长度
maxlen = _maxlen - 2
# 标签长度
maxlabel = 6  # (0,1,2,3,4,5)


# 读取pkl文件
def readPkl(filename):
    return pickle.load(open(filename, 'rb'), encoding='utf-8')


# 截取单条文本
def cut2Maxlen(line):
    line = line.strip()
    if len(line) < maxlen:
        return line
    else:
        return line[:maxlen]


# 截取一个bach的文本长度
def bCut2Maxlen(lines):
    res = []
    for line in lines:
        res.append(cut2Maxlen(line))
    return res


# label的onehot编码
def onehotCoding(num):
    label = [0] * maxlabel
    label[num] = 1
    return label


# 一个bach的label的onehot编码
def bOnehotCoding(nums):
    res = []
    for num in nums:
        res.append(onehotCoding(num))
    return res


# 单条文本token化
def tokenizeString(line, tokenizer):
    line = tokenization.convert_to_unicode(line)
    token = tokenizer.tokenize(line)
    return token


# 单条文本的bert输入构造
def buildBertInput(line, tokenizer):
    line = tokenizeString(line, tokenizer)
    token = ["[CLS]"]
    for t in line:
        token.append(t)
    token.append("[SEP]")

    word_ids = tokenizer.convert_tokens_to_ids(token)
    word_mask = [1] * len(word_ids)

    dlen = _maxlen - len(word_ids)
    word_ids = word_ids + [0] * dlen
    word_mask = word_mask + [0] * dlen

    word_segment_ids = [0] * len(word_ids)
    return word_ids, word_mask, word_segment_ids


# 一个bach的文本的bert输入构造
def bBuildBertInput(lines, tokenizer):
    word_ids = []
    word_mask = []
    word_segment_ids = []
    for line in lines:
        tid, tmask, tsegid = buildBertInput(line, tokenizer)
        word_ids.append(tid)
        word_mask.append(tmask)
        word_segment_ids.append(tsegid)
    return np.array(word_ids), np.array(word_mask), np.array(word_segment_ids)
