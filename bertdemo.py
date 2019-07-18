# -*- coding:utf-8 -*-

import tensorflow as tf
from bert import tokenization, modeling
import os
from utils import *
import numpy as np

# 配置文件
data_root = 'bertmodel/chinese/'
bert_config_file = data_root + 'bert_config.json'
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
init_checkpoint = data_root + 'bert_model.ckpt'
bert_vocab_file = data_root + 'vocab.txt'

datarootpath = 'data/'

# 输入占位符
input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')

# 初始化BERT
model = modeling.BertModel(
    config=bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False)

# 加载bert模型
tvars = tf.trainable_variables()
(assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
tf.train.init_from_checkpoint(init_checkpoint, assignment)

# 获取句向量和词向量。
encoder_sentence_layer = model.get_pooled_output()
encoder_tokens_layer = model.all_encoder_layers


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tokenizer = tokenization.FullTokenizer(vocab_file=bert_vocab_file)
    trainx = readPkl(datarootpath+"train_X.pkl")
    trainy = readPkl(datarootpath+"train_Y.pkl")
    
    # 取一个bach试试
    bach_size = 100
    btrainx = trainx[:bach_size]
    btrainy = trainy[:bach_size]

    # label进行onehot编码
    labels = bOnehotCoding(btrainy)

    # 把长的文本先切短
    btrainx = bCut2Maxlen(btrainx)

    # 构造bert输入
    word_ids, word_mask, word_segment_ids = bBuildBertInput(btrainx,tokenizer)
    print(len(word_ids[0]), len(word_mask[0]), len(word_segment_ids[0]))
    fd = {input_ids: word_ids, input_mask: word_mask, segment_ids: word_segment_ids}

    tokens = sess.run([encoder_sentence_layer], feed_dict=fd)


    tokens = np.array(tokens)   # token 原本是list，这里改成np.array是为了看它的shape
    print("tokens'shape",tokens.shape)
    print("总句子数：",len(tokens[0]))
    print("第0个句子的表示长度和标签：",len(tokens[0][0]), labels[0])
