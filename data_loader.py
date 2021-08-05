import copy
import csv
import json
import logging
import os

import random

import torch
from torch.utils.data import TensorDataset

from utils import get_label

import pandas as pd

from transformers import BertTokenizer
import stanza

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, dep_mask1):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        # self.e1_mask = e1_mask
        # self.e2_mask = e2_mask
        # self.dep_mask0 = dep_mask0
        self.dep_mask1 = dep_mask1
        # self.dep_mask2 = dep_mask2
        # self.dep_mask3 = dep_mask3
        # self.pos1 = pos1
        # self.pos2 = pos2

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SemEvalProcessor(object):
    """Processor for the Semeval data set """

    def __init__(self, args):
        self.args = args
        self.relation_labels = get_label(args)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    # 
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = self.relation_labels.index(line[0])
            if i % 1000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None

        if mode == "train":
            if self.args.no_blind:
                file_to_read = 'train_dep_no_blind.tsv'
            else:
                file_to_read = self.args.train_file

        elif mode == "dev":
            if self.args.no_blind:
                file_to_read = 'dev_dep_no_blind.tsv'
            else:
                file_to_read = self.args.dev_file
          
        elif mode == "test":
            if self.args.no_blind:
                file_to_read = 'test_dep_no_blind.tsv'
            else:
                file_to_read = self.args.test_file
        
        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_tsv(os.path.join(self.args.data_dir, file_to_read)), mode)


processors = {"ddi": SemEvalProcessor}

def get_depandency_mask(sen, nlp, tokenizer, no_entity_mark):
    '''
    获得每个句子的depandency_mask序列【经过BERT分词后】
    
    Parameters:
    
    sen: str, 一个句子
    nlp: obj, 一个自然语言处理工具
    tokenizer: obj, BERT分词器
    
    Return: 
    
    depan_mask_0，最短路径的mask向量
    depan_mask_1, 到最短路径距离为1的关键路径的mask向量
    
    words_pieces，整个句子用bert，token后的结果，为了统一stanza和bert的tokenizer，例如cannot，stanza会分成can not， 而bert不会。
    
    '''
#     delete [SEP]
    # sep = sen[sen.index('[SEP]'):]
    
    sen = sen[:sen.index('[SEP]')]
    
    sen = sen.replace('<e1>', '')
    sen = sen.replace('</e1>', '')
    sen = sen.replace('<e2>', '')
    sen = sen.replace('</e2>', '')
    doc = nlp(sen)
    
    # 先求每个词的head节点
    words = []
    head = []

    subj_pos = -1
    obj_pos = -1
    
    befor_sentences_words = 0

    for sentence in doc.sentences:
    #     对不同句子，stanza分别作分析，因此后面的句子中的head应该偏移前面句子含有词的个数
        befor_sentences_words = len(words)

        for index, word in enumerate(sentence.words):
#             print(word, index)
            if word.text == 'DRUG1':
                subj_pos = index + befor_sentences_words
            elif word.text == 'DRUG2':
                obj_pos = index + befor_sentences_words

    #         root 为0， 和其他的句子不同
            if word.head == 0:
                 head += [0]
            else:
                head += [word.head + befor_sentences_words]
            words += [word.text]
    
#     根据head，求得每个词到依赖路径的距离
    subj_pos = [subj_pos]
    obj_pos = [obj_pos]

    len_ = len(words)
    
    cas = None # 两个实体的共同祖先

    subj_ancestors = set(subj_pos)
    for s in subj_pos: 
        h = head[s] # durg1的依赖词索引
        tmp = [s]
        while h > 0: # 一直添加DRUG1的祖先节点的索引到subj-ancestors中
            tmp += [h-1] #16
            subj_ancestors.add(h-1)
            h = head[h-1]

        if cas is None: # 暂且未知
            cas = set(tmp)
        else:
            cas.intersection_update(tmp)

    obj_ancestors = set(obj_pos)
    for o in obj_pos:
        h = head[o]
        tmp = [o]
        while h > 0:
            tmp += [h-1]
            obj_ancestors.add(h-1)
            h = head[h-1]
        cas.intersection_update(tmp)
    
#     对于没有公共祖先节点的两个实体，我们认为它是负例，直接过滤
    if len(cas) == 0:
        return [-1]
    
    # find lowest common ancestor
    # lca 是两个实体的最小公共祖先
    if len(cas) == 1:
        lca = list(cas)[0]
    else:
        child_count = {k:0 for k in cas}
        for ca in cas:
            if head[ca] > 0 and head[ca] - 1 in cas:
                child_count[head[ca] - 1] += 1

        # the LCA has no child in the CA set
        for ca in cas:
            if child_count[ca] == 0:
                lca = ca
                break

    path_nodes = subj_ancestors.union(obj_ancestors).difference(cas) # 一个实体到另一个实体的路径
    path_nodes.add(lca)

    # compute distance to path_nodes
    dist = [-1 if i not in path_nodes else 0 for i in range(len_)] # -1不是最短路径上的节点， 0为最短路径上的节点
    
    for i in range(len_):
        if dist[i] < 0:
            stack = [i]
            while stack[-1] >= 0 and stack[-1] not in path_nodes:
                stack.append(head[stack[-1]] - 1)

            if stack[-1] in path_nodes:
                for d, j in enumerate(reversed(stack)):
                    dist[j] = d
            else:
                for j in stack:
                    if j >= 0 and dist[j] < 0:
                        dist[j] = int(1e4) # aka infinity
    
#     求依赖序列标注向量
    depan_mask_1 = [] # 1 is start 2 is other 0 is not in depan path 
    depan_mask_0 = [] # shorest path
    depan_mask_2 = []
    depan_mask_3 = []
    # words_pieces = [] # words after bert 

    # 保存路径1的词索引
    for index, item in enumerate(dist):

        word_piece = tokenizer.tokenize(words[index])
        # if words[index] == 'DRUG1':
        #     words_pieces += ['$']+word_piece+['$']
        # elif words[index] == 'DRUG2':
        #     words_pieces += ['#']+word_piece+['#']
        # else:
        #     words_pieces += word_piece

        len_word_piece = len(word_piece)

        if item <= 1:
    #         print(words[index]," distant to shortest dependancy path is ",item)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
                depan_mask_1 += [0]
            depan_mask_1 += [1]
            depan_mask_1 += [2]*(len_word_piece-1)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
                depan_mask_1 += [0]
        else:
            depan_mask_1 += [0]*(len_word_piece)
        
        if item <= 0:
    #         print(words[index]," distant to shortest dependancy path is ",item)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
                depan_mask_0 += [0]
            depan_mask_0 += [1]
            depan_mask_0 += [2]*(len_word_piece-1)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
                depan_mask_0 += [0]
        else:
            depan_mask_0 += [0]*(len_word_piece)

        if item <= 2:
    #         print(words[index]," distant to shortest dependancy path is ",item)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
                depan_mask_2 += [0]
            depan_mask_2 += [1]
            depan_mask_2 += [2]*(len_word_piece-1)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
                depan_mask_2 += [0]
        else:
            depan_mask_2 += [0]*(len_word_piece)

        if item <= 3:
    #         print(words[index]," distant to shortest dependancy path is ",item)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
                depan_mask_3 += [0]
            depan_mask_3 += [1]
            depan_mask_3 += [2]*(len_word_piece-1)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
                depan_mask_3 += [0]
        else:
            depan_mask_3 += [0]*(len_word_piece)
            
    # 统一stanza和bert的tokenizer
    # sep = tokenizer.tokenize(sep)
    
    return depan_mask_0, depan_mask_1, depan_mask_2, depan_mask_3

def get_depandency_mask_for_unblindedData(sen, nlp, tokenizer, no_entity_mark):
    '''
    获得每个未药物致盲的句子的depandency_mask序列【经过BERT分词后】
    
    Parameters:
    
    sen: str, 一个句子
    nlp: obj, 一个自然语言处理工具
    tokenizer: obj, BERT分词器
    no_entity_mark：bool, 是否有实体标记
    
    Return: 
    
    depan_mask_0，最短路径的mask向量
    depan_mask_1, 到最短路径距离为1的关键路径的mask向量

    words_pieces，整个句子用bert，token后的结果，为了统一stanza和bert的tokenizer，例如cannot，stanza会分成can not， 而bert不会。
    '''
#     get drug name
#   
    sen_list = sen.split()

    drug1_name = "".join(w+' ' for w in sen_list[sen_list.index('<e1>')+1:sen_list.index('</e1>')])
    drug2_name = "".join(w+' ' for w in sen_list[sen_list.index('<e2>')+1:sen_list.index('</e2>')])
    
    sen_list = sen_list[:sen_list.index('<e1>')] + ['DRUG1']+sen_list[sen_list.index('</e1>')+1:]
    sen_list = sen_list[:sen_list.index('<e2>')] + ['DRUG2']+sen_list[sen_list.index('</e2>')+1:]
    
    sen = "".join(w+' ' for w in sen_list)  
    
    doc = nlp(sen)
    
    # 先求每个词的head节点
    words = []
    head = []

    subj_pos = -1
    obj_pos = -1
    
    befor_sentences_words = 0

    for sentence in doc.sentences:
    #     对不同句子，stanza分别作分析，因此后面的句子中的head应该偏移前面句子含有词的个数
        befor_sentences_words = len(words)

        for index, word in enumerate(sentence.words):
#             print(word, index)
            if word.text == 'DRUG1':
                subj_pos = index + befor_sentences_words
            elif word.text == 'DRUG2':
                obj_pos = index + befor_sentences_words

    #         root 为0， 和其他的句子不同
            if word.head == 0:
                 head += [0]
            else:
                head += [word.head + befor_sentences_words]
            words += [word.text]
    
#     根据head，求得每个词到依赖路径的距离
    subj_pos = [subj_pos]
    obj_pos = [obj_pos]

    len_ = len(words)
    
    cas = None # 两个实体的共同祖先

    subj_ancestors = set(subj_pos)
    for s in subj_pos: 
        h = head[s] # durg1的依赖词索引
        tmp = [s]
        while h > 0: # 一直添加DRUG1的祖先节点的索引到subj-ancestors中
            tmp += [h-1] #16
            subj_ancestors.add(h-1)
            h = head[h-1]

        if cas is None: # 暂且未知
            cas = set(tmp)
        else:
            cas.intersection_update(tmp)

    obj_ancestors = set(obj_pos)
    for o in obj_pos:
        h = head[o]
        tmp = [o]
        while h > 0:
            tmp += [h-1]
            obj_ancestors.add(h-1)
            h = head[h-1]
        cas.intersection_update(tmp)
    
#     对于没有公共祖先节点的两个实体，我们认为它是负例，直接过滤 ？？？处理这个
    if len(cas) == 0:
        return [-1]
    
    # find lowest common ancestor
    # lca 是两个实体的最小公共祖先
    if len(cas) == 1:
        lca = list(cas)[0]
    else:
        child_count = {k:0 for k in cas}
        for ca in cas:
            if head[ca] > 0 and head[ca] - 1 in cas:
                child_count[head[ca] - 1] += 1

        # the LCA has no child in the CA set
        for ca in cas:
            if child_count[ca] == 0:
                lca = ca
                break

    path_nodes = subj_ancestors.union(obj_ancestors).difference(cas) # 一个实体到另一个实体的路径
    path_nodes.add(lca)

    # compute distance to path_nodes
    dist = [-1 if i not in path_nodes else 0 for i in range(len_)] # -1不是最短路径上的节点， 0为最短路径上的节点

    for i in range(len_):
        if dist[i] < 0:
            stack = [i]
            while stack[-1] >= 0 and stack[-1] not in path_nodes:
                stack.append(head[stack[-1]] - 1)

            if stack[-1] in path_nodes:
                for d, j in enumerate(reversed(stack)):
                    dist[j] = d
            else:
                for j in stack:
                    if j >= 0 and dist[j] < 0:
                        dist[j] = int(1e4) # aka infinity
    
#     求依赖序列标注向量 ？？？注释掉
    depan_mask_1 = [] # 1 is start 2 is other 0 is not in depan path 
    # depan_mask_0 = [] # shorest path
    # depan_mask_2 = []
    # depan_mask_3 = []
    # words_pieces = [] # words after bert 
    
    # 保存路径1的词索引
    for index, item in enumerate(dist):

        word_piece = tokenizer.tokenize(words[index])

        # if words[index] == 'DRUG1':
        #     words_pieces += ['$']+tokenizer.tokenize(drug1_name)+['$']
        # elif words[index] == 'DRUG2':
        #     words_pieces += ['#']+tokenizer.tokenize(drug2_name)+['#']
        # else:
        #     words_pieces += word_piece

        len_word_piece = len(word_piece)

        if item <= 1:
    #         print(words[index]," distant to shortest dependancy path is ",item)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
                if not no_entity_mark:
                    depan_mask_1 += [0]
            depan_mask_1 += [1]
            depan_mask_1 += [2]*(len_word_piece-1)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
                if words[index] == 'DRUG1':
                    depan_mask_1 += [2]*(len(tokenizer.tokenize(drug1_name))-1)
                else:
                    depan_mask_1 += [2]*(len(tokenizer.tokenize(drug2_name))-1)
                if not no_entity_mark:
                    depan_mask_1 += [0]
        else:
            depan_mask_1 += [0]*(len_word_piece)
        
    #     if item <= 0:
    # #         print(words[index]," distant to shortest dependancy path is ",item)
    #         if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
    #             depan_mask_0 += [0]
    #         depan_mask_0 += [1]
    #         depan_mask_0 += [2]*(len_word_piece-1)
    #         if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
    #             if words[index] == 'DRUG1':
    #                 depan_mask_0 += [2]*(len(tokenizer.tokenize(drug1_name))-1)
    #             else:
    #                 depan_mask_0 += [2]*(len(tokenizer.tokenize(drug2_name))-1)
    #             depan_mask_0 += [0]
    #     else:
    #         depan_mask_0 += [0]*(len_word_piece)

    #     if item <= 2:
    # #         print(words[index]," distant to shortest dependancy path is ",item)
    #         if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
    #             depan_mask_2 += [0]
    #         depan_mask_2 += [1]
    #         depan_mask_2 += [2]*(len_word_piece-1)
    #         if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
    #             if words[index] == 'DRUG1':
    #                 depan_mask_2 += [2]*(len(tokenizer.tokenize(drug1_name))-1)
    #             else:
    #                 depan_mask_2 += [2]*(len(tokenizer.tokenize(drug2_name))-1)
    #             depan_mask_2 += [0]
    #     else:
    #         depan_mask_2 += [0]*(len_word_piece)

    #     if item <= 3:
    # #         print(words[index]," distant to shortest dependancy path is ",item)
    #         if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
    #             depan_mask_3 += [0]
    #         depan_mask_3 += [1]
    #         depan_mask_3 += [2]*(len_word_piece-1)
    #         if words[index] == 'DRUG1' or words[index] == 'DRUG2': # add #的占位符
    #             if words[index] == 'DRUG1':
    #                 depan_mask_3 += [2]*(len(tokenizer.tokenize(drug1_name))-1)
    #             else:
    #                 depan_mask_3 += [2]*(len(tokenizer.tokenize(drug2_name))-1)
    #             depan_mask_3 += [0]
    #     else:
    #         depan_mask_3 += [0]*(len_word_piece)

    return depan_mask_1

def getPositionVec(dis):
    if dis <= -31:
        return 0
    elif -30 <= dis <= -21:
        return 1
    elif -20 <= dis <= -11:
        return 2
    elif -10 <= dis <= -6:
        return 3
    elif dis == -5:
        return 4
    elif dis == -4:
        return 5
    elif dis == -3:
        return 6
    elif dis == -2:
        return 7
    elif dis == -1:
        return 8
    elif dis == 0:
        return 9
    elif dis == 1:
        return 10
    elif dis == 2:
        return 11
    elif dis == 3:
        return 12
    elif dis == 4:
        return 13
    elif dis == 5:
        return 14
    elif 6 <= dis <= 10:
        return 15
    elif 11 <= dis <= 20:
        return 16
    elif 21 <= dis <= 30:
        return 17
    elif 31 <= dis:
        return 18

def convert_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    add_sep_token=False,
    mask_padding_with_zero=True,
    use_full_sequence = False,
    nlp=None,
    no_blind=False,
    no_entity_mark = False
):
    features = []

    error_index = []
    error_text = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        if no_entity_mark:
            text_a = example.text_a
            text_a = text_a.replace('<e1>', '')
            text_a = text_a.replace('</e1>', '')
            text_a = text_a.replace('<e2>', '')
            text_a = text_a.replace('</e2>', '')
            tokens_a = tokenizer.tokenize(text_a) 
        else:
            tokens_a = tokenizer.tokenize(example.text_a) 

            e11_p = tokens_a.index("<e1>")  # the start position of entity1
            e12_p = tokens_a.index("</e1>")  # the end position of entity1
            e21_p = tokens_a.index("<e2>")  # the start position of entity2
            e22_p = tokens_a.index("</e2>")  # the end position of entity2

            # Replace the token
            tokens_a[e11_p] = "$"
            tokens_a[e12_p] = "$"
            tokens_a[e21_p] = "#"
            tokens_a[e22_p] = "#"

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1
        if len(tokens_a) > max_seq_len - special_tokens_count: 
            tokens_a = tokens_a[: (max_seq_len - special_tokens_count)]

        tokens = tokens_a
        if add_sep_token:
            tokens += [sep_token]

        token_type_ids = [sequence_a_segment_id] * len(tokens) 

        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids) 

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length) 
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # get the depandency mask vector
        dep_mask1 = get_depandency_mask_for_unblindedData(example.text_a, nlp, tokenizer, no_entity_mark)

        # filter the instance that do not have LCA 
        if dep_mask1[0] == -1:
            error_index.append(ex_index)
            error_text.append(example.text_a)
            continue

        dep_mask1_pad = [0] * len(attention_mask)

        for i,item in enumerate(dep_mask1):
            if item !=0:
                dep_mask1_pad[i+1] = 1
        
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )

        label_id = int(example.label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("dep_mask1: %s" % " ".join([str(x) for x in dep_mask1_pad]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id,
                dep_mask1=dep_mask1_pad
            )
        )

    # save error
    error_index = pd.DataFrame(data=error_index)
    error_text = pd.DataFrame(data=error_text)

    if no_blind:
        pd.concat([error_index,error_text], ignore_index=True, axis=1).to_csv('no_blind_error.tsv', sep='\t',header=None, index=False)
    else:
        pd.concat([error_index,error_text], ignore_index=True, axis=1).to_csv('blind_error.tsv', sep='\t',header=None, index=False)
    return features

def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # stanza.download('en', package='craft')
        nlp = stanza.Pipeline('en', package='craft',dir='./stzanza/stanza_resources/')

        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train") 
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")
        
        features = convert_examples_to_features(
            examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token, use_full_sequence=args.use_full_sequence, nlp=nlp, no_blind=args.no_blind, no_entity_mark=args.no_entity_mark
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_dep_mask1 = torch.tensor([f.dep_mask1 for f in features], dtype=torch.long)  # add dep mask1
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    # Change the label to false with a probability 
    if mode == "train" and args.label_noise_rate != -1:
        for j, item in enumerate(all_label_ids):
            if random.random() < args.label_noise_rate:
                label_true = item
                label_choice = [0,1,2,3,4]
                del(label_choice[label_true])
                label_flase = random.choice(label_choice)
                all_label_ids[j] = label_flase

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_label_ids,
        all_dep_mask1,
    )
    return dataset
