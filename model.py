import torch
import torch.nn as nn

import torch.nn.utils.rnn as rnn_utils

from transformers import BertModel, BertPreTrainedModel

from GHMC_Loss import GHMC_Loss
from MultiFocalLoss import MultiFocalLoss

import numpy as np

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(RBERT, self).__init__(config)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.num_labels = config.num_labels

        # self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        # self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        # dep_mask_fc_layer
        # self.dep_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)

        self.args = args
        if self.args.Loss == 'GHM':
            self.Loss = GHMC_Loss(bins=self.args.bins, alpha=self.args.alpha)
            print("use GHM Loss")
        elif self.args.Loss == 'Focal':
            # if mean loss is so small, so sum
            self.Loss = MultiFocalLoss(5,[0.15, 0.08, 0.67, 0.10, 0.01])
            print("use Focal Loss")
        else:
            self.Loss = nn.CrossEntropyLoss()

        self.loss_factor = self.args.loss_factor

        # bilstm for dep0
        # self.rnn0 = nn.LSTM(input_size=config.hidden_size, hidden_size=args.lstm_hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        # bilstm for dep1
        # pos
        self.rnn1 = nn.LSTM(input_size=config.hidden_size, hidden_size=args.lstm_hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        # 使用orthogonal初始化LSTM的参数
        # nn.init.orthogonal(self.rnn.weight_hh_l0)
        # nn.init.orthogonal(self.rnn.weight_hh_l0_reverse)
        # nn.init.orthogonal(self.rnn.weight_ih_l0)
        # nn.init.orthogonal(self.rnn.weight_ih_l0_reverse)

        # MLP for dep0
        # self.MLP0 = FCLayer(
        #     args.lstm_hidden_size*2,
        #     args.MLP_hidden_size,
        #     args.dropout_rate,
        #     use_activation=True,
        # )
        # pos
        # 董珂学姐的相对位置嵌入
        # self.pos_embs = nn.Embedding(19, 15)

        # MLP for dep1
        self.MLP1 = FCLayer(
            args.lstm_hidden_size*2,
            args.MLP_hidden_size,
            args.dropout_rate,
            use_activation=True,
        )

        # self.entity_fc_layer = FCLayer(config.hidden_size, args.MLP_hidden_size, args.dropout_rate)

        self.label_classifier = FCLayer( 
            args.MLP_hidden_size,
            config.num_labels,
            args.dropout_rate,
            use_activation=False,
        )

        # self.label_binary_classifier = FCLayer( 
        #     args.MLP_hidden_size,
        #     2,
        #     args.dropout_rate,
        #     use_activation=False,
        # )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim] //这里意思是对取到的每个句子中的实体向量作平均
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0] //0表示不是该实体，1表示是该实体
        :return: [batch_size, dim] //每个句子返回一个实体均值向量
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    @staticmethod
    def sen_max_pooling(hidden_output, e_mask):
        """
        对bat个句子，每个句子中每个词向量每个维度上最大的值形成句子向量
        """
        a = e_mask[0].unsqueeze(0).T * hidden_output[0]
        b = a.max(0)[0]
        b = b.unsqueeze(0)
        c = b

        for i in range(1,hidden_output.shape[0]):
            a = e_mask[i].unsqueeze(0).T * hidden_output[i]
            b = a.max(0)[0]
            b = b.unsqueeze(0)
            c = torch.cat([c,b],dim=0)

        return c

    
    def semantic_feature(self, sequence_output, dep_mask, rnn):
        """
        对dep_mask中取值为1的词，通过rnn，获取其语义信息。

        return: 
        
        lstm_out: lstm的输出。
        """
    
        # 随机取一个词的某一个字词代表这个词，增加泛化性
        #for one in dep_mask:
         #   # print(one)
          #  start, end = 0,0
           # need_random = False
            #for i, item in enumerate(one):
             #   if item == 1 or item == 0:
              #            #             对一个词的子词中只取一个代表，增加泛化性
               #     if need_random :
                #        end = i
                 #       one[start:end] = 0
                  #      one[np.random.randint(start, end)] = 1
              #    #      print(one)
                    #if item == 1:
                     #   start = i
                   # need_random = False
               # if item == 2:
                #    need_random = True

        # biLSTM
        dep_mask_bool = dep_mask.type(torch.bool)

        batch_size = sequence_output.shape[0]

        # 取子句
        lstm_input = [sequence_output[0][dep_mask_bool[0]]]
        for i in range(1, sequence_output.shape[0]):
            lstm_input += [sequence_output[i][dep_mask_bool[i]]]
            
        lens = [len(x) for x in lstm_input]

        # 对bat中的句子按长度排序，以方便后续输入lstm
        lens_sorted, idx = torch.tensor(lens).sort(0, descending=True)
        _, un_idx = torch.sort(idx, dim=0)

        # 对短句补长
        lstm_input2 = rnn_utils.pad_sequence(lstm_input, batch_first=True)

        # 按长度递减排序
        lstm_input2 = lstm_input2[idx]

        # 打包以高效率的输入到LSTM中
        lstm_input3 = rnn_utils.pack_padded_sequence(lstm_input2, lens_sorted, batch_first=True)

        # 双向LSTM的初始输入
        h0 = torch.zeros(2, batch_size, self.args.lstm_hidden_size).to(self.device)
        c0 = torch.zeros(2, batch_size, self.args.lstm_hidden_size).to(self.device)

        # 通过bilstm
        out, (hn, cn) = rnn(lstm_input3,(h0, c0))
        # out, hn = rnn(lstm_input3, h0)
        # 解除打包
        out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)

        un_idx = un_idx.to(self.device)
        out_pad = out_pad.to(self.device)
        out_len = out_len.to(self.device)

        # 还原顺序，以对应标签
        out_pad = torch.index_select(out_pad, 0, un_idx)
        out_len = torch.index_select(out_len, 0, un_idx)

        # 取出双向LSTM最后时间步的两个输出且拼接 前向最后时间步前hidden个，后向第一个时间步后hidden个。
        lstm_output = torch.cat((out_pad[0][out_len[0]-1][:self.args.lstm_hidden_size], out_pad[0][0][self.args.lstm_hidden_size:])).unsqueeze(0)

        for i in range(1, out_pad.shape[0]):
            
            lstm_output = torch.cat([lstm_output, torch.cat((out_pad[i][out_len[i]-1][:self.args.lstm_hidden_size], out_pad[i][0][self.args.lstm_hidden_size:])).unsqueeze(0)], dim=0)
        
        return lstm_output


    # 最后的修改
    def forward(self, input_ids, attention_mask, token_type_ids, labels, dep_mask1):
        
        dep_mask = dep_mask1
        # if self.args.distance==0:
        #     dep_mask = dep_mask0
        # elif self.args.distance == 2:
        #     dep_mask = dep_mask2
        # elif self.args.distance == 3:
        #     dep_mask = dep_mask3

        # if self.args.no_DrugName:
        #     # SEP_index = tokenizer.convert_tokens_to_ids('[SEP]') # 102
        #     for i,item in enumerate(input_ids):
        #         sep_index = (item == 102).nonzero()[0]
        #         input_ids[i][sep_index+1:] = 0
        #         attention_mask[i][sep_index+1:] = 0

        # 两个待分类药物不使用DRUG1,DRUG2，用真的名字
        # if self.args.no_blind_drug1a2:
        #     for i,item in enumerate(input_ids):
        #         # input_ids $$之间用第一个[SEP]之间替换，##之间用第二个[SEP]替换，[SEP]及后面赋值0
        #         # attention_mask [SEP]之后赋值0
        #         # dep_mask $之间用第一个[SEP]之间token的个数个1替换，##之间用第二个[SEP]之间token的个数个1替换
        #         item = item.tolist()
                
        #         sep1_index = item.index(102)
        #         sep2_index = item.index(102,sep1_index+1)
        #         sep3_index = item.index(102,sep2_index+1)
                
        #         drug1 = input_ids[i][sep1_index+1:sep2_index].clone().detach()
        #         drug2 = input_ids[i][sep2_index+1:sep3_index].clone().detach()
        #         # 第一个
        #         d1_p = item.index(109)

        #         input_ids[i] = torch.cat((input_ids[i][:d1_p+1], drug1, input_ids[i][d1_p+2:]))[:input_ids[i].size(0)]
        #         dep_mask[i] = torch.cat((dep_mask[i][:d1_p+1], torch.ones(drug1.size(0), dtype=torch.int64).to(self.device), dep_mask0[i][d1_p+2:]))[:input_ids[i].size(0)]
                
        #         # 第二个
        #         d2_p = input_ids[i].tolist().index(108)
        #         input_ids[i] = torch.cat((input_ids[i][:d2_p+1], drug2, input_ids[i][d2_p+2:]))[:input_ids[i].size(0)]
        #         dep_mask[i] = torch.cat((dep_mask[i][:d2_p+1], torch.ones(drug2.size(0), dtype=torch.int64).to(self.device), dep_mask0[i][d2_p+2:]))[:input_ids[i].size(0)]
                
        #         attention_mask[i][sep1_index+len(drug1)+len(drug2)-2:] = 0

        # # 去掉实体标记
        # if self.args.no_entity_mark:
        #     for i,item in enumerate(input_ids):

        #         #定位到mark $ 109 # 108
        #         mark_1_1 = (item == 109).nonzero()[0]
        #         mark_1_2 = (item == 109).nonzero()[1]

        #         # 把mark去掉
        #         input_ids[i] = torch.cat((input_ids[i][:mark_1_1],input_ids[i][mark_1_1+1:mark_1_2],input_ids[i][mark_1_2+1:], torch.zeros(2, dtype=torch.int64).to(self.device)), dim=0)
        #         attention_mask[i] = torch.cat((attention_mask[i][:mark_1_1],attention_mask[i][mark_1_1+1:mark_1_2],attention_mask[i][mark_1_2+1:], torch.zeros(2, dtype=torch.int64).to(self.device)), dim=0)
        #         dep_mask[i] = torch.cat((dep_mask[i][:mark_1_1],dep_mask[i][mark_1_1+1:mark_1_2],dep_mask[i][mark_1_2+1:], torch.zeros(2, dtype=torch.int64).to(self.device)), dim=0)

        #         #定位到mark $ 109 # 108
        #         mark_2_1 = (item == 108).nonzero()[0]
        #         mark_2_2 = (item == 108).nonzero()[1]

        #         # 把mark去掉
        #         input_ids[i] = torch.cat((input_ids[i][:mark_2_1],input_ids[i][mark_2_1+1:mark_2_2],input_ids[i][mark_2_2+1:], torch.zeros(2, dtype=torch.int64).to(self.device)), dim=0)
        #         attention_mask[i] = torch.cat((attention_mask[i][:mark_2_1],attention_mask[i][mark_2_1+1:mark_2_2],attention_mask[i][mark_2_2+1:], torch.zeros(2, dtype=torch.int64).to(self.device)), dim=0)
        #         dep_mask[i] = torch.cat((dep_mask[i][:mark_2_1],dep_mask[i][mark_2_1+1:mark_2_2],dep_mask[i][mark_2_2+1:], torch.zeros(2, dtype=torch.int64).to(self.device)), dim=0)

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        
        if self.args.use_full_sequence:
            for i,item in enumerate(input_ids):
                # 以0作为结束
                sep_index = (item == 0).nonzero()[0]
                dep_mask[i,1:sep_index] = 1
                
        lstm_output1 = self.semantic_feature(sequence_output=sequence_output, dep_mask=dep_mask, rnn=self.rnn1)

        # use MLP
        MLP_output1 = self.MLP1(lstm_output1)


        logits = self.label_classifier(MLP_output1)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # # DDI 分类
                # loss_cross = nn.CrossEntropyLoss()
                # loss1 = loss_cross(logits.view(-1, self.num_labels), labels.view(-1))

                loss1 = self.Loss(logits.view(-1, self.num_labels), labels.view(-1)) * self.loss_factor

                # # DDI识别
                # loss_cross2 = nn.CrossEntropyLoss()
                # labels_binary = torch.ones_like(labels.view(-1))

                # for i, item in enumerate(labels.view(-1)):
                #     if item == 4:
                #         labels_binary[i] = 0

                # loss = loss_cross2(logits_binary.view(-1, 2), labels_binary.view(-1))

                # 看一下label 是否对应
                # loss_focalloss = MultiFocalLoss(5,[0.2, 0.2, 0.2, 0.2, 0.2])
                # loss2 = loss_focalloss(logits.view(-1, self.num_labels), labels.view(-1))

                # loss = 0.1*loss1 + 0.9*loss2
                # loss = loss1 + loss2

                # loss_my = MyLoss(theta=theta)
                # loss = loss_my(logits.view(-1, self.num_labels), labels.view(-1)   
                loss = loss1
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
