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

        self.rnn1 = nn.LSTM(input_size=config.hidden_size, hidden_size=args.lstm_hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        # MLP for dep1
        self.MLP1 = FCLayer(
            args.lstm_hidden_size*2,
            args.MLP_hidden_size,
            args.dropout_rate,
            use_activation=True,
        )

        self.label_classifier = FCLayer( 
            args.MLP_hidden_size,
            config.num_labels,
            args.dropout_rate,
            use_activation=False,
        )

    def semantic_feature(self, sequence_output, dep_mask, rnn):
        """
        对dep_mask中取值为1的词，通过rnn，获取其语义信息。

        return: 
        
        lstm_out: lstm的输出。
        """
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
                
                loss1 = self.Loss(logits.view(-1, self.num_labels), labels.view(-1)) * self.loss_factor
                loss = loss1
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
