import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup

from model import RBERT
from utils import compute_metrics, get_label, write_prediction

import matplotlib.pyplot as plt

import torch.nn.functional as F
import random

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, tokenizer_length=1):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.label_lst = get_label(args)
        self.num_labels = len(self.label_lst)

        self.config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(self.label_lst)},
            label2id={label: i for i, label in enumerate(self.label_lst)},
        )
        self.model = RBERT.from_pretrained(args.model_name_or_path, config=self.config, args=args)
        # resize model embedding because of add other words
        self.model.resize_token_embeddings(tokenizer_length)
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
        )

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        bert_params = []
        not_bert_params = []
        
        print("******************************************************")
        for name, params in self.model.named_parameters():
            if 'bert' in name:
                bert_params += [name]
            else:
                print(name)
                not_bert_params += [name]
        print("******************************************************")
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and (n in not_bert_params)],
                "weight_decay": self.args.weight_decay,
                "lr":self.args.learning_rate_other,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and (n in bert_params)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and (n in not_bert_params)],
                "weight_decay": 0.0,
                "lr":self.args.learning_rate_other,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)  and (n in bert_params)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate_bert,
            eps=self.args.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        # train/dev loss F1 value
        train_loss = []
        dev_loss = []
        f1_values = []

        loss_total_logging_step = 0.0

        # 记录最好的F1值

        Best_result = 0
        i = 0
        
        for _ in train_iterator:
    
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):

                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "dep_mask1": batch[4],
                }

                outputs = self.model(**inputs)

                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    # to calculate total logging_step loss
                    loss_total_logging_step += loss

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        
                        print("global:",global_step)
                        train_loss.append(loss_total_logging_step/self.args.logging_steps)
                        loss_total_logging_step = 0.0

                        results = self.evaluate("dev")  # There is no dev set for semeval task

                        dev_loss.append(results['loss'])
                        f1_values.append(results['f1'])

                        logger.info("The Epoch  is %d"%(i))
                        logger.info("The Best F1 is %.4f"%(Best_result))

                        if results['f1'] > Best_result:
                            logger.info("Congratulations! F1_score Up!!! from %.4f to %.4f"%(Best_result, results['f1']))
                            self.save_model()
                            Best_result = results['f1']
                        
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
            i += 1

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
        
        # plot loss curve
        self.plot_loss_curve(train_loss, dev_loss, f1_values)
        
        return global_step, tr_loss / global_step

    def plot_loss_curve(self, train_loss, dev_loss, f1_values):

        x = range(1,len(train_loss)+1)
        plt.plot(x,train_loss, label='train loss')
        plt.plot(x,dev_loss, label='dev loss')
        plt.legend()
        plt.savefig('loss_curves.png')
        plt.show()

        x = range(1,len(train_loss)+1)
        fig, axs = plt.subplots(2, 1)
        
        axs[0].plot(x,train_loss, label='train loss')
        axs[0].plot(x,dev_loss, label='dev loss')
        axs[1].plot(x,f1_values, label='dev f1_value')
        
        axs[0].legend()
        axs[1].legend()
        plt.savefig('loss_f1_curves.png')
        plt.show()

        with open('result.txt', 'a') as f:
            f.write("**************************************************************************************\n")
            f.write("Note: %s\n"%(self.args.note))
            f.write("Seed: %d\n"%(self.args.seed))
            f.write("dev_loss: \nBest :%s  All: "%"".join(str(np.max(dev_loss))))
            f.write("%s\n" % " ".join([str(x) for x in dev_loss]))
            f.write("f1_values: \nBest :%s  All: "%"".join(str(np.max(f1_values))))
            f.write("%s\n" % " ".join([str(x) for x in f1_values]))
            

    def evaluate(self, mode, training=True):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "dep_mask1": batch[4],
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        p = torch.from_numpy(preds) 
        preds_write = F.softmax(p, dim=1)
        
        preds = np.argmax(preds, axis=1)

        v, _ = torch.max(preds_write, axis=1)

        if mode == 'test':
            write_prediction(self.args, os.path.join(self.args.eval_dir, self.args.eval_filename), preds, v)

        result = compute_metrics(preds, out_label_ids, no_blind=self.args.no_blind)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  {} = {:.4f}".format(key, results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        self.args = torch.load(os.path.join(self.args.model_dir, "training_args.bin"))
        self.model = RBERT.from_pretrained(self.args.model_dir, args=self.args)
        self.model.to(self.device)
        logger.info("***** Model Loaded *****")
        
