import logging
import os
import random

import numpy as np
import torch
from transformers import BertTokenizer, AutoTokenizer

from sklearn.metrics import f1_score

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8")]


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    tokenizer.add_tokens(['DRUG0', 'DRUG1', 'DRUG2'])
    return tokenizer


def write_prediction(args, output_file, preds, p):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)

    # binary
    # relation_labels = ['other', 'ddi']


    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\t{}\n".format(8001 + idx, relation_labels[pred], p[idx]))


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    # args.seed = random.randint(1, 1000)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels, no_blind):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels, no_blind=no_blind)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, no_blind=False, average="macro"):
    acc = simple_accuracy(preds, labels)

    # the positive instances filtered by negative instance filted strategy
    labels = np.append(labels, [1,1,3,3,3,3,1])
    preds = np.append(preds, [4]*7)

    if no_blind:
        labels = np.append(labels, [1])
        preds = np.append(preds, [4])

    F = f1_score(labels, preds, labels=[0,1,2,3], average='micro')

    return {
        "acc": acc,
        "f1": F,
    }
