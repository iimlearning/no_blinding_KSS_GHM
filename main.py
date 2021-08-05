import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse

from data_loader import load_and_cache_examples
from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed
# import stanza




def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)
    
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")

    trainer = Trainer(args, train_dataset=train_dataset, test_dataset=dev_dataset, dev_dataset=test_dataset,tokenizer_length=len(tokenizer))

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="ddi", type=str, help="The name of the task to train")
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument(
        "--eval_dir",
        default="./eval",
        type=str,
        help="Evaluation script, result directory",
    )
    parser.add_argument("--train_file", default="train_dep.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="test_dep.tsv", type=str, help="Test file")
    parser.add_argument("--dev_file", default="dev_dep.tsv", type=str, help="Dev file")

    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./BERT/biobert_v1.1_pubmed",
        help="Model Name or Path eg:scibert_scivocab_uncased/",
    )

    parser.add_argument("--seed", type=int, default=73, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--max_seq_len",
        default=390,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--learning_rate_bert",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam for Bert.",
    )

    parser.add_argument(
        "--learning_rate_other",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam for unbert.",
    )

    parser.add_argument(
        "--lstm_hidden_size",
        default=768,
        type=int,
        help="The hidden size of BiLSTM"
    )

    parser.add_argument(
        "--MLP_hidden_size",
        default=300,
        type=int,
        help="The hidden size of MLP"
    )

    parser.add_argument(
        "--num_train_epochs",
        default=10.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some. 0.01")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout for fully-connected layers",
    )

    parser.add_argument("--logging_steps", type=int, default=250, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=250,
        help="Save checkpoint every X updates steps.",
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--add_sep_token",
        action="store_true",
        help="Add [SEP] token at the end of the sentence",
    )
   
    parser.add_argument("--use_full_sequence", action="store_true", help="Whether to use full_sequence instead of dependency sequence")
    parser.add_argument("--Loss", type=str, default='GHM',help="Whick Loss we use eg.GHM Focal crossentropy")
    parser.add_argument("--loss_factor", type=int, default=1, help="Loss amplification factor, because the focal loss is so small.")
    parser.add_argument("--bins", default=5, type=int, help="GHM parameters: Number of the unit regions for distribution calculation.")
    parser.add_argument("--alpha", default=0.75, type=float, help="GHM parameters: The parameter for moving average.")
    parser.add_argument("--no_blind", action="store_true", help="Use no_blinded data to ddie")
    parser.add_argument("--no_entity_mark", action="store_true", help="delete # and $ mark !!! need to repreprocessing the data")
    parser.add_argument("--label_noise_rate", type=float, default=-1, help="The probability that the instance label be changed to false.")
    parser.add_argument("--eval_filename", type=str, default='proposed_answers.txt', help="eval result saving filename")
    parser.add_argument("--note", default="None", type=str, help="Something will be writen in order to record")
    
    args = parser.parse_args()

    main(args)
