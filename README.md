# no_blinding_KSS_GHM

You need to download the BERT model such as BioBERT from https://huggingface.co/models? put in the directory BERT/

You need to download the stanza in the directory stanza/.

python main.py  --do_train --do_eval  --eval_filename='best.txt' --Loss='GHM' --seed=73 --no_blind --max_seq_len=390 --model_name_or_path="./BERT/biobert_v1.1_pubmed"
