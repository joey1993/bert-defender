from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from bert_model import BertForNgramClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear

from bert_utils import *

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--word_embedding_file",
                        default='emb/crawl-300d-2M.vec',
                        type=str,
                        help="The input directory of word embeddings.")
    parser.add_argument("--index_path",
                        default='emb/p_index.bin',
                        type=str,
                        help="The input directory of word embedding index.")
    parser.add_argument("--word_embedding_info",
                        default='emb/vocab_info.txt',
                        type=str,
                        help="The input directory of word embedding info.")
    parser.add_argument("--data_file",
                        default='',
                        type=str,
                        help="The input directory of input data file.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_ngram_length",
                        default=16,
                        type=int,
                        help="The maximum total ngram sequence")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--embedding_size",
                        default=300,
                        type=int,
                        help="Total batch size for embeddings.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--num_eval_epochs',
                        type=int,
                        default=0,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--single',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    logger.info("loading embeddings ... ")
    if args.do_train:
        emb_dict, emb_vec, vocab_list, emb_vocab_size = load_vectors(args.word_embedding_file)
        write_vocab_info(args.word_embedding_info, emb_vocab_size, vocab_list)
    if args.do_eval:
        emb_vocab_size, vocab_list = load_vocab_info(args.word_embedding_info)
        #emb_dict, emb_vec, vocab_list, emb_vocab_size = load_vectors(args.word_embedding_file)
        #write_vocab_info(args.word_embedding_info, emb_vocab_size, vocab_list)
    logger.info("loading p index ...")
    if not os.path.exists(args.index_path):      
        p = load_embeddings_and_save_index(range(emb_vocab_size), emb_vec, args.index_path)
    else:
        p = load_embedding_index(args.index_path, emb_vocab_size, num_dim=args.embedding_size)

    train_examples = None
    num_train_optimization_steps = None
    w2i, i2w, vocab_size = {},{},1
    if args.do_train:
        
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        train_features, w2i, i2w, vocab_size = convert_examples_to_features_gnrt_train(\
            train_examples, label_list, args.max_seq_length, args.max_ngram_length, tokenizer, emb_dict)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Num token vocab = %d", vocab_size)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_ngram_ids = torch.tensor([f.ngram_ids for f in train_features], dtype=torch.long)
        all_ngram_labels = torch.tensor([f.ngram_labels for f in train_features], dtype=torch.long)
        all_ngram_masks = torch.tensor([f.ngram_masks for f in train_features], dtype=torch.long)
        all_ngram_embeddings = torch.tensor([f.ngram_embeddings for f in train_features], dtype=torch.float)

    # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(args.local_rank))
        model = BertForNgramClassification.from_pretrained(args.bert_model, 
                                                        cache_dir=cache_dir, 
                                                        num_labels=num_labels, 
                                                        embedding_size=args.embedding_size, 
                                                        max_seq_length=args.max_seq_length, 
                                                        max_ngram_length=args.max_ngram_length)
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                warmup=args.warmup_proportion,
                                t_total=num_train_optimization_steps)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0

    #if args.do_train:

        train_data = TensorDataset(all_ngram_ids, all_ngram_labels, all_ngram_masks, all_ngram_embeddings)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for ind in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                ngram_ids, ngram_labels, ngram_masks, ngram_embeddings = batch
                loss = model(ngram_ids, ngram_masks, ngram_embeddings) 
                if n_gpu > 1:
                    loss = loss.mean() 

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            loss = tr_loss/nb_tr_steps if args.do_train else None
            result = {
                    'loss': loss,
                    }

            output_eval_file = os.path.join(args.output_dir, "train_results.txt")
            with open(output_eval_file, "a") as writer:
                #logger.info("***** Training results *****")
                writer.write("epoch"+str(ind)+'\n')
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
                writer.write('\n')
                    
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir, "epoch"+str(ind)+WEIGHTS_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())

    # Load a trained model and config that you have fine-tuned
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        eval_examples = processor.get_gnrt_dev_examples(args.data_file)
        eval_features, w2i, i2w, vocab_size = convert_examples_to_features_gnrt_eval(
            eval_examples, label_list, args.max_seq_length, args.max_ngram_length, tokenizer, w2i, i2w, vocab_size)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Num token vocab = %d", vocab_size)
        logger.info("  Batch size = %d", args.eval_batch_size)
    
        all_token_ids = torch.tensor([f.token_ids for f in eval_features], dtype=torch.long)
        # all_flaw_labels: indexes of wrong words predicted by disc
        all_flaw_labels = torch.tensor([f.flaw_labels for f in eval_features], dtype=torch.long) 
        all_ngram_ids = torch.tensor([f.ngram_ids for f in eval_features], dtype=torch.long)
        all_ngram_mask = torch.tensor([f.ngram_mask for f in eval_features], dtype=torch.long)
        all_ngram_labels = torch.tensor([f.ngram_labels for f in eval_features], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_token_ids, all_ngram_ids, all_ngram_mask, all_ngram_labels, all_label_id, all_flaw_labels)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if args.single:
            eval_range = trange(int(args.num_eval_epochs), int(args.num_eval_epochs+1), desc="Epoch")
        else:
            eval_range = trange(int(args.num_eval_epochs), desc="Epoch")

        for epoch in eval_range:

            output_file = os.path.join(args.data_dir, "epoch"+str(epoch)+"gnrt_outputs.tsv")
            with open(output_file,"w") as csv_file:
                writer = csv.writer(csv_file, delimiter='\t')
                writer.writerow(["sentence", "label"])

            output_model_file = os.path.join(args.output_dir, "epoch"+str(epoch)+WEIGHTS_NAME)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            config = BertConfig(output_config_file)
            model = BertForNgramClassification(config,
                                                num_labels=num_labels,
                                                embedding_size=args.embedding_size,
                                                max_seq_length=args.max_seq_length, 
                                                max_ngram_length=args.max_ngram_length)
            model.load_state_dict(torch.load(output_model_file))
            model.to(device)
            model.eval()
    
            for token_ids, ngram_ids, ngram_mask, ngram_labels, label_id, flaw_labels in tqdm(eval_dataloader, desc="Evaluating"):
                
                ngram_ids = ngram_ids.to(device)
                ngram_mask = ngram_mask.to(device)

                with torch.no_grad():
                    logits = model(ngram_ids, ngram_mask)

                logits = logits.detach().cpu().numpy()
                flaw_labels = flaw_labels.to('cpu').numpy()
                label_id = label_id.to('cpu').numpy()
                token_ids = token_ids.to('cpu').numpy()
                masks = ngram_mask.to('cpu').numpy()


                with open(output_file,"a") as csv_file:

                    for i in range(len(label_id)):
                        
                        correct_tokens = look_up_words(logits[i], masks[i], vocab_list, p) 
                        token_new = replace_token(token_ids[i], flaw_labels[i], correct_tokens, i2w)
                        token_new = ' '.join(token_new)
                        label = str(label_id[i])
                        writer = csv.writer(csv_file, delimiter='\t')
                        writer.writerow([token_new, label])


if __name__ == "__main__":
    main() 
