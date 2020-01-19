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
from bert_model import BertForDiscriminator, BertConfig, WEIGHTS_NAME, CONFIG_NAME
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
                        default='./emb/crawl-300d-2M.vec',
                        type=str,
                        help="The input directory of word embeddings.")
    parser.add_argument("--index_path",
                        default='./emb/p_index.bin',
                        type=str,
                        help="The input directory of word embedding index.")
    parser.add_argument("--word_embedding_info",
                        default='./emb/vocab_info.txt',
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
    parser.add_argument("--num_eval_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of eval epochs to perform.")
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
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--single',
                        action='store_true',
                        help="Whether only evaluate a single epoch")
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

    train_examples = None
    num_train_optimization_steps = None
    w2i, i2w, vocab_size = {},{},1
    if args.do_train:
        
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        train_features,w2i,i2w,vocab_size = convert_examples_to_features_disc_train(train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Num token vocab = %d", vocab_size)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_tokens = torch.tensor([f.token_ids for f in train_features], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    # load embeddings
    if args.do_train:
        logger.info("Loading word embeddings ... ")
        emb_dict, emb_vec, vocab_list, emb_vocab_size = load_vectors(args.word_embedding_file)
        if not os.path.exists(args.index_path):
            
            write_vocab_info(args.word_embedding_info, emb_vocab_size, vocab_list)
            p = load_embeddings_and_save_index(range(emb_vocab_size), emb_vec, args.index_path)
        else:
            #emb_vocab_size, vocab_list = load_vocab_info(args.word_embedding_info)
            p = load_embedding_index(args.index_path, emb_vocab_size, num_dim=args.embedding_size)
        #emb_dict, emb_vec, vocab_list, emb_vocab_size, p = None, None, None, None, None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(args.local_rank))
    model = BertForDiscriminator.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels)
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
    nb_tr_steps = 1
    tr_loss = 0
    if args.do_train:

        train_data = TensorDataset(all_tokens, all_label_id) 
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for ind in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            flaw_eval_f1 = []
            flaw_eval_recall = []
            flaw_eval_precision = []
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                tokens,_ = batch #, label_id, ngram_ids, ngram_labels, ngram_masks

                # module1: learn a discriminator
                tokens = tokens.to('cpu').numpy()
                train_features = convert_examples_to_features_flaw(tokens, 
                                                                   args.max_seq_length, 
                                                                   args.max_ngram_length, 
                                                                   tokenizer, 
                                                                   i2w,
                                                                   emb_dict,
                                                                   p,
                                                                   vocab_list) 

                flaw_mask = torch.tensor([f.flaw_mask for f in train_features], dtype=torch.long).to(device)     # [1, 1, 1, 1, 0,0,0,0]
                flaw_ids = torch.tensor([f.flaw_ids for f in train_features], dtype=torch.long).to(device)       # [12,25,37,54,0,0,0,0]
                flaw_labels = torch.tensor([f.flaw_labels for f in train_features], dtype=torch.long).to(device) # [0, 1, 1, 1, 0,0,0,0]

                loss, logits = model(flaw_ids, flaw_mask, flaw_labels)
                logits = logits.detach().cpu().numpy()

                if n_gpu > 1:
                    loss = loss.mean() 

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()

                nb_tr_examples += flaw_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # eval during training
                flaw_labels = flaw_labels.to('cpu').numpy()
                
                flaw_tmp_eval_f1, flaw_tmp_eval_recall, flaw_tmp_eval_precision = f1_3d(logits, flaw_labels)
                flaw_eval_f1.append(flaw_tmp_eval_f1)
                flaw_eval_recall.append(flaw_tmp_eval_recall)
                flaw_eval_precision.append(flaw_tmp_eval_precision)

                nb_eval_examples += flaw_ids.size(0)
                nb_eval_steps += 1

            flaw_f1 = sum(flaw_eval_f1)/len(flaw_eval_f1)
            flaw_recall = sum(flaw_eval_recall)/len(flaw_eval_recall)
            flaw_precision = sum(flaw_eval_precision)/len(flaw_eval_precision)
            loss = tr_loss/nb_tr_steps if args.do_train else None
            result = {
                    'flaw_f1': flaw_f1,
                    "flaw_recall": flaw_recall,
                    "flaw_precision": flaw_precision,
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

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        eval_examples = processor.get_disc_dev_examples(args.data_file)
        eval_features,w2i,i2w,vocab_size = convert_examples_to_features_disc_eval(
            eval_examples, label_list, args.max_seq_length, tokenizer, w2i, i2w, vocab_size)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Num token vocab = %d", vocab_size)
        logger.info("  Batch size = %d", args.eval_batch_size)
    
        all_token_ids = torch.tensor([f.token_ids for f in eval_features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_flaw_labels = torch.tensor([f.flaw_labels for f in eval_features], dtype=torch.long)
        all_flaw_ids = torch.tensor([f.flaw_ids for f in eval_features], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_chunks = torch.tensor([f.chunks for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_token_ids, all_input_ids, all_input_mask, all_flaw_ids, all_flaw_labels, all_label_id, all_chunks)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Load a trained model and config that you have fine-tuned
        if args.single:
            eval_range = trange(int(args.num_eval_epochs), int(args.num_eval_epochs+1), desc="Epoch")
        else:
            eval_range = trange(int(args.num_eval_epochs), desc="Epoch")

        for epoch in eval_range:

            output_file = os.path.join(args.data_dir, "epoch"+str(epoch)+"disc_outputs.tsv")
            with open(output_file,"w") as csv_file:
                writer = csv.writer(csv_file, delimiter='\t')
                writer.writerow(["sentence", "label", "ids"])

            output_model_file = os.path.join(args.output_dir, "epoch"+str(epoch)+WEIGHTS_NAME)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

            config = BertConfig(output_config_file)
            model = BertForDiscriminator(config, num_labels=num_labels)
            model.load_state_dict(torch.load(output_model_file))

            model.to(device)
            model.eval()
            predictions, truths = [], []
            eval_loss, nb_eval_steps, nb_eval_examples = 0, 0, 0
            eval_accuracy = 0
    
            for token_ids, input_ids, input_mask, flaw_ids, flaw_labels, label_id, chunks in tqdm(eval_dataloader, desc="Evaluating"):
                
                token_ids = token_ids.to(device)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                flaw_labels = flaw_labels.to(device)
                flaw_ids = flaw_ids.to(device)

                with torch.no_grad():
                    tmp_eval_loss,_ = model(input_ids, input_mask, flaw_labels)
                    logits = model(input_ids, input_mask)
                    flaw_logits = torch.argmax(logits, dim=2)

                logits = logits.detach().cpu().numpy()
                flaw_logits = flaw_logits.detach().cpu().numpy()
                flaw_ids = flaw_ids.to('cpu').numpy()
                label_id = label_id.to('cpu').numpy()
                chunks = chunks.to('cpu').numpy()
                token_ids = token_ids.to('cpu').numpy()
                
                flaw_logits = logit_converter(flaw_logits, chunks) # each word only has one '1'
                true_logits = []
                for i in range(len(flaw_ids)):
                    tmp = [0] * len(flaw_logits[i])
                    for j in range(len(flaw_ids[0])):
                        if flaw_ids[i][j] == 0: break
                        if flaw_ids[i][j] >= len(tmp): continue
                        tmp[flaw_ids[i][j]] = 1

                    true_logits.append(tmp)

                tmp_eval_accuracy = accuracy_2d(flaw_logits, true_logits)
                eval_accuracy += tmp_eval_accuracy 

                predictions += true_logits
                truths += flaw_logits
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

                with open(output_file, "a") as csv_file:
                    for i in range(len(label_id)):
                        token = ' '.join([i2w[x] for x in token_ids[i] if x != 0])
                        flaw_logit = flaw_logits[i]
                        label = str(label_id[i])
                        logit = ','.join([str(i) for i,x in enumerate(flaw_logit) if x == 1]) 
                        logit = '-1' if logit == '' else logit
                        writer = csv.writer(csv_file, delimiter='\t')
                        writer.writerow([token, label, logit])

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_steps
            eval_f1_score, eval_recall_score, eval_precision_score  = f1_2d(truths, predictions)
            loss = tr_loss/nb_tr_steps if args.do_train else None
            result = {'eval_loss': eval_loss,
                    'eval_f1': eval_f1_score,
                    'eval_recall': eval_recall_score,
                    'eval_precision': eval_precision_score,
                    'eval_acc': eval_accuracy}

            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        
if __name__ == "__main__":
    main() 
