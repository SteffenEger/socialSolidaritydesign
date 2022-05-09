from transformers import AutoModelForSequenceClassification, AdamW, AutoTokenizer
from utils.pytorchtools import EarlyStopping
from utils.data_processor import build_dataloader
from sklearn import metrics
from argparse import ArgumentParser
import os
import random
import numpy as np
import torch
import csv
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def train(dataloader_train, model, optimizer, device):
    model.train()
    start = time.time()
    for i, data in enumerate(dataloader_train, 0):
        optimizer.zero_grad()


        if (i % 1000 == 0 and i != 0):
            end = time.time()
            print(f'Iteration {i} finished, Processing Time {int((end - start) / 60)} minutes')
            start = end

        ids = data['ids'].to(device, dtype=torch.long)

        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, labels=targets)
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()


def evaluate(dataloader_val, model, device):
    model.eval()
    pred, gold = [], []

    for _, data in enumerate(dataloader_val):
        ids = data['ids'].to(device, dtype=torch.long)

        targets = data['targets'].to(device, dtype=torch.long)

        with torch.no_grad():
            outputs = model(ids, labels=targets)
            _, idx = torch.max(outputs[1].data, dim=1)

        pred.extend(idx.tolist())
        gold.extend(targets.tolist())

    return pred, gold


def show_performance(truth, pred):
    precision = metrics.precision_score(truth, pred, average=None)
    recall = metrics.recall_score(truth, pred, average=None)
    f1 = metrics.f1_score(truth, pred, average=None)
    print(f'Precision:{precision}, Recall:{recall},  F1: {f1}')
    f1_score_micro = metrics.f1_score(truth, pred, average='micro')
    f1_score_macro = metrics.f1_score(truth, pred, average='macro')
    print(f"F1 Score (Micro) = {f1_score_micro}, F1 Score (Macro) = {f1_score_macro}")
    return f1_score_macro


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument("--model_type", type=str, required=True,
                        help="bert or xlm")
    parser.add_argument('--split', type=int, required=True, help="Choose split from 1/2/3")        
    
    
    parser.add_argument('--oversample_from_train', action="store_true")
    parser.add_argument('--translation', action="store_true")
    parser.add_argument('--auto_data', action="store_true")
    parser.add_argument("--use_pretrain", action="store_true")
    
    parser.add_argument('--model_dir', default='saved_weights')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--do_merge', type=bool, default=True,
                        help="Whether merge ambivalent and non-applicable classes")
    parser.add_argument("--max_len", type=int, default=150, help="Maximal sequence length of bert or xlm model")
    parser.add_argument("--epochs", type=int, default=10, help="Maximal number of epochs to train")
    parser.add_argument("--patience", type=int, default=5,
                        help="How many epochs to wait after last improvement (for early stopping).")
    parser.add_argument("--batch_size", default=16, type=int, help="The batch size for training.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument('--seed', type=int, default=32, help="random seed")
    parser.add_argument('--do_lower_case', action="store_true", help="Whether lowercase text before tokenization")
    
    

    


    args = parser.parse_args()

    # fix random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(
        f'max_len={args.max_len}, batch_size={args.batch_size}, epochs={args.epochs}, learning_rate={args.learning_rate}, seed={args.seed}, patience={args.patience}, do_lower_case={args.do_lower_case} split = {args.split}')

    strategies = []
    strategies.append(args.model_type)
    if args.oversample_from_train:
        strategies.append("oversample_from_train")
    if args.translation:
        strategies.append("translation")
    if args.auto_data:
        strategies.append("auto_label")
    if args.use_pretrain:
        strategies.append("pretrain")

    strategies.append('split' + str(args.split))
   
    strategies = ' '.join(strategies)
    print(strategies)

    num_labels = 3 if args.do_merge else 4

    if args.model_type == 'bert':
        weight = 'bert-base-multilingual-cased'
    if args.model_type == 'xlm':
        weight = 'xlm-roberta-base'
    if args.use_pretrain:
        weight = 'pretrain/xlm-r_finetune_mlm_100k'
    tokenizer = AutoTokenizer.from_pretrained(weight, do_lower_case=args.do_lower_case)
    model = AutoModelForSequenceClassification.from_pretrained(weight, num_labels=num_labels).to(args.device)

    dataloader_train, dataloader_dev, dataloader_test = build_dataloader(src_path=args.data_dir, split=args.split,
                                                                         do_merge=args.do_merge,
                                                                         tokenizer=tokenizer,
                                                                         max_len=args.max_len,
                                                                         batch_size=args.batch_size,
                                                                         oversample_from_train=args.oversample_from_train,
                                                                         translation=args.translation,
                                                                         auto_data=args.auto_data
                                                                         )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    model_path = os.path.join(args.model_dir, f'{strategies}.bin')
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, monitor='val_f1')

    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}')
        start = time.time()
        train(dataloader_train, model, optimizer, device=args.device)
        predictions, true_vals = evaluate(dataloader_dev, model, device=args.device)
        val_f1 = metrics.f1_score(true_vals, predictions, average='macro')
        end = time.time()
        print(f'Epoch {epoch} Processing Time {int((end - start) / 60)} minutes')
        print(f'F1 score (macro) on dev set: {val_f1}')
        # model_path = os.path.join('saved_weights', f'{strategies}_{str(round(val_f1,4))}.bin')

        early_stopping(val_f1, model, model_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # if val_f1 < early_stopping.best_score:
        # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    predictions, true_vals = evaluate(dataloader_train, model, args.device)
    print('Train scores:', metrics.f1_score(true_vals, predictions, average=None),
          metrics.f1_score(true_vals, predictions, average='macro'))
    print('Evaluation....')
    predictions, true_vals = evaluate(dataloader_dev, model, args.device)
    f1_val = show_performance(true_vals, predictions)
    print('Testing....')
    predictions, true_vals = evaluate(dataloader_test, model, args.device)
    f1_test = show_performance(true_vals, predictions)

    with open('results.csv', 'a+', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([model_path, round(f1_val, 3), round(f1_test, 3)])

