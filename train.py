from transformers import AutoTokenizer, BertModel
from utils.models import BertForSequenceClassification, XLMRobertaForSequenceClassification
from transformers import AutoModelForSequenceClassification, AdamW
from utils.pytorchtools import EarlyStopping
from utils.models import bertCNN, bertDPCNN
from utils.data_processor import build_dataloader
from sklearn import metrics
from argparse import ArgumentParser
import os
import random
import numpy as np
import torch
from torch import cuda
import csv
import time 
device = 'cuda' if cuda.is_available() else 'cpu'


def train(dataloader_train, model, optimizer, is_norm=False):
    
    model.train()

    loss_train_total = 0

    for i, data in enumerate(dataloader_train, 0):
        if(i==0):
            start = time.time()
        if(i%1000==0 and i!=0):
            end = time.time()
            print(f'iteration {i} finished, processing time {end-start}')
            start = end
            
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        if is_norm:
            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, labels=targets, is_norm=is_norm)
        else:
            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, labels=targets)
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    loss_train_avg = loss_train_total / len(dataloader_train)
    
    print(f'Training loss: {loss_train_avg}')


def evaluate(dataloader_val, model, is_norm=False):
    model.eval()
    predictions, true_vals = [], []

    for _, data in enumerate(dataloader_val, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        with torch.no_grad():
            if is_norm:
                outputs = model(ids, mask, token_type_ids, labels=targets, is_norm=is_norm)
            else:
                outputs = model(ids, mask, token_type_ids, labels=targets)
        big_val, big_idx = torch.max(outputs[1].data, dim=1)

        predictions.extend(big_idx.tolist())
        true_vals.extend(targets.tolist())

    return predictions, true_vals


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
    parser.add_argument("--model_type", type=str, required=True,
                        help="Choose model type from bert, xlm, bert_cnn, bert_dpcnn")
    parser.add_argument("--pretrained_weights", type=str, required=True,
                        help="""Name of the pretrained weights from huggingface transformers 
                        (e.g. 'bert-base-multilingual-cased', 'xlm-roberta-base'), or path of self-trained weights""")
    #parser.add_argument("--model_path", default=None, type=str,
                        #help="path to save the model")

    parser.add_argument("--oversample_from_train", action="store_true",
                        help="Whether do oversampling from training data")
    parser.add_argument("--oversample_from_trans", action="store_true",
                        help="Whether do oversampling from translated data")
    parser.add_argument("--translation", action="store_true",
                        help="Whether add translated data for training")
    parser.add_argument("--auto_data", action="store_true",
                        help="Whether add auto-labeled data for training")
    parser.add_argument("--is_norm", action="store_true",
                        help="Whether add batch normalization after the last hidden layer of bert or xlm model")

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
    parser.add_argument('--split', type = int, required = True, help="Choose split from 1/2/3")
    parser.add_argument('--expert', action="store_true", help="Only contained expert annotations in the train set")
    parser.add_argument('--expert_only_hashtags', action="store_true", help="Only contained expert annotations (after removing everything except hashtags) in the train set")
    parser.add_argument('--all_only_hashtags', action="store_true", help="contained expert + crowd annotations (after removing everything except hashtags) in the train set")
    parser.add_argument('--all_no_hashtags', action="store_true", help="contained expert + crowd annotations (after removing hashtags) in the train set")


    args = parser.parse_args()

    # fix random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # define some key variables for training
    DO_MERGE = args.do_merge
    MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    print(
        f'max_len={MAX_LEN}, batch_size={BATCH_SIZE}, epochs={EPOCHS}, learning_rate={LEARNING_RATE}, seed={args.seed}, patience={args.patience}, do_lower_case={args.do_lower_case} split = {args.split}')
    
    strategies = []
    strategies.append(args.model_type)
    if args. oversample_from_train:
        strategies.append("oversample_from_train")
    if args.oversample_from_trans:
        strategies.append("oversample_from_trans")
    if args.translation:
        strategies.append("translation")
    if args.auto_data:
        strategies.append("auto_label")
    if args.is_norm:
        strategies.append("normalization")
    if args.expert:
        strategies.append("expert")
    if args.expert_only_hashtags:
        strategies.append("expert_annotations_with_only_hashtags")
    if args.all_only_hashtags:
        strategies.append("expert+crowd_annotations_with_only_hashtags")
    if args.all_no_hashtags:
        strategies.append("expert+crowd_annotations_with_no_hashtags")
    
    strategies.append('split'+str(args.split))
    strategies.append(args.pretrained_weights.split('/')[-1])
    
    strategies = ' '.join(strategies)
    print(strategies)
    
    
    
     

    num_labels = 3 if DO_MERGE else 4

    if args.model_type in ['bert', 'xlm']:
        if args.is_norm:
            if args.model_type == 'bert':
                model = BertForSequenceClassification.from_pretrained(args.pretrained_weights, num_labels=num_labels,
                                                                  output_hidden_states=True)
            else:
                model = XLMRobertaForSequenceClassification.from_pretrained(args.pretrained_weights, num_labels=num_labels,
                                                                        output_hidden_states=True)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_weights, num_labels=num_labels)
    elif args.model_type == 'bert_cnn':
        embed_model = BertModel.from_pretrained(args.pretrained_weights)
        model = bertCNN(embed_model=embed_model, dropout=0.2, kernel_num=4, kernel_sizes=[3, 4, 5, 6],
                            num_labels=num_labels)
    elif args.model_type == 'bert_dpcnn':
        embed_model = BertModel.from_pretrained(args.pretrained_weights)
        model = bertDPCNN(embed_model=embed_model, num_filters=100, num_labels=num_labels)
    else:
        raise ValueError(
            'Error model type! Model type must be one of the following 4 types: bert, xlm, bert_cnn or bert_dpcnn')

    print(f'model_type={args.model_type}, pretrained_weights={args.pretrained_weights}, num_labels={num_labels}')

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_weights, do_lower_case=args.do_lower_case)

    dataloader_train, dataloader_dev, dataloader_test = build_dataloader(src_path=f'data', split = args.split, do_merge=DO_MERGE,
                                                                         tokenizer=tokenizer, max_len=MAX_LEN,
                                                                         batch_size=BATCH_SIZE,
                                                                         oversample_from_train=args.oversample_from_train,
                                                                         oversample_from_trans=args.oversample_from_trans,
                                                                         translation=args.translation,
                                                                         auto_data=args.auto_data,
                                                                         expert = args.expert,
                                                                         expert_only_hashtags = args.expert_only_hashtags,
                                                                         all_only_hashtags = args.all_only_hashtags,
                                                                         all_no_hashtags = args.all_no_hashtags)

    model.to(device)
    optimizer = AdamW(model.parameters(),
                      lr=LEARNING_RATE,
                      eps=1e-8)
    model_path = os.path.join('saved_weights', f'{strategies}.bin')
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, monitor='val_f1')
 
    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch {epoch}')
        start = time.time()
        train(dataloader_train, model, optimizer, is_norm=args.is_norm)
        predictions, true_vals = evaluate(dataloader_dev, model, is_norm=args.is_norm)
        val_f1 = metrics.f1_score(true_vals, predictions, average='macro')
        end = time.time()
        print(f'for epoch {epoch} the training time is {end-start}')
        print(f'F1 score (macro) on dev set: {val_f1}')
        #model_path = os.path.join('saved_weights', f'{strategies}_{str(round(val_f1,4))}.bin')
  
        early_stopping(val_f1, model,model_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # if val_f1 < early_stopping.best_score:
        # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    predictions, true_vals = evaluate(dataloader_train, model, is_norm=args.is_norm)
    print('Train scores:', metrics.f1_score(true_vals, predictions, average=None),
          metrics.f1_score(true_vals, predictions, average='macro'))
    print('Evaluation....')
    predictions, true_vals = evaluate(dataloader_dev, model, is_norm=args.is_norm)
    f1_val = show_performance(true_vals, predictions)
    print('Testing....')
    predictions, true_vals = evaluate(dataloader_test, model, is_norm=args.is_norm)
    f1_test = show_performance(true_vals, predictions)
    
    with open('results.csv','a+',newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([model_path,round(f1_val,3),round(f1_test,3)])

