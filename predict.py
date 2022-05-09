from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.data_processor import convert_data_into_features
from train import show_performance
import glob


def predict(model, test_loader, device):
    model.eval()
    preds = []

    for _, data in enumerate(test_loader):
        ids = data['ids'].to(device, dtype=torch.long)

        with torch.no_grad():
            outputs = model(ids)
            _, big_idx = torch.max(outputs[0].data, dim=1)
            preds.extend(big_idx.tolist())

    return preds


def ensemble_predict(df, model_paths, has_label):
    num_labels = 3 if args.do_merge else 4

    pred_all = []

    for model_path in model_paths:
        print(model_path)
        if model_path.split('/')[-1].startswith('bert'):
            weights = 'bert-base-multilingual-cased'
        # elif model_path.split('/')[-1].startswith('xlm_'):
        # weights = 'fine_tune/xlm-r_finetune_mlm_100k'
        else:
            weights = 'xlm-roberta-base'
        tokenizer = AutoTokenizer.from_pretrained(weights, do_lower_case=args.do_lower_case)
        model = AutoModelForSequenceClassification.from_pretrained(weights, num_labels=num_labels).to(args.device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

        test_loader = convert_data_into_features(df, tokenizer, max_len=args.max_len, batch_size=args.batch_size,
                                                 shuffle=False, has_label=args.has_label)

        preds = predict(model, test_loader, args.device)
        if has_label:
            f1_score_macro = show_performance(df.label, preds)

        pred_all.append(preds)

    if len(pred_all) == 1:
        return pred_all

    else:
        pred_all = list(map(list, zip(*pred_all)))
        pred_ensemble = [np.argmax(np.bincount(x)) for x in pred_all]
        return pred_all, pred_ensemble


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)

    parser.add_argument("--pred_dir", type=str, default='predictions')

    # parser.add_argument("--num_labels", type=int, default=3, help="The number of classes for classification")
    parser.add_argument('--do_lower_case', action="store_true", help="Whether lowercase text before tokenization")
    parser.add_argument('--has_label', action="store_true")
    parser.add_argument('--do_merge', type=bool, default=True,
                        help="When we have labels and we want to merge class 2 and class 3")
    parser.add_argument("--max_len", type=int, default=150, help="Maximal sequence length of bert or xlm model")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--use_ensemble', action="store_true")
    parser.add_argument('--model_dir', default='best_weights')
    parser.add_argument('--model_name', default=None)
    parser.add_argument("--batch_size", default=16, type=int, help="The batch size for training.")

    args = parser.parse_args()

    if args.use_ensemble:
        model_paths = glob.glob(f'{args.model_dir}/*.bin')
    else:
        model_paths = [os.path.join(args.model_dir, args.model_name)]

    try:
        df = pd.read_csv(args.file_path)
    except:
        df = pd.read_csv(args.file_path, lineterminator="\n")
    df['text'] = df['text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
    print(len(model_paths))

    if len(model_paths) == 1:
        pred = ensemble_predict(df, model_paths, args.has_label)
        df['predict_ensemble'] = pred[0]

    else:
        pred_all, pred_ensemble = ensemble_predict(df, model_paths, args.has_label)
        # df['predict'] = [",".join(str(i) for i in x) for x in pred_all]
        df['predict_ensemble'] = pred_ensemble

    if args.has_label:
        f1_score_macro = show_performance(df.label, df.predict_ensemble)

    if not os.path.exists(args.pred_dir):
        os.mkdir(args.pred_dir)
    df.to_csv(os.path.join(args.pred_dir, args.file_path.split('/')[-1]), index=False)





