from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
import torch
from transformers import BertModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.models import BertForSequenceClassification, XLMRobertaForSequenceClassification
from transformers import RobertaForSequenceClassification
from utils.models import bertCNN, bertDPCNN
from utils.data_processor import convert_data_into_features
from utils.models import BertForSequenceClassification
from train import  show_performance
import time 
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

MAX_LEN = 150
BATCH_SIZE = 20


def predict(model, data_loader):
    model.eval()

    predictions = []

    for _, data in enumerate(data_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        with torch.no_grad():
            
            outputs = model(ids, mask, token_type_ids)
            big_val, big_idx = torch.max(outputs[0].data, dim=1)
            predictions.extend(big_idx.tolist())
    
    return predictions


def ensemble_model_pred(model_path_ls, df, label = True, merge = True):
    """
    run prediction for one model or a list of models, and ensemble the results
    :param model_path_ls: a list of model path
    :param df: data to be predicted
    :return: predictions for each model and their ensemble results(majority vote)
    """
    pred = []
    for path in model_path_ls:
        print(path)
        if path.startswith('bert'):
            
            pretrained_weights = 'fine_tune/finetuned_lm_180k'
        else:
            pretrained_weights = 'fine_tune/xlm-r_finetune_mlm_100k'

        tokenizer = AutoTokenizer.from_pretrained(pretrained_weights, do_lower_case=args.do_lower_case)
          
        
        model =AutoModelForSequenceClassification.from_pretrained(pretrained_weights, num_labels=args.num_labels)
       
            

        model.to(device)
        model.load_state_dict(torch.load(args.model_dir+'/'+path, map_location=torch.device('cpu')))
        data_loader = convert_data_into_features(df, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE, shuffle=False, label = label, merge = merge)
        start = time.time()
        predictions = predict(model, data_loader)
        end = time.time()
        print(f'processing time for one model is {end-start}')
        pred.append(predictions)

    if len(pred) == 1:  # prediction for single model, no ensemble
        return pred
    else:
        # reverse row and colum
        
        pred = list(map(list, zip(*pred)))
        # get mode for each row (every instance)
        pred_ensemble = [np.argmax(np.bincount(x)) for x in pred]
        
        return pred, pred_ensemble


def run(model_names, read_dir, write_dir, label = False, merge = True):
    print(read_dir)
    if label:
        files = [f for f in os.listdir(read_dir) if f.endswith("dev.csv") or f.endswith("test.csv")]
    else:
        files = [f for f in os.listdir(read_dir) if f.endswith(".csv")]
    print(files)
    
    for file in files:
        f_name = os.path.join(read_dir,file)
        print(f_name)
        try:
            df = pd.read_csv(f_name)
        except:
            df = pd.read_csv(f_name, lineterminator="\n")
        print(df)
        if len(model_names) == 1:
            predictions = ensemble_model_pred(model_names, df, label)
            print(len(df), len(predictions[0]))
            df['predict'] = predictions[0]
            df = df[['id', 'date', 'text','predict']]
        else:
           
            predictions, predictions_ensemble = ensemble_model_pred(model_names, df, label, merge)
            print(len(df), len(predictions_ensemble))
            if label:
                if 'test' in file:
                    print('+++++++test+++++++')
                if 'dev' in file: 
                    print('+++++++evaluation+++++++')
                f1_score_macro = show_performance(df.label, predictions_ensemble)
            df['predict'] = [",".join(str(i) for i in x) for x in predictions]
            df['predict_ensemble'] = predictions_ensemble
            #df = df[['id', 'date','text','predict', 'predict_ensemble']]
            
           
        print(df)

        df.to_csv(os.path.join(write_dir, file), index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,default='saved_weights/', help="Directory which saves models")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model, for single model prediction")
    parser.add_argument("--data_dir", type=str, default='new_output',
                        help="Directory which saves data to be predicted")
    parser.add_argument("--output_dir", type=str, default='new_output/output', help="Path to save the prediction results")
    parser.add_argument("--num_labels", type=int, default=3, help="The number of classes for classification")
    parser.add_argument('--do_lower_case', action="store_true", help="Whether lowercase text before tokenization")

    parser.add_argument('--label', type=bool, help="Whether we have labels, i.e. in a test mode, the datasets can be test/dev set")
    parser.add_argument('--do_merge', type=bool, default = True, help="When we have labels and we want to merge class 2 and class 3")
   
    args = parser.parse_args()
    print(f'data_dir={args.data_dir}, output_dir={args.output_dir}')
    if args.model_name is not None:
        model_path_ls = [args.model_name]
    else:
        model_path_ls = os.listdir(args.model_dir)
    print(f'model paths: {model_path_ls}')
   
    for path in model_path_ls:
        if path.startswith('xlm') or path.startswith('bert'):
            
            continue
        else:
            raise ValueError(
                f'Error model name {path}! Model name must start with the following 4 types: bert, xlm, bert_cnn or bert_dpcnn')
   
    
    run(model_path_ls, args.data_dir, args.output_dir, args.label, args.do_merge)
