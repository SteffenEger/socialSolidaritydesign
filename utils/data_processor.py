from torch.utils.data import DataLoader
from utils.custom_dataset import CustomDataset
import pandas as pd
import os
import datetime
import re
import glob

def get_datetime_from_string(datetime_string, new_format=""):
    """
    convert Twitter API date format to python datetime format
    """
    dt_obj = datetime.datetime.strptime(datetime_string, '%a %b %d %H:%M:%S %z %Y')
    if new_format == "":
        return dt_obj

    return dt_obj.strftime(new_format)


def merge(label):
    """merge ambivalent and non-applicable labels (2, 3 -> 2)"""
    if label <= 2:
        return label
    else:
        return 2

def label_process(data, do_merge = True):
    #data['text'] = data['text'].apply(lambda x: process_tweet(x))
    if do_merge:
        data['label'] = data['label'].apply(lambda x: merge(x))
    return data


def load_translation(path, do_merge=False):
    """ function that reads translated texts and the corresponding labels """
    data = pd.read_csv(path)
    if do_merge:
        data['label'] = data['label'].apply(lambda x: merge(x))
    data = data[['text', 'label']]
    # print(data.head())
    #data = data.rename(columns={'translation': 'text'})  # rename column
    # print(data.head())
    return data


def oversampling(df):
    """
    function that randomly selects samples from minority classes until reaching the same number with the majority class
    :param df: original dataset
    :param df_trans: translated data. If it's not None, sample from it. Otherwise, sample from the original dataset
    :return:
    """
    max_size = df['label'].value_counts().max()  # the number of items in the majority class
    ls = [df]
    
    print('sample from the original dataset...')
    for class_index, group in df.groupby('label'):
        ls.append(group.sample(max_size - len(group), replace=True))
    # concat the original data and all the selected samples, so all the classes have the same number of items
    df_new = pd.concat(ls)
    df_new = df_new.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    
    return df_new


def convert_data_into_features(data, tokenizer, max_len, batch_size, shuffle=True, has_label = True, merge = True):
    if has_label and merge:
        data = label_process(data, merge)
    dataset = CustomDataset(data, tokenizer, max_len, has_label)
    data_loader= DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    return data_loader

def load_df(path, do_merge=False):
    """
    function that reads tweet texts and labels from a given path
    :param path: path of a csv. file
    :param do_merge: whether merge ambivalent and non-applicable classes
    :return: a dataframe contains desired data
    """

    df = pd.read_csv(path)
    df['text'] = df['text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
    print(df.iloc[0])

    if do_merge:
        df['label'] = df['label'].apply(lambda x: merge(x))

    return df


def build_dataloader(src_path, split, do_merge, tokenizer, max_len, batch_size,\
                     oversample_from_train=False,\
                     translation=False,\
                     auto_data=False):
    # Load dataset
    df_train = load_df(os.path.join(src_path, 'split'+str(split), 'train.csv'), do_merge)
    
    if auto_data:
        print("add auto data ...")
        df_extra = load_df(os.path.join(src_path, 'auto_labeled_35000.csv'))
        df_train = pd.concat([df_train, df_extra])

    # add translation
    if translation:
        print("add translated data ...")
        df_trans = load_translation(os.path.join(src_path, 'split'+str(split), f'train_translated.csv'), do_merge=do_merge)  # translated data
        df_train = pd.concat([df_train, df_trans])
        df_train = df_train.reset_index(drop = True)
    
    # oversampling
    if oversample_from_train:
        df_train = oversampling(df_train)  # sample from original data

    df_dev = load_df(os.path.join(src_path, 'split'+str(split),'dev.csv'), do_merge)
    df_test = load_df(os.path.join(src_path,'split'+str(split), 'test.csv'), do_merge)
    

    print(len(df_train), len(df_dev), len(df_test))
    print(df_train.loc[:, 'label'].value_counts())
    
    train_loader=convert_data_into_features(df_train,tokenizer,max_len, batch_size)
    dev_loader=convert_data_into_features(df_dev,tokenizer,max_len, batch_size)
    test_loader = convert_data_into_features(df_test, tokenizer, max_len, batch_size)

    return train_loader, dev_loader, test_loader
