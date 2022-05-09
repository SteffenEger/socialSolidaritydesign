# Changes in European Solidarity Before and During COVID-19: Evidence from a Large Crowd- and Expert-Annotated Twitter Dataset

Data and code for our paper

```
@inproceedings{ils-etal-2021-changes,
    title = "Changes in {E}uropean Solidarity Before and During {COVID}-19: Evidence from a Large Crowd- and Expert-Annotated {T}witter Dataset",
    author = "Ils, Alexandra  and
      Liu, Dan  and
      Grunow, Daniela  and
      Eger, Steffen",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.129",
    doi = "10.18653/v1/2021.acl-long.129",
    pages = "1623--1637",
}
```


## Getting Started
```python
pip install -r requirements.txt
```

## Dataset
### Train/Dev/Test sets (labeled)
Under the folder data, you can find three dataset splits; each is a different split of 2299 annotations. \
The labels in the dataset mean: 0 is solidarity, 1 is anti-solidarity, two is ambivalent, and 3 is not applicable.

### Test set (unlabeled)
2021-07-01_2022-05-06_no_duplicate.csv:\
includes tweets from 2021-07-01 to 2022-05-06
You can crawl your own tweets using codes in tweets_crawl.

## Train
```python
# train on train set of split 1 using oversampling and translation
python train.py --split 1 --oversample_from_train --translation
```

## Evaluate

```python
# evaluate with a single model (labeled test set)
python predict.py  --file_path data/split3/test.csv   --model_name 'bert oversample_from_train auto_label split3 bert-base-multilingual-cased.bin'  --has_label
```

```python
# evaluate with ensemble models (labeled test set)
python predict.py  --file_path data/split3/test.csv   --model_dir best_weights --has_label --use_ensemble
```

```python
# evaluate with ensemble models (unlabeled test set)
python predict.py  --file_path '2021-07-01_2022-05-06_no_duplicate.csv'   --model_dir best_weights  --use_ensemble
```

