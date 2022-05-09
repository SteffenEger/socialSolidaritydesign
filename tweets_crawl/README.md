# Instructions for Crawling Tweets
## Getting Started
```python
>>> pip install -r requirements.txt
```
## Crawling Tweets
```python
# obtain consumerKey, consumerSecret, accessToken and accessTokenSecret from https://developer.twitter.com/en/docs/twitter-api
python tweets_crawl.py --since 2021-07-01 --until 2022-05-06  --consumerKey [CONSUMERKEY] \
                                                              --consumerSecret [CONSUMERSECRET] \
                                                              --accessToken [ACCESSTOKEN]\
                                                              --accessTokenSecret [ACCESSTOKENSECRET]
```
## Processing Tweets
```python
# filtering out non-English or non-german or non-European tweets 
python tweets_processing.py --since 2021-07-01 --until 2022-05-06 
```

# output:
```
.
├── 2021-07-01_2022-05-06
│        └── ids
│        │    └──#asylumseeker.csv: includes the tweet ids and texts of all tweets in the period containing #asylumseeker  
│        │    └── ...
│        └── jsons
│             └──#asylumseekers.json: includes all info of all tweets in the period containing #asylumseekers
│             └── ...
│── 2021-07-01_2022-05-06.csv: includes six attributes of each tweet, namely id, text, language, country, and category, \
│                              where category = 1  means refugee-related, 2 means euro crisis related, 0 related to both;
│                               lang = 'en' or 'de';
│                              country = one of 39 European countries (from https://geopandas.org/en/stable/) or None
│                        
│── 2021-07-01_2022-05-06_no_duplicate.csv: includes tweets with unique tweet ids

```
