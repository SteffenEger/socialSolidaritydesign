import pandas as pd
import glob
import tweepy
import os
import json
import re
import sys
import os
import argparse
import nest_asyncio
nest_asyncio.apply()
# from . import run
# from . import config
# from . import storage

from GetOldTweets3 import *


hashtags = pd.read_csv('Hashtags.csv',encoding='utf8')
year = '2020'
dir = 'ids'

if not os.path.exists(year):
  os.mkdir(year)

if not os.path.exists(os.path.join(year,'ids')):
  os.mkdir(os.path.join(year,'ids'))

if not os.path.exists(os.path.join(year,'json')):
  os.mkdir(os.path.join(year,'json'))
  
 


'''
since = '2019-09-01'
until = '2020-01-01'

c = config.Config()
c.Since = since
c.Until = until
c.Store_csv = True

exist_hashtags = [item.split('/')[-1][:-4] for item in exist_files]
all_hashtags = list(item.lower().strip() for item in hashtags['Hashtags'])
remining_hashtags = all_hashtags

for hashtag in remining_hashtags:
  print(hashtag)
  c.Output = os.path.join(year, dir, hashtag+'.csv')
  c.Search = hashtag
  run.Search(c)
'''

since = year+'-01-01'
until = year+'-12-21'

c = config.Config()
c.Since = since
c.Until = until
c.Store_csv = True

for hashtag in remining_hashtags:
  print(hashtag)
  c.Output = os.path.join(year, dir, hashtag+'.csv')
  c.Search = hashtag
  run.Search(c)



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


prefix = year+'/'+dir+'/'
exist_files = glob.glob(prefix+'*.csv')

consumerKey = ""
consumerSecret = ""
accessToken = ""
accessTokenSecret = ""
auth = tweepy.AppAuthHandler(consumerKey, consumerSecret)
# auth.set_access_token(the_access_token_key, the_access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

for hashtag in exist_files:
    data = pd.read_csv(hashtag)
    ids = set(data.id)
    print('number of tweets ', len(ids))
    output = os.path.join(year, 'json', hashtag.split('/')[-1][:-4] + ".json")
    count = 0
    ids = list(ids)
    with open(output, 'w') as file:
        for i in chunks(ids, 100):
            try:
                tweets = api.statuses_lookup(i, tweet_mode="extended")
                for tweet in tweets:
                    j = json.dumps(tweet._json)
                    file.write(j)
                    file.write("\n")
                    count += 1
                    if count%100 == 0:
                      print(count)

            except:
                print('error')
    print(f"Crawling for {hashtag} finished, the number of tweets is {count}")
   
