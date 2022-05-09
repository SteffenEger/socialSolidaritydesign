import pandas as pd
import glob
import tweepy
import os
import sys
import argparse
import snscrape.modules.twitter as sntwitter
from tweepy.auth import OAuthHandler
import json


class SNS_Crawler():
    def __init__(self,since, until, save_got_dir, hashtag_path):
        self.since = since
        self.until = until
        self.save_got_dir = save_got_dir
        self.hashtags = pd.read_csv(hashtag_path,encoding='utf8')
        print(f'The number of Hashtags is {len(self.hashtags)}.')
        
    def crawl(self):
        save_dir = self.since + '_' + self.until
        if not os.path.exists(f'{save_dir}/{self.save_got_dir}'):
            os.makedirs(f'{save_dir}/{self.save_got_dir}')
        
        all_hashtags = list(item.lower().strip() for item in self.hashtags['Hashtags'])
        
        for hashtag in all_hashtags:
          output_path = os.path.join(save_dir, self.save_got_dir, hashtag+'.csv')
          tweets = []
          query = f"{hashtag} since:{self.since} until:{self.until}"
          for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                tweets.append([tweet.id, tweet.content])
          tweets = pd.DataFrame(tweets, columns=['Tweet Id', 'Text'])
          tweets.to_csv(output_path, index = False)
          print(f'{hashtag} Crawled Finished. {len(tweets)} in total')
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield [str(item) for item in lst[i:i + n]]

class Tweepy_Crawler():
    def __init__(self, consumerKey, consumerSecret, accessToken, accessTokenSecret, since, until, save_got_dir, save_tweepy_dir):
        
        
        self.auth = OAuthHandler(consumerKey, consumerSecret)
        self.auth.set_access_token(accessToken, accessTokenSecret)
        
        self.since = since
        self.until = until 
        self.save_got_dir = save_got_dir
        self.save_tweepy_dir = save_tweepy_dir
        
    def crawl(self):
        api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        save_dir = self.since + '_' + self.until
        hashtag_files = glob.glob(f'{save_dir}/{self.save_got_dir}/*.csv')
        for file in hashtag_files:
            df = pd.read_csv(file, lineterminator="\n")
            ids = set(df['Tweet Id'])
            print(f'Hashtag {file.split("/")[-1][:-4]} Number of Tweets {len(ids)}')
            if not os.path.exists(f'{save_dir}/{self.save_tweepy_dir}'):
                os.makedirs(f'{save_dir}/{self.save_tweepy_dir}')
            output = os.path.join(save_dir, self.save_tweepy_dir, file.split('/')[-1][:-4] + ".json")
            count = 0
            with open(output, 'w') as file:
                for i in chunks(list(ids), 100):
                    try:
                        tweets = api.statuses_lookup(i, tweet_mode="extended")
                
                        for tweet in tweets:
                            j = json.dumps(tweet._json)
                            file.write(j)
                            file.write("\n")
                            count += 1
                        print(len(tweets))
                    except:
                        print('HTTP ERROR')
            print(f'{count} Tweets crawled')
                    
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--since', required = True)
    parser.add_argument('--until', required = True)
    parser.add_argument('--hashtag_path', type = str, default = 'Hashtags.csv')
    parser.add_argument('--save_got_dir', type = str, default = 'ids')
    parser.add_argument('--save_tweepy_dir', type = str, default = 'jsons')
    parser.add_argument('--consumerKey', type = str, required = True)
    parser.add_argument('--consumerSecret', type = str, required = True)
    parser.add_argument('--accessToken', type = str, required = True)
    parser.add_argument('--accessTokenSecret', type = str, required = True)
    args = parser.parse_args()
    
    crawler = SNS_Crawler(args.since, args.until, args.save_got_dir, args.hashtag_path)
    crawler.crawl()
    crawler = Tweepy_Crawler(args.consumerKey, args.consumerSecret, args.accessToken, args.accessTokenSecret, args.since, args.until, args.save_got_dir, args.save_tweepy_dir)
    crawler.crawl()