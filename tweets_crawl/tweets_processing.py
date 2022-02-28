import json
import re
import glob
import csv
import pandas as pd
from datetime import datetime
import numpy as np
from shapely.geometry import Point, Polygon
import geopandas as gpd
from shapely.geometry import Point, Polygon
from countrygroups import EUROPEAN_UNION

map_file = "twitter_locations_from_google.json"
with open(map_file) as f:
    mapped_loc = json.load(f)

hashtags = pd.read_csv('Hashtags.csv')
refugee = list(item[1:].lower().strip() for item in hashtags.Hashtags[:24])
euro = list(
    item.split()[0][1:].lower().strip() if item.startswith('#austerity') or item.startswith('#austerität') else item[
                                                                                                                1:].lower().strip()
    for item in hashtags.Hashtags[24:])


def remove_url(text):
    return re.sub(r'http\S+', '', text)


def get_coordinate(tweet):
    coordinate = None
    try:
        if tweet['coordinates']:
            coordinate = [tweet['coordinates']['coordinates'][0], tweet['coordinates']['coordinates'][1]]
        elif tweet['place']:
            coordinate = [(float(tweet['place']['bounding_box']['coordinates'][0][0][0]) + float(
                tweet['place']['bounding_box']['coordinates'][0][1][0])) / 2, (
                                  float(tweet['place']['bounding_box']['coordinates'][0][0][1]) + float(
                              tweet['place']['bounding_box']['coordinates'][0][3][1])) / 2]
        elif tweet['user']['location']:
            loc = tweet['user']['location']
            if loc in mapped_loc and mapped_loc[loc]:
                coordinate = [mapped_loc[loc]['geometry']['location']['lng'],
                              mapped_loc[loc]['geometry']['location']['lat']]
    except:
        coordinate = None

    return coordinate


def convert_time(time):
    time = datetime.strptime(time, "%a %b %d %H:%M:%S +0000 %Y")
    time = datetime.strftime(time, "%Y-%m-%d %H:%M:%S +0000")
    return time


# -1: not in our topic
# 0: both
# 1: refugee
# 2: euro
def extract_category(tweet):
    output = []
    hashtags = [tweet['entities']['hashtags'][i]['text'].lower() for i in range(len(tweet['entities']['hashtags']))]
    for hashtag in hashtags:
        if hashtag in refugee:
            output.append('refugee')
        if hashtag in euro:
            output.append('euro')
    output = set(output)
    if len(output) == 2:
        return 0
    elif 'refugee' in output:
        return 1
    elif 'euro' in output:
        return 2
    else:
        return -1


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
countries = []
countries.append('Germany')
countries.append('United Kingdom')
countries.extend(list(set(list(world[world.continent == 'Europe'].name)) - set(['United Kingdom', 'Germany'])))

year = '2020'
dir = 'json'
json_files = glob.glob(year + '/' + dir + '/*.json')
tweets = []
for file in json_files:
    with open(file) as file:
        for line in file:
            tweets.append(json.loads(line))
print(len(tweets))
with open(year + '.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'date', 'text', 'lang', 'country', 'category'])
    count = 0
    for i, tweet in enumerate(tweets):
        lang = tweet['lang']
        if lang == 'de' or lang == 'en':
            coordinate = get_coordinate(tweet)
            region = None
            if coordinate:
                for country in countries:
                    ger = world[world.name == country]
                    num = int(np.where(np.array(world.name == country))[0])
                    if Point(coordinate[0], coordinate[1]).within(ger.loc[num, 'geometry']):
                        count += 1
                        region = country
                        print(i, count, country)
                        break
            category = extract_category(tweet)
            if (category != -1) and ((lang == 'de') or (region!=None)):
                date = convert_time(tweet['created_at'])
                text = remove_url(tweet['full_text'])
                id = tweet['id']
                writer.writerow([id, date, text, lang, region, category])




data = pd.read_csv(f'{year}.csv')
data = data.drop_duplicates()
data.to_csv(f'{year}_v.csv', index=False)
