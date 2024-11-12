#!/usr/bin/env python
# coding: utf-8

# In[1]:

import requests
from datetime import datetime, timedelta
from newspaper import Article
import nltk
import logging
import time

logging.basicConfig(level=logging.INFO)
logging.info("Logs start!")
nltk.download('punkt_tab')

# Calculate the start and end dates for this week
end_date = datetime.now()

# Calculate the date for the previous Friday
friday_date = end_date - timedelta(days=end_date.weekday() + 3)  # 3 days before Sunday (weekday = 6)

# Calculate the date for the upcoming Monday
monday_date = end_date - timedelta(days=end_date.weekday())  # This gives you the most recent Monday

logging.info(f"Friday: {friday_date}")
logging.info(f"Monday: {monday_date}")
# print(f"Friday: {friday_date}")
# print(f"Monday: {monday_date}")

base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
params = {
    "query": "domain:motorsport.com (Formula One OR F1)",
    "mode": "ArtList",
    "maxrecords": 250,
    "startdatetime": friday_date.strftime("%Y%m%d000000"),  # Format: YYYYMMDDHHMMSS
    "enddatetime": monday_date.strftime("%Y%m%d235959"),      # Format: YYYYMMDDHHMMSS
    "format": "json"
}
'''
    Using gdelt to get all motorsport articles posted this week
'''
# Try fetching data from GDELT API
try:
    logging.info("Attempting to fetch data from GDELT API...")
    time.sleep(5)
    response = requests.get(base_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        logging.info("Status Code: " + str(response.status_code))
        articles = response.json()
        logging.info(f"Number of articles retrieved: {len(articles.get('articles', []))}")
    else:
        logging.error(f"Failed to fetch articles, Status Code: {response.status_code}")
        logging.error(f"Response Text: {response.text}")
        articles = {}

except requests.exceptions.RequestException as e:
    logging.error("An error occurred while fetching data from the GDELT API.")
    logging.error(e)
    articles = {}
'''
    Now all motorsport articles are collected. filter to get all F1 related articles.
'''
if response.status_code == 200:
    f1Articles = []
    if 'articles' in articles:
        for article in articles['articles']:
            url = article.get('url', '')
            if "motorsport.com/f1/news" in url:
                f1Articles.append(article)
    # print(f1Articles)
    '''
        Use Newspaper3k to download the content/text of each article via the url
    '''
    f1_articles_summaries = []
    for article in f1Articles:
        url = article.get('url', '')
        try:
            currentArticle = Article(url)
            currentArticle.download()
            currentArticle.parse()
            f1_articles_summaries.append(currentArticle.text)
        except:
            logging.info("Issue handling article")
    # print(f1_articles_summaries)
    '''
        Use facebook/bart-large-cnn to summarize EACH article
    '''
    from transformers import BartForConditionalGeneration, BartTokenizer
    
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    def summarize_to_bullets(text):
        inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"], 
                                     max_length=175, 
                                     min_length=50, 
                                     length_penalty=2.0, 
                                     num_beams=4,
                                     no_repeat_ngram_size=6,
                                     early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    summaryResult = []
    
    for text in f1_articles_summaries:
        summary = summarize_to_bullets(text)
        summaryResult.append(summary)
    logging.info('summaryResult:' + str(summaryResult))
    logging.info('------------------------------')
    
    '''
        Filter out F1 articles that dont have keywords
    '''
    filtered = []
    filteredText = ''
    
    def filter_relevant_summary(summary, keywords):
        # Check if any keyword is in the summary
        for keyword in keywords:
            if keyword.lower() in summary.lower():
                return True
        return False  # Return None if no relevant keyword found
    teams = ['Sauber', 'Alpine', 'Aston Martin', 'Aston', 'Ferrari', 'Haas', 'McLaren', 'Mercedes', 'RedBull', 'Red Bull', 'Williams',
             'Visa Cash App RB', 'VCARB','audi']
    
    drivers = ['bottas', 'zhou', 'guanyu', 'esteban', 'ocon', 'pierre', 'gasly', 'fernando', 'alonso', 'lance',
               'stroll', 'charles', 'leclerc', 'carlos', 'sainz', 'kevin', 'magnussen', 'nico', 'hulkenberg', 'lando', 'norris', 'oscar', 'piastri',
               'lewis', 'hamilton', 'george', 'russell', 'max', 'verstappen', 'sergio', 'perez', 'alex', 'albon', 'logan', 'sargeant', 'yuki', 'tsunoda',
               'nyck', 'de vries', 'franco', 'colapinto','Liam','Lawson']
    
    event = ['pit stops', 'qualifying', 'quali', 'practice sessions', 'driver\'s briefing', 'press conference', 'race strategy',
             'safety car', 'red flag', 'yellow flag', 'track condition', 'penalties', 'penalty', 'tyre choice', 'tire choice','steward',
             'race weekend', 'weather forecasts', 'race result', 'championship standings', 'incidents', 'overtake', 'passes', 'pit lane', 'lap times',
             'reprimand', 'adjustment of', 'tech','disqualified','pit']
    
    keywords = teams + drivers + event
    
    for summary in summaryResult:
        chunk = summary.split('.')
        for x in chunk:
            if filter_relevant_summary(x,keywords):
                filtered.append(x)
                filteredText += x + '. '
    logging.info('filtered: ' + filteredText)
    
    '''
        Shorten/summarize the total text to fit twitter's char limit
    '''
    #Shorten to 280

    def shorten(text):
        inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"], 
                                     max_length=55,           # max tokens
                                     min_length=20,           # min tokens
                                     length_penalty=2.0,     # > 1 favors shorter summaries
                                     no_repeat_ngram_size=5,  # Avoid repetition and irrelevant sentences
                                     num_beams=5,             # > 4 improves quality and coherence
                                     early_stopping=True
                                    )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        
    if len(filteredText) > 275:
        short = shorten(filteredText)
    else:
        short = filteredText
    logging.info(short)
    
    '''
        Reformat the shorten summary to be in bullet notation.
    '''
    # Reformat text
    # Split text by period to separate sentences
    sentences = short.split('.')
    
    # Format each sentence with a bullet point
    bullet_points = "\n".join([f"• {sentence.strip()}" for sentence in sentences if sentence.strip()])
    
    logging.info(bullet_points)
    logging.info(len(bullet_points))
    
    '''
        Post to Twitter
    '''
    import requests
    import json
    from requests_oauthlib import OAuth1
    
    API_KEY = 'MRcrFnNBKj5U7YKEa7Msm8Oc1'
    API_SECRET_KEY = 'YMOWRJCcaWfM9CBMv2jZnFi25LJPHdUt2Eb91qesU0Xa7D1eL4'
    ACCESS_TOKEN = '1855362853575782401-YDrWgNT6bDomcXfpeIOIp17JLl0R3C'
    ACCESS_TOKEN_SECRET = 'afZb6osNib8DxQ7PBC8Ox3ZeZec8L6pXAQAvzynIupMAH'
    
    # Replace with your Bearer Token
    # BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAESFwwEAAAAAiGAFXA0JvLG6LoH6V8V%2B26bwK3E%3D3NngaDoCGWSNYpgbjZTPQPks2fLEWdX7Tam6N7tafRkamjmfYp'
    
    auth = OAuth1(API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    
    headers = {
      'content-type': 'application/json',
    };
    
    # Create the tweet content
    tweet_content = {
        "text": f'{bullet_points}'
    }
    
    # API URL for posting a tweet
    url = 'https://api.x.com/2/tweets'
    
    # Make the POST request to the Twitter API
    response = requests.post(url, auth=auth, json=tweet_content, headers=headers)
    
    # Check if the tweet was posted successfully
    if response.status_code == 201:
        logging.info("Tweet posted successfully!")
    else:
        logging.info(f"Error: {response.status_code} - {response.text}")
else:
    logging.info("No articles. GDELT Failed.")





