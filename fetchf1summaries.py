#!/usr/bin/env python
# coding: utf-8

import json
import os
import certifi
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nltk
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from gdeltdoc import GdeltDoc, Filters
from gdeltdoc.errors import RateLimitError

# macOS python.org builds often lack system CA bundle for urllib; NLTK downloads need this
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

logging.basicConfig(level=logging.INFO)
logging.info("Logs start!")
try:
    nltk.download("punkt_tab", quiet=True)
except Exception as e:
    logging.warning("NLTK punkt_tab download failed (tokenization may break): %s", e)

# Motorsport.com returns 403 to newspaper's default client; real browser headers get 200.
SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.motorsport.com/f1/",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Upgrade-Insecure-Requests": "1",
}


def build_scrape_session():
    s = requests.Session()
    s.headers.update(SCRAPE_HEADERS)
    s.verify = certifi.where()
    return s


def fetch_page_html(session, url, timeout=15):
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    return r.text


def article_text_fallback(html):
    """If newspaper returns empty body, pull plain text from common article roots."""
    soup = BeautifulSoup(html, "lxml")
    root = soup.find("article")
    if root is None:
        root = soup.find("main")
    if root is None:
        return ""
    for tag in root.find_all(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()
    text = root.get_text(separator="\n", strip=True)
    return "\n".join(line for line in (ln.strip() for ln in text.splitlines()) if line)

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
'''
    Using gdelt to get all motorsport articles posted this week
'''
startDate = friday_date.strftime("%Y-%m-%d")  # Format previous Friday's date
endDate = monday_date.strftime("%Y-%m-%d")    # Format monday date

f = Filters(
    start_date = startDate,
    end_date = endDate,
    domain = "motorsport.com",
    num_records = 250,
)

gd = GdeltDoc()
# Search for articles matching the filters

def fetch_articles_with_retry(gd, f, max_retries=5):
    for attempt in range(max_retries):
        try:
            return gd.article_search(f)
        except RateLimitError:
            wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s...
            print(f"Rate limited. Retry {attempt+1}/{max_retries} in {wait}s...")
            time.sleep(wait)

    print("Failed after retries. Returning empty list.")
    return []

articles = fetch_articles_with_retry(gd, f)
'''
    Now all motorsport articles are collected. filter to get all F1 related articles.
'''
if not articles.empty:
    f1Articles = []
    for row in articles.itertuples():
        if "motorsport.com/f1/news" in row.url:
            f1Articles.append(row.url)
    print("Total Articles:", len(f1Articles))
    # print(f1Articles)
    '''
        Fetch HTML with browser-like headers, then parse with newspaper3k (avoids 403 on download).
    '''
    f1_articles_summaries = []
    scrape_session = build_scrape_session()
    for article_url in f1Articles:
        print(f"Processing URL: {article_url}")
        try:
            html = fetch_page_html(scrape_session, article_url)
            currentArticle = Article(article_url, request_timeout=15)
            currentArticle.download(input_html=html)
            currentArticle.parse()
            body = (currentArticle.text or "").strip()
            if not body:
                body = article_text_fallback(html)
            if body:
                f1_articles_summaries.append(body)
            else:
                logging.warning("Empty article text after parse + fallback: %s", article_url)
        except requests.RequestException as e:
            logging.warning("HTTP error for %s: %s", article_url, e)
        except Exception as e:
            logging.warning("Issue handling article %s: %s", article_url, e)
        time.sleep(0.4)

    print("Total Scraped Articles:", len(f1_articles_summaries))

    if not f1_articles_summaries:
        logging.info("No article bodies scraped; skipping summarization and Twitter post.")
    else:
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
                                         forced_bos_token_id=0,
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
                                         max_length=50,           # max tokens
                                         min_length=10,           # min tokens
                                         length_penalty=2.5,     # > 1 favors shorter summaries
                                         no_repeat_ngram_size=5,  # Avoid repetition and irrelevant sentences
                                         num_beams=6,             # > 4 improves quality and coherence
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

        bullet_items = [s.strip() for s in sentences if s.strip()]
        output_path = Path(__file__).resolve().parent / "f1_summary_output.json"
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gdelt": {
                "start_date": startDate,
                "end_date": endDate,
                "domain": "motorsport.com",
            },
            "f1_article_urls": f1Articles,
            "scraped_article_count": len(f1_articles_summaries),
            "per_article_summaries": summaryResult,
            "filtered_text": filteredText,
            "short_summary": short,
            "bullet_items": bullet_items,
            "bullet_markdown": bullet_points,
        }
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logging.info("Wrote results to %s", output_path)
else:
    logging.info("No articles. GDELT Failed.")





