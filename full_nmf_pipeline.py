#Imports
import numpy as np
import sklearn
import websockets
import yfinance as yf

from sklearn.decomposition import NMF
from bs4 import BeautifulSoup
import requests
import pandas as pd
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

#function signatures
from scipy.sparse import csr_matrix
from typing import Sequence, Tuple


#Used for stemming
porter = nltk.PorterStemmer()

#Globals
API_key = '8kzbjhg9e2mhh3obkrr6q5rrnxppzsspkl2qo0hr'
base_stockAPI_url = 'https://stocknewsapi.com/api/v1'
base_yf_url = 'https://finance.yahoo.com/'

#Functions for word processing

def ready_for_nltk():

    global stop_words
    
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))


    try:
        #attempting to directly tokenize article
        word_tokenize(" ")
    
    except LookupError:
        #runs when the 'punkt' tokenizer isn't installed
        print("NLTK tokenizer not found and we must download 'punkt' and 'punkt_tab'")

        nltk.download('punkt')

    try:
        nltk.download('punkt_tab')
    except Exception:
        #pass if punkt_tab doesn't exist
        pass

#Bag of Words approach
def tokenize(text: str) -> list[str]:
    
    text = text.lower()
    tokenized_article = word_tokenize(text)
    
    return tokenized_article

#Removal of stopwords
def trimmed_of_stopwords(tokenized_article:list[str]) -> list[str]:
    trimmed_article = [i for i in tokenized_article if i not in stop_words]
    return trimmed_article

#Porter Stemmer
def stem_words(trimmed_stopwords: list[str]) -> list[str]:
    porter_stem_list = [porter.stem(i) for i in trimmed_stopwords]
    return porter_stem_list

#Non-negative matric factorization, prints topics and most appearing words
def apply_nmf_topic_models(prep_nmf : csr_matrix, possible_ncomp : int, index_map : np.ndarray) ->np.ndarray:
    nmf = NMF(n_components = possible_ncomp, solver='mu', max_iter = 500) #shape (d x n)
    H = nmf.fit_transform(prep_nmf)
    W = nmf.components_

    #taking transpose to be consistent with math conventions for matrix dimensions
    #W_conv = W.T #shape (d x r)
    #H_conv = H.T #shape (r x n)

    #Print topics, topic->rows of W,
    for topic_index, topic in enumerate(W):
        top_indicies = topic.argsort()[-10:][::-1] #shows largest first
        top_words = index_map[top_indicies]
        print(f'Topic {topic_index + 1}: {','.join(top_words)}')
    return H

#Full word processing pipeline, prepares for NMF
def full_pipeline(url_series : Sequence[str], max_feat : int, min_df : int) -> Tuple[csr_matrix, list[str], TfidfVectorizer, np.ndarray]:
    
    corpus = []
    for url in url_series:
        raw = get_text_requests(url)
        
        #Protect against failed urls in dataframe
        if raw is None:
            continue
        
        tokens = tokenize(raw)
        remove_stopwords = trimmed_of_stopwords(tokens)
        stem = stem_words(remove_stopwords)
        doc_str = ' '.join(stem)
        corpus.append(doc_str)

    vectorizer = TfidfVectorizer(max_features= max_feat, min_df = min_df, stop_words=None)
    prep_NMF = vectorizer.fit_transform(corpus)
    index_map = np.array(vectorizer.get_feature_names_out())

    return prep_NMF, corpus, vectorizer, index_map


#Web Scraping Functions for yahoo finance

#For yahoo Finance
def scrape_articles_yf(url: str, n: int) -> pd.DataFrame:

    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
    response = requests.get(url, headers=headers)
    response.raise_for_status() #Error if failed request

    soup = BeautifulSoup(response.text, 'html.parser')

    web_links = []
    domain_base = urlparse(url).netloc
    
    for a in soup.find_all('a', href=True):
        article_title = a.get_text(strip=True)

        joined_url = urljoin(url, a['href'])
        
        #link filters
        if domain_base not in joined_url:
            continue

        if '/news/' not in joined_url:
            continue

        if len(article_title) < 12:
            continue

        discard = {'news', 'financial news', 'newsletters'}
        if article_title.lower() in discard:
            continue

        web_links.append((article_title, joined_url))

    df = pd.DataFrame(web_links[:n], columns =['Article Title', 'url'])

    return df
    

#Using requests library and BeautifulSoup, function fetches main article text from a webpage
def get_text_requests(url):
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)

    #For api, to avoid articles that take too long to render
    try:
        response = requests.get(url,headers=headers, timeout=3)
    #skip if timesout
    except requests.exceptions.RequestException:
        return None

    #for API, to avoid crashing whole run
    if response.status_code in (401, 403, 404, 500):
        return None

    #response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    paragraphs = [par.get_text() for par in soup.find_all('p')]

    page = '\n'.join(paragraphs)

    return page

#For using get_text_selenium(...) function, allows me to scrape articles that require java script to be rendered
def configure_headless_browser():
    # Configure headless Chrome
    headless_config = Options()
    headless_config.add_argument("--headless=new")   # run without opening a window
    #turn off the sandbox feature of chrome, without this line chrome will not run
    headless_config.add_argument("--no-sandbox")  

    non_suspicious_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
    headless_config.add_argument(f'--user-agent={non_suspicious_agent}')

    #Create Selenium Chrome driver and automatically download proper ChromeDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=headless_config)

    return driver


#scrapes articles using selenium library
def get_text_selenium(url: str) -> str:

    driver = configure_headless_browser()

    try:
        # Load page
        driver.get(url)
    
        # Wait for JS to run
        time.sleep(2)

        # Get the fully rendered HTML
        html = driver.page_source

        # Parse with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        
        return "\n".join(paragraphs)

    finally:
        #Close chrome no matter what
        driver.quit()


#For stockNews API consumption
def news_todate(year, ticker, end_date=None):
    start_date = datetime(year, 1, 1)

    if end_date is None:
        end_date = datetime(year, 12, 31)
    else:
        if not isinstance(end_date, str) or len(end_date) != 8:
            raise ValueError('End date must have form "MMDDYYYY" no spaces. E.g. January 1, 2025 = 01012025')

        end_date = datetime.strptime(end_date, '%m%d%Y')

    #For stock newsAPI date range
    date_range = f'{start_date:%m%d%Y}-{end_date:%m%d%Y}'

    max_page_requests = 10 #From API
    rows = []
    page = 1 #50 requests per page i

    while page <= max_page_requests:
        print(f"Fetching page {page} ...")

        parameters = {'token' : API_key, 'tickers': ticker, 'items': 50, 'type': 'article', 'sortby': 'rank', 'page': page, 'date' : date_range}
        
        resp = requests.get(base_stockAPI_url, params = parameters)

        resp.raise_for_status() #Will throw error if not successful
        full_JSON_resp = resp.json()
        data = full_JSON_resp.get('data', []) #in case data is empty

        #No more articles on the page
        if not data:
            break
            
        #add articles to list
        rows.extend(data)
        
        page+=1

        #No articles for date range
    if not rows:
        return pd.DataFrame(columns=['title', 'news_url', 'date', 'sentiment'])
        
    df = pd.DataFrame(rows)[['title', 'news_url', 'date', 'sentiment']]
    df['ticker'] = ticker
    df['date'] = pd.to_datetime(df['date'])
        
    return df

def url_failed(url):
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    #For api, to avoid articles that take too long to render
    try:
        response = requests.get(url,headers=headers, timeout=3)
    #skip if timesout
    except requests.exceptions.RequestException:
        return 'FAILED'

    #for API, to avoid crashing whole run
    if response.status_code in (401, 403, 404, 500):
        return 'FAILED'
    return url