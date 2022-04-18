import datetime
import sqlite3 as sql 
import tweepy
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')

def db_init(con):
    cur = con.cursor()
    cur.execute("""CREATE TABLE tweets (user_id INTEGER, tweet_id INTEGER, query_idx INTEGER, urls BLOB, text TEXT, users BLOB, retweet TEXT, rt_user TEXT, hash_tags text, pull_time TEXT)""")
    cur.execute("""CREATE TABLE queries (query_idx INTEGER PRIMARY KEY AUTOINCREMENT, query BLOB, First_ran TEXT, Last_ran TEXT)""")
    cur.execute("""CREATE TABLE users (user_id INTEGER PRIMARY KEY, name TEXT, username TEXT, most_similar INTEGER, capture_count INTEGER DEFAULT 1)""")
    cur.execute("""CREATE TABLE tweet_setiment (tweet_id INTEGER , tokens BLOB, setiment_neg REAL, setiment_neu REAL, setiment_pos REAL, setiment_comp REAL)""")
    cur.execute("""CREATE TABLE query_setiment (query_idx INTEGER, pull_time TEXT, tokens BLOB, setiment_neg REAL, setiment_neu REAL, setiment_pos REAL, setiment_comp REAL, average_similarity REAL, rt_included TEXT)""")
    con.commit()
    

def add_query(twitter_query, con):
    cur = con.cursor()
    time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    cur.execute(f"""
        INSERT INTO queries (query, First_ran) VALUES ('{twitter_query}','{time}')
    """)
    con.commit()
    


    
def get_queried_tweets(con, client, querydf_row, run_time):
    cur = con.cursor()
    query_idx = querydf_row['query_idx']
    query = querydf_row['query']
    if querydf_row['Last_ran'] is None:
        start_time = querydf_row['First_ran']
    else:
        start_time = querydf_row['Last_ran']
    res = client.search_recent_tweets(query=query, start_time=start_time ,expansions=["author_id"] )
    cur.execute(f"""UPDATE queries
                    SET Last_ran = '{run_time}'
                    WHERE query_idx = {query_idx}""")
    con.commit()
    
    for i in range(len(res.includes['users'])):
        try:
            user_id = res.includes['users'][i].id
            public_name = res.includes['users'][i].name
            user_name = res.includes['users'][i].username
            tweet_id = res.data[i].id
            tweet_text = res.data[i].text
            tweet_text =tweet_text.replace('\n', ' ').replace('"',"'")
            retweet = False
            rt_user = None
            retweet_str = re.findall(r"RT @\S+", tweet_text)
            if retweet_str:
                retweet = True 
                rt_user = re.findall(r"@\S+",retweet_str[0].replace(":",""))
            users = re.findall(r"@\S+", tweet_text)
            if retweet_str:
                rtu = retweet_str[0].replace('RT ','')
                users.remove(rtu)
            urls = re.findall(r'https?:\S+', tweet_text)
            hash_tags = re.findall(r'#\S+', tweet_text)
            sql_query_text = f'''INSERT INTO tweets (user_id, tweet_id, query_idx, urls, text, users, retweet, rt_user, hash_tags, pull_time) 
            VALUES ({user_id}, {tweet_id}, {query_idx}, "{urls}", "{tweet_text}", "{users}", "{retweet}", "{rt_user}", "{hash_tags}", "{run_time}")'''
            sql_query_user = f'''INSERT INTO users (user_id , name , username)
            VALUES ("{user_id}","{public_name}","{user_name}")
            ON CONFLICT(user_id) DO UPDATE SET capture_count=capture_count+1
            '''
        
            cur.execute(sql_query_user)
            cur.execute(sql_query_text)
            con.commit()
        except Exception as e:
            print(e)

    

def init_scrap(con, client, query=None):
    output_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    if query:
        df = pd.read_sql_query(f"SELECT * FROM queries WHERE query_idx = {query}",con)   
    else:
        df = pd.read_sql_query("SELECT * FROM queries",con)   
    for idx, row in df.iterrows():
        get_queried_tweets(con, client, row, output_date)
    


pos_to_lemmatize={'NN':'n','NNS':'n','NNP':'n','NPPS':'n','WP':'n','WP$':'n',
                 'VB':'v','VBD':'v','VBG':'v','VBN':'v','VBP':'v','VBZ':'v',
                 'JJ':'a','JJR':'a','JJS':'a',
                 'RB':'r','RBR':'r','RBS':'r','WRB':'r'}

def processed_feature(text):
    # Removing URLS
    processed_feature = re.sub(r'https?:\S+', '', text)
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', processed_feature)
    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    # Converting to Lowercase remove RT for retweets
    processed_feature = processed_feature.lower().replace('RT ','')
    return processed_feature

def create_bag_of_words(text, lemmatize=True):
    tokenized_word = nltk.tokenize.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words("english"))
    filtered_sent=[]
    for w in tokenized_word:
        if w not in stop_words:
            filtered_sent.append(w)
    pos_taged = nltk.pos_tag(filtered_sent)
    ps  = nltk.stem.PorterStemmer()
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    bag_of_words = []
    if lemmatize:
        for tag in pos_taged:
            if tag[1] in pos_to_lemmatize:
                bag_of_words.append(lem.lemmatize(tag[0],pos_to_lemmatize[tag[1]]))
            else:
                bag_of_words.append(ps.stem(tag[0]))
    else:
        for w in filtered_sent:
            bag_of_words.append(ps.stem(w))
    return bag_of_words

def vadar(text):
    out_put={'neg':None,
             'neu':None,
             'pos':None,
             'Comp':None}
    sia = SentimentIntensityAnalyzer()
    out_put['neg'] = sia.polarity_scores(text)['neg']
    out_put['neu'] = sia.polarity_scores(text)['neu']
    out_put['pos'] = sia.polarity_scores(text)['pos']
    out_put['Comp'] = sia.polarity_scores(text)['compound']
    return out_put

def cos_similarity(textlist):
    TfidfVec = TfidfVectorizer()
    tfidf = TfidfVec.fit_transform(textlist)
    return (tfidf * tfidf.T).toarray()

# n for noun files, v for verb files, a for adjective files, r for adverb
pos_to_lemmatize={'NN':'n','NNS':'n','NNP':'n','NPPS':'n','WP':'n','WP$':'n',
                 'VB':'v','VBD':'v','VBG':'v','VBN':'v','VBP':'v','VBZ':'v',
                 'JJ':'a','JJR':'a','JJS':'a',
                 'RB':'r','RBR':'r','RBS':'r','WRB':'r'}

def processed_feature(text):
    # Removing URLS
    processed_feature = re.sub(r'https?:\S+', '', text)
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', processed_feature)
    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    # Converting to Lowercase remove RT for retweets
    processed_feature = processed_feature.lower().replace('RT ','')
    return processed_feature

def create_bag_of_words(text, lemmatize=True):
    tokenized_word = nltk.tokenize.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words("english"))
    filtered_sent=[]
    for w in tokenized_word:
        if w not in stop_words:
            filtered_sent.append(w)
    pos_taged = nltk.pos_tag(filtered_sent)
    ps  = nltk.stem.PorterStemmer()
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    bag_of_words = []
    if lemmatize:
        for tag in pos_taged:
            if tag[1] in pos_to_lemmatize:
                bag_of_words.append(lem.lemmatize(tag[0],pos_to_lemmatize[tag[1]]))
            else:
                bag_of_words.append(ps.stem(tag[0]))
    else:
        for w in filtered_sent:
            bag_of_words.append(ps.stem(w))
    return bag_of_words

def vadar(text):
    out_put={'neg':None,
             'neu':None,
             'pos':None,
             'comp':None}
    sia = SentimentIntensityAnalyzer()
    out_put['neg'] = sia.polarity_scores(text)['neg']
    out_put['neu'] = sia.polarity_scores(text)['neu']
    out_put['pos'] = sia.polarity_scores(text)['pos']
    out_put['comp'] = sia.polarity_scores(text)['compound']
    return out_put

def cos_similarity(textlist):
    TfidfVec = TfidfVectorizer()
    tfidf = TfidfVec.fit_transform(textlist)
    return (tfidf * tfidf.T).toarray()

def setiment(con, inc_rt=False):
    cur = con.cursor()
    _proc = pd.read_sql_query("SELECT tweet_id FROM tweet_setiment",con)
    _proc = _proc['tweet_id'].tolist()
    _proc = ','.join([str(i) for i in _proc])
    if _proc:
        if not inc_rt:
            df = pd.read_sql_query(f'SELECT * FROM tweets WHERE retweet = "False" AND tweet_id NOT IN ({_proc})',con)
        else:
            df = pd.read_sql_query(f'SELECT * FROM tweets WHERE tweet_id NOT IN ({_proc})',con)
    else:
        if not inc_rt:
            df = pd.read_sql_query(f'SELECT * FROM tweets WHERE retweet = "False"',con)
        else:
            df = pd.read_sql_query(f'SELECT * FROM tweets',con)
    full_pt = []
    for x,y in list(set(list(zip(df['pull_time'].tolist(),df['query_idx'].tolist())))):
        df_chunck = df[(df['pull_time'] == x) & (df['query_idx'] == y) ]
        for idx, row in df_chunck.iterrows():
            pt = processed_feature(row['text'])
            tw= create_bag_of_words(pt)
            op = vadar(pt)
            cur.execute(f"""INSERT INTO tweet_setiment (tweet_id , tokens , setiment_neg , setiment_neu , setiment_pos , setiment_comp )
                            VALUES({row['tweet_id']},"{tw}",{op['neg']},{op['neu']},{op['pos']},{op['comp']})""")
            con.commit()
            full_pt.append(pt)
        sim = cos_similarity(full_pt)
        query_pull_mean = np.matrix(sim).mean()
        bag_group = create_bag_of_words(' '.join(full_pt))
        str_agg = vadar(' '.join(full_pt))
        fd = nltk.probability.FreqDist(bag_group)
        feq = fd.most_common(10)
        cur.execute(f"""INSERT INTO query_setiment (query_idx , pull_time , tokens , setiment_neg , setiment_neu , setiment_pos , setiment_comp , average_similarity, rt_included )
                        VALUES ({row['query_idx']},"{row['pull_time']}","{feq}",{str_agg['neg']},{str_agg['neu']},{str_agg['pos']},{str_agg['comp']}, {query_pull_mean}, "{inc_rt}")""")
    con.commit()