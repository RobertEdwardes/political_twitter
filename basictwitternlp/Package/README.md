Basic Twitter NLP
----
Description: Simple set of comands to work with twitter and text sediment analysis

NEEDED: Twitter Developer account for bearer token [Twitter Developer](https://developer.twitter.com/en/docs/platform-overview)

### Getting Started 
-----
Build Database to Store tweets and analysis. It will create the 5 tables needed for the process 
```python
db_init(con)
```
Add Querys. It takes any query that is allowed by your Twitter api acesses. [Twitter Query Help Guide](https://developer.twitter.com/en/docs/tutorials/building-high-quality-filters)
```python
add_query(twitter_query, con)
```
Run Process. This will run the scrape and NLP process and store them in the tables
```python
run_tw_nlp(con, client, query=None, inc_rt=False)
````
con : SQLite3 connection   ``` con = sqlite3.connection('DATABASE_NAME.db')```

client : twitter bearer token ``` client =  tweepy.Client(bearer_token='bearer') ```

query : query index number which can be found in the "query_idx" column in the queries table

inc_rt : to include retweets in text analysis