# PoliticalTweetTopics
Poster for Suffolk STEAM Reception: https://drive.google.com/file/d/1x8KmIDuYnOsiTz4BbkfmHYZn4zxgJYpe/view?usp=sharing

Discovering which politicians are leaders or responders in conversations. Tweets by politicians were analyzed in Python over a 4 year span. To extract the top 3 topics for each candidate each month, tweets were grouped monthly using Pandas, LDA models were built for each month. The correlation of each topic was found with all others and is used to build a weighted graph in using NetworkX.
