#Data Science Political Tweets Topic Modeling
import pickle
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import csv
import re
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

#Compute the correlation between 2 topics
def corr(t1, t2):
    df1 = pd.DataFrame.from_dict(dict(t1), orient='index')
    df2 = pd.DataFrame.from_dict(dict(t2), orient='index')
    cat = pd.concat([df1,df2], axis=1, sort=False).fillna(0)
    return cat.min(axis=1).sum()/cat.max(axis=1).sum()

#Open the topic dataframe if we have it, otherwise create it
if os.path.isfile('./topicDf.p'):
    df = pd.read_pickle('./topicDf.p')
else:
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    
    # Unpickle all files into "data"
    name = ("clinton", "cruz", "kasich", "rubio", "sanders", "stein", "trump")
    data = []
    for i in range(len(name)):
        fName = 'tweets/' + name[i] + '.p'
        with open(fName, 'rb') as f:
            data.append(pickle.load(f))
    
    # Group by month, format for processing. Now a 2D list: [politician][month]
    monthly = [x['text'].groupby(pd.Grouper(freq='M')) for x in data]
    monthlyList = [[group.tolist() for (date, group) in x] for x in monthly]
    
    # For each politician, a list of models of each month added to "allMods"
    allMods = []
    for p in range(7):
        LDAmods = []
        m = 0
        # Avoid tweetless months
        for docSet in monthlyList[p]:
            if len(docSet) == 0:
                LDAmods.append("Empty Month")
                print("Empty Month")
                m += 1
                continue
            
            # Clean data & add models
            texts = []
            for i in docSet:
                raw = i.lower()
                raw = re.sub(r"(http.*? )|(http.*?$)|(@.*? )|(@.*?$)", "", raw, flags=re.DOTALL)
                tokens = tokenizer.tokenize(raw)
                stopped_tokens = [x for x in tokens if not x in en_stop]
                stemmed_tokens = [p_stemmer.stem(x) for x in stopped_tokens]
                clean_tokens = [x for x in stemmed_tokens 
                                if not (x in ("co", "t", "amp", "click","0","1","2",
                                "3","4","5","6","7","8","9") or len(x) == (0 or 1))]
                if len(clean_tokens) == 0:
                    continue
                texts.append(clean_tokens)
            print("Creating " + name[p] + " model #" + str(m))
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            #Small passes for testing
            LDAmods.append( gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes = 25) )
            m += 1
        allMods.append(LDAmods)
    allMods = [[(x.show_topic(0), x.show_topic(1), x.show_topic(2)) 
                if type(x)==gensim.models.ldamodel.LdaModel else ([("NULL", 0)],[("NULL", 0)],[("NULL", 0)]) for x in y] 
                    for y in allMods]
    # Columns are politicians 0-6, rows are months 0-47 starting from 1/2014
    d = {name[i]: pd.Series(allMods[i]) for i in range(7)}
    df = pd.DataFrame(d, index=range(48))
    df.to_pickle('./topicDf.p')

#Open correlations if we have it, otherwise compute & save them
if os.path.isfile('./cDict.p'):
    with open('cDict.p', 'rb') as f:
        cDict = pickle.load(f)
else:
    #Initialize dictionary & indicies for a DataFrame, add correlations to it
    cDict = {(name[j], i, k): [] for j in range(7) for i in range(48) for k in range(3)}
    cIndex = [(name[b], a, c) for b in range(7) for a in range(48) for c in range(3)]
    for j in range(7):
        for i in range(48):
            for k in range(3):
                for b in range(7):
                    for a in range(48):
                        for c in range(3):
                            cDict[(name[j], i, k)].append(corr(df.iloc[i,j][k], df.iloc[a,b][c]))
        print(str(int(((j+1)/7)*100)) + "% complete")
    with open('cDict.p', 'wb') as f:
        pickle.dump(cDict, f, protocol=pickle.HIGHEST_PROTOCOL)
    cdf = pd.DataFrame(cDict, index = cIndex)
    cdf.to_pickle('./CorrDf.p')
    
if os.path.isfile('./CorrDf.p'):
    with open('CorrDf.p', 'rb') as f:
        cdf = pickle.load(f)

#For threshold, 1s -> 0s, get max series, drop null values, get minimum
threshold = cdf.replace(1,0).max().drop(index=[('stein', i,j) for i in range(5, 8) for j in range(3)]).min()
print(threshold)

'''
ax = plt.subplot(111)
im = ax.imshow(cdf)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)
plt.savefig('plot.png')
'''