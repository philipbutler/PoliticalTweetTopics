#Data Science Political Tweets Topic Modeling
import pickle
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import re
import os
import numpy as np
import networkx as nx
import community
import csv
import operator

#Compute the correlation between 2 topics
def corr(t1, t2):
    df1 = pd.DataFrame.from_dict(dict(t1), orient='index')
    df2 = pd.DataFrame.from_dict(dict(t2), orient='index')
    cat = pd.concat([df1,df2], axis=1, sort=False).fillna(0)
    return cat.min(axis=1).sum()/cat.max(axis=1).sum()

NAME = ("clinton", "cruz", "kasich", "rubio", "sanders", "stein", "trump")

#Open the topic dataframe if we have it, otherwise create it
if os.path.isfile('./topicDf.p'):
    df = pd.read_pickle('./topicDf.p')
else:
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    print('Unpickling data')
    # Unpickle all files into "data"
    data = []
    for i in range(len(NAME)):
        fName = 'tweets/' + NAME[i] + '.p'
        with open(fName, 'rb') as f:
            data.append(pickle.load(f))
    print('grouping')
    # Group by month, format for processing. Now a 2D list: [politician][month]
    monthly = [x['text'].groupby(pd.Grouper(freq='2W')) for x in data]
    monthlyList = [[group.tolist() for (date, group) in x] for x in monthly]
    
    # For each politician, a list of models of each month added to "allMods"
    allMods = []
    for p in range(7):
        print('Working file')
        LDAmods = []
        m = 0
        # Avoid tweetless months
        for docSet in monthlyList[p]:
            if len(docSet) == 0:
                LDAmods.append(np.nan)
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
                #After inspection of some preliminary results, add a few stop words ot our own
                clean_tokens = [x for x in stemmed_tokens 
                                if not (x in ("co", "t", "amp", "click","0","1","2",
                                "3","4","5","6","7","8","9") or len(x) == (0 or 1))]
                if len(clean_tokens) == 0:
                    continue
                texts.append(clean_tokens)
            print("Creating " + NAME[p] + " model #" + str(m))
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            #Small passes for testing
            LDAmods.append( gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes = 25) )
            m += 1
        allMods.append(LDAmods)
    print('adding topics to df')
    allMods = [[(x.show_topic(0), x.show_topic(1), x.show_topic(2)) 
                if type(x)==gensim.models.ldamodel.LdaModel else ([("NULL", 0)],[("NULL", 0)],[("NULL", 0)]) for x in y] 
                    for y in allMods]
    # Columns are politicians 0-6, rows are months 0-47 starting from 1/2014
    d = {NAME[i]: pd.Series(allMods[i]) for i in range(7)}
    df = pd.DataFrame(d, index=range(105))
    #df.to_pickle('./topicDf.p')

#Open correlation dataframe if we have it, otherwise compute & save them
if os.path.isfile('./CorrDf.p'):
    with open('./CorrDf.p', 'rb') as f:
        cdf = pickle.load(f)
else:
    print('finding corrs')
    #Initialize dictionary & indicies for a DataFrame, add correlations to it
    cDict = {(NAME[j], i, k): [] for j in range(7) for i in range(105) for k in range(3)}
    cIndex = [(NAME[b], a, c) for b in range(7) for a in range(105) for c in range(3)]
    for j in range(7):
        for i in range(105):
            for k in range(3):
                for b in range(7):
                    for a in range(105):
                        for c in range(3):
                            cDict[(NAME[j], i, k)].append(corr(df.iloc[i,j][k], df.iloc[a,b][c]))
    #with open('cDict.p', 'wb') as f:
    #    pickle.dump(cDict, f, protocol=pickle.HIGHEST_PROTOCOL)
    cdf = pd.DataFrame(cDict, index = cIndex)
    cdf.to_pickle('./CorrDf.p')

'''
#For a good option of a threshold, 1s -> 0s, get max series, drop null values, get minimum
maxs = (cdf*(1-np.eye(cdf.shape[0]))).max()
threshold = maxs[maxs>0].min()  #Some Columns & Rows are all 0
cdf = cdf.stack().sort_values()
'''

#Create Topic Graph
s2t = {"('" + NAME[j] + "', " + str(i) + ", " + str(k) + ")" : (NAME[j], i, k) 
              for j in range(7) for i in range(105) for k in range(3)}
t = cdf.stack([0,1,2]).sort_values(0, ascending=False).reset_index()
t['end'] = t[["level_1","level_2","level_3"]].apply(tuple, axis=1)
t['A'] = t['level_0'].apply(str)
t['B'] = t['end'].apply(str)
t.rename(columns={0: "weight"}, inplace=True)
# 0.15 is a better threshold to choose than from the comment above, gives a good modularity
t = t[['A', 'B', 'weight']][t['A'] != t['B']][t['weight']>0.15]
G = nx.from_pandas_edgelist(t, "A", "B", "weight")
p = community.best_partition(G)
m = community.modularity(p, G)
nx.write_gexf(G, "topicGraph.gexf")

#Create Candidate graph
#p is a dictionary of form {topic ID : partition ID}, create a list of lists
#each inner lists containing topics belonging to that group
pGroups = [[] for i in range(71)]
for key in p:
    pGroups[p[key]].append(key)
#Initialize a dataframe
fd = {x:0 for x in NAME}
f = pd.DataFrame(fd, index=NAME)
#Increment a measure of following-ness if a candidate follows another on a topic in the next month
for g in pGroups:
    for a in g:
        for b in g:
            if a != b:
                if s2t[b][1] == s2t[a][1] + 1:
                    f[s2t[a][0]][s2t[b][0]] += 1
                if s2t[a][1] == s2t[b][1] + 1:
                    f[s2t[b][0]][s2t[a][0]] += 1
                else:
                    continue
#Remove loops by multiplying by an inverse identity matrix, divide all by max
f = f*(1-np.eye(f.shape[0]))
fs = f.stack().reset_index()
fs.rename(columns={0:'weight'}, inplace = True)
fs['weight'] = fs['weight']/fs['weight'].max()
CG = nx.from_pandas_edgelist(fs, "level_0", "level_1", "weight", create_using = nx.DiGraph())
nx.write_gexf(CG, "candidGraph.gexf")

print('find macro topics')

#Choose top words based on sum of weights
mds2 = []
i = 0
macro2 = [[df[s2t[x][0]][s2t[x][1]][s2t[x][2]] for x in g] for g in pGroups]
for x in macro2:
    mds2.append({})
    for y in x:
        for z in y:
            if z[0] in mds2[i].keys():
                mds2[i][z[0]] += z[1]
            else:
                mds2[i][z[0]] = z[1]
    i += 1
mds2 = sorted(mds2, key=len, reverse=True)
with open("macro.csv", "w") as outfile:
    writer = csv.writer(outfile)
    for i in range(9):
        sm = sorted(mds2[i].items(), key=operator.itemgetter(1), reverse=True)
        sm = [a for (a,b) in sm]
        writer.writerow(sm)
        
'''
Code used for generating a spreadsheet to create a graph which denotes 
popularity of topics over time
o = sorted(pGroups, key=len, reverse=True)[:9]
with open("freq.csv", "w") as outfile1:
    writer = csv.writer(outfile1)
    header = [i for i in range(105)]
    writer.writerow(header)
    for g in o:
        freq = [0 for i in range(105)]
        for x in g:
            freq[s2t[x][1]] += 1
        writer.writerow(freq)
'''