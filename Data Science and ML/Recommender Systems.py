import pandas as pd

r_cols=['user_id','movie_id','rating']
ratings=pd.read_csv('E://GitHub/Python-Codes/Data Science and ML/ml-100k/u.data',sep='\\t',names=r_cols,usecols=range(3))

m_cols=['movie_id','title']
movies=pd.read_csv('E://GitHub/Python-Codes/Data Science and ML/ml-100k/u.item',sep='\|',names=m_cols,usecols=range(2))

ratings=pd.merge(movies,ratings)
ratings.head()

movieRatings=ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
movieRatings.head()

starWarsRatings=movieRatings['Star Wars (1977)']
starWarsRatings.head()

similarMovies=movieRatings.corrwith(starWarsRatings)
similarMovies=similarMovies.dropna()
df=pd.DataFrame(similarMovies)
df.head()

similarMovies.sort_values(ascending=False)

import numpy as np
movieStats=ratings.groupby('title').agg({'rating':[np.size,np.mean]})
movieStats.head()

popularMovies=movieStats['rating']['size']>=100
movieStats[popularMovies].sort_values([('rating','mean')],ascending=False)[:15]

df=movieStats[popularMovies].join(pd.DataFrame(similarMovies,columns=['similarity']))
df.head()

df.sort_values(['similarity'],ascending=False)[:15]





import pandas as pd

r_cols=['user_id','movie_id','rating']
ratings=pd.read_csv('E://GitHub/Python-Codes/Data Science and ML/ml-100k/u.data',sep='\\t',names=r_cols,usecols=range(3))

m_cols=['movie_id','title']
movies=pd.read_csv('E://GitHub/Python-Codes/Data Science and ML/ml-100k/u.item',sep='\|',names=m_cols,usecols=range(2))

ratings=pd.merge(movies,ratings)
ratings.head()

userRatings=ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
userRatings.head()

corrMatrix=userRatings.corr()
corrMatrix.head()

corrMatrix=userRatings.corr(method='pearson',min_periods=100)
corrMatrix.head()

myRatings=userRatings.loc[0].dropna()
myRatings

simCandidates=pd.Series()
for i in range(0,len(myRatings.index)):
    print("Adding sims for "+myRatings.index[i]+"...")
    #Retrive similar movies to this one that I rated
    sims=corrMatrix[myRatings.index[i]].dropna()
    #Now scale its similarity by how well I rated this movie
    sims=sims.map(lambda x:x*myRatings[i])
    #Add the score to the list of similarity candidates
    simCandidates=simCandidates.append(sims)

#Glance at our results so far:
print("sorting...")
simCandidates.sort_values(inplace=True,ascending=False)
print(simCandidates.head(10))

simCandidates=simCandidates.groupby(simCandidates.index).sum()
simCandidates.sort_values(inplace=True,ascending=False)
simCandidates.head(10)

filteredSims=simCandidates.drop(myRatings.index)
filteredSims.head(10)
