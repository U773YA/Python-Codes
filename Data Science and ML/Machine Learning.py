%matplotlib inline
import numpy as np
from pylab import *

#Train/Test
np.random.seed(2)

pageSpeeds=np.random.normal(3.0,1.0,100)
purchaseAmount=np.random.normal(50.0,30.0,100)/pageSpeeds
scatter(pageSpeeds,purchaseAmount)

trainX=pageSpeeds[:80]
testX=pageSpeeds[80:]

trainY=purchaseAmount[:80]
testY=purchaseAmount[80:]

scatter(trainX,trainY)
scatter(testX,testY)

x=np.array(trainX)
y=np.array(trainY)

p4=np.poly1d(np.polyfit(x,y,4))

import matplotlib.pyplot as plt

xp=np.linspace(0,7,100)
axes=plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0,200])
plt.scatter(x,y)
plt.plot(xp,p4(xp),c='r')
plt.show()

testx=np.array(testX)
testy=np.array(testY)
axes=plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0,200])
plt.scatter(testx,testy)
plt.plot(xp,p4(xp),c='r')
plt.show()

from sklearn.metrics import r2_score
r2=r2_score(testy,p4(testx))
print(r2)

from sklearn.metrics import r2_score
r2=r2_score(np.array(trainY),p4(np.array(trainX)))
print(r2)

#Naive Bayes Classifier
import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root,dirnames,filenames in os.walk(path):
        for filename in filenames:
            path=os.path.join(root,filename)
            inBody=False
            lines=[]
            f=io.open(path,'r',encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line=='\n':
                    inBody=True
            f.close()
            message='\n'.join(lines)
            yield path,message
            
def dataFrameFromDirectory(path,classification):
    rows=[]
    index=[]
    for filename,message in readFiles(path):
        rows.append({'message':message,'class':classification})
        index.append(filename)
    return DataFrame(rows,index=index)

data=DataFrame({'message':[],'class':[]})

data=data.append(dataFrameFromDirectory('E:\GitHub\Python-Codes\Data Science and ML\emails\spam','spam'))
data=data.append(dataFrameFromDirectory('E:\GitHub\Python-Codes\Data Science and ML\emails\ham','ham'))

data.head()

vectorizer=CountVectorizer()
counts=vectorizer.fit_transform(data['message'].values)

classifier=MultinomialNB()
targets=data['class'].values
classifier.fit(counts,targets)

examples=['Free Money now!!!', "Hi Bob, how about a game of golf tomorrow?"]
example_counts=vectorizer.transform(examples)
predictions=classifier.predict(example_counts)
predictions

#K-Mean classifier
from numpy import random,array
#Create fake income/age clusters for N people in k clusters
def createClusteredData(N,k):
    random.seed(10)
    pointsPerCluster=float(N)/k
    X=[]
    for i in range(k):
        incomeCentroid=random.uniform(20000.0,200000.0)
        ageCentroid=random.uniform(20.0,70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid,10000.0),random.normal(ageCentroid,2.0)])
    X=array(X)
    return X

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random, float

data=createClusteredData(100,5)
model=KMeans(n_clusters=5)
#Note I'm scaling the data to normalize it! Important for good results
model=model.fit(scale(data))
#We can look at the clusters each data point was assigned to 
print (model.labels_)
#And we'll visualize it
plt.figure(figsize=(8,6))
plt.scatter(data[:,0],data[:,1],c=model.labels_.astype(float))
plt.show()

#Decision Tree classifier
import numpy as np
import pandas as pd
from sklearn import tree

input_file="E:/GitHub/Python-Codes/Data Science and ML/PastHires.csv"
df=pd.read_csv(input_file,header=0)

df.head()

d={'Y':0,'N':1}
df['Hired']=df['Hired'].map(d)
d={'Y':1,'N':0}
df['Employed?']=df['Employed?'].map(d)
df['Top-tier school']=df['Top-tier school'].map(d)
df['Interned']=df['Interned'].map(d)
d={'BS':0,'MS':1,'PhD':2}
df['Level of Education']=df['Level of Education'].map(d)
df.head()

features=list(df.columns[:6])
features

y=df["Hired"]
X=df[features]
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X,y)

import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38/bin/'

from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus

dot_data=StringIO()
tree.export_graphviz(clf,out_file=dot_data,feature_names=features)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf=clf.fit(X,y)
#Predict employment of an employed 10-year veteran
print(clf.predict([[10,1,4,0,0,0]]))
#... and an unemployed 10-year veteran
print(clf.predict([[10,0,4,0,0,0]]))

#Support Vector Machines
import numpy as np

#Create fake income/age clusters for N people in k clusters
def createClusteredData(N,k):
    pointsPerCluster=float(N)/k
    X=[]
    y=[]
    for i in range(k):
        incomeCentroid=np.random.uniform(20000.0,200000.0)
        ageCentroid=np.random.uniform(20.0,70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid,10000.0),np.random.normal(ageCentroid,2.0)])
            y.append(i)
    X=np.array(X)
    y=np.array(y)
    return X,y

%matplotlib inline
from pylab import *

(X,y)=createClusteredData(100,5)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0],X[:,1],c=y.astype(np.float))
plt.show()

from sklearn import svm,datasets
C=1.0
svc=svm.SVC(kernel='linear',C=C).fit(X,y)

def plotPredictions(clf):
    xx,yy=np.meshgrid(np.arange(0,250000,10),np.arange(10,70,0.5))
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    plt.figure(figsize=(8,6))
    Z=Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,cmap=plt.cm.Paired,alpha=0.8)
    plt.scatter(X[:,0],X[:,1],c=y.astype(np.float))
    plt.show()
    
plotPredictions(svc)

svc.predict([[200000,40]])
svc.predict([[500000,65]])
