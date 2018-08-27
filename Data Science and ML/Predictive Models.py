%matplotlib inline
import numpy as np
from pylab import *

pageSpeeds=np.random.normal(3.0,1.0,1000)
purchaseAmount=100-(pageSpeeds+np.random.normal(0,0.1,1000))*3
scatter(pageSpeeds,purchaseAmount)

from scipy import stats

slope,intercept,r_value,p_value,std_err=stats.linregress(pageSpeeds,purchaseAmount)

r_value**2

import matplotlib.pyplot as plt

def predict(x):
    return slope*x+intercept

fitLine=predict(pageSpeeds)
plt.scatter(pageSpeeds,purchaseAmount)
plt.plot(pageSpeeds,fitLine,c='r')
plt.show()

np.random.seed(2)
pageSpeeds=np.random.normal(3.0,1.0,1000)
purchaseAmount=np.random.normal(50.0,10.0,1000)/pageSpeeds
scatter(pageSpeeds,purchaseAmount)

x=np.array(pageSpeeds)
y=np.array(purchaseAmount)
p4=np.poly1d(np.polyfit(x,y,4))

xp=np.linspace(0,7,100)
plt.scatter(x,y)
plt.plot(xp,p4(xp),c='r')
plt.show()

from sklearn.metrics import r2_score

r2=r2_score(y,p4(x))
print(r2)

import pandas as pd

df=pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')
df.head()

import statsmodels.api as sm

df['Model_ord']=pd.Categorical(df.Model).codes
X=df[['Mileage','Model_ord','Doors']]
y=df[['Price']]

X1=sm.add_constant(X)
est=sm.OLS(y,X1).fit()
est.summary()

y.groupby(df.Doors).mean()