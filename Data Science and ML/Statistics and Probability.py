import numpy as np

incomes=np.random.normal(27000,15000,10000)
np.mean(incomes)

%matplotlib inline
import matplotlib.pyplot as plt
plt.hist(incomes,50)
plt.show()

np.median(incomes)

incomes=np.append(incomes,[1000000000])

np.median(incomes)
np.mean(incomes)

ages=np.random.randint(18,high=90,size=500)
ages

from scipy import stats
stats.mode(ages)

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
incomes=np.random.normal(100.0,20.0,10000)
plt.hist(incomes,50)
plt.show()

incomes.std()
incomes.var()

import numpy as np
import matplotlib.pyplot as plt

values=np.random.uniform(-10.0,10.0,100000)
plt.hist(values,50)
plt.show()

from scipy.stats import norm
import matplotlib.pyplot as plt

x=np.arange(-3,3,0.001)
plt.plot(x,norm.pdf(x))

mu=5.0
sigma=2.0
values=np.random.normal(mu,sigma,10000)
plt.hist(values,50)
plt.show()

from scipy.stats import expon
x=np.arange(0,10,0.001)
plt.plot(x,expon.pdf(x))

from scipy.stats import binom
x=np.arange(0,10,0.01)
plt.plot(x,binom.pmf(x,10,0.5))

from scipy.stats import poisson
mu=500
x=np.arange(400,600,0.5)
plt.plot(x,poisson.pmf(x,mu))

vals=np.random.normal(0,0.5,10000)
plt.hist(vals,50)
plt.show()

np.percentile(vals,50)
np.percentile(vals,90)
np.percentile(vals,20)

vals=np.random.normal(0,0.5,10000)
plt.hist(vals,50)
plt.show()
np.mean(vals)
np.var(vals)

import scipy.stats as sp
sp.skew(vals)
sp.kurtosis(vals)
