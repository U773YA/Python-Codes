listOfNumbers=[1,2,3,4,5,6]

for number in listOfNumbers:
    print (number),
    if(number%2==0):
        print("is even")
    else:
        print("is odd")
        
print("Hooray! We're all done. Let's party!")

import numpy as np
A=np.random.normal(55.0,5.0,10)
print(A)

x=[1,2,3,4,5,6]
print(len(x))
x[:3]
x[:4]
x[3:]
x[-2:]

x.extend([7,8])
x

x.append(9)
x

y=[10,11,12]
listOfLists=[x,y]
listOfLists

y[1]

z=[3,2,1]
z.sort()
z

z.sort(reverse=True)
z

#Tuples are just immutable lists. Use () instead of []
x=(1,2,3)
len(x)

y=(4,5,6)
y[2]

listOfTuples=[x,y]
listOfTuples

(age,income)="32,120000".split(',')
print(age)
print(income)

#Like a map or hash table in other languages
captains={}
captains["Enterprise"]="Kirk"
captains["Enterprise D"]="Picard"
captains["Deep Space Nine"]="Sisko"
captains["Voyager"]="Janeway"

print(captains['Voyager'])

print(captains.get("Enterprise"))

print(captains.get("NX-01"))

for ship in captains:
    print(ship+":"+captains[ship])
    

def SquareIt(x):
    return x*x
print (SquareIt(2))

#You can pass functions around as parameters
def DoSomething(f,x):
    return f(x)
print(DoSomething(SquareIt,3))

#Lambda functions let you inline simple functions
print(DoSomething(lambda x: x*x*x,3))

print(1==3)
print(True or False)

print(1 is 3)

if 1 is 3:
    print ("How did that happen?")
elif 1>3:
    print("Yikes")
else:
    print("All is well with the world!")
    

for x in range(10):
    print(x),
    
for x in range(10):
    if(x is 1):
        continue
    if(x>5):
        break
    print(x),
    

x=0
while(x<10):
    print(x),
    x+=1
    
