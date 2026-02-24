import numpy as np
import time as time

pf = np.zeros(100)
v = np.zeros(100)
p = np.zeros(100)
a = np.zeros(100)
num = 1
for i in range(0,100): # for number variation
    p[i] = num
    num += 0.2
    if(i == 50):
        num = 2

alpha = 0.8
t = 0.01

for k in range(1,100):
    timeStamp = time.perf_counter()
    while((time.perf_counter() - timeStamp) < 0.01):
        pass
    pf[k] = (1-alpha)*pf[k-1]+alpha*p[k]
    v[k] = (pf[k] - pf[k-1])/0.01
    a[k] = (v[k] - v[k-1])/0.01

print(pf)
print(v)
print(a)