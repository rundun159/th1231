import matplotlib.pyplot as plt
import numpy as np
import math
def sigmoid(x):
    a = []
    for item in x:
        a.append(1 / (1+math.exp(-item)))  # 우리가 알고있는 Sigmoid Function을 
                                           # 코드로 표현하면 이렇게 표현됩니다.
    return a
x = np.arange(-10., 10., 0.2)
sig = sigmoid(x)
plt.plot(x, sig)
plt.show()

p = np.arange(0., 1., 0.01)
odds = p/(1-p)
plt.plot(p, odds)
plt.show()

# P값이 1에 가까워 질수록 Odds 값이 무한대로 증가하는 것을 알 수 있다.
# 반대로 P값이 0에 가까워질수록 Odds값은 0에 가까워진다.

def log_odds(p):
    return np.log(p/(1 - p))

x = np.arange(0.005, 1, 0.005)
log_odds_x = log_odds(x)
plt.plot(x, log_odds_x)
plt.ylim(-8, 8)
plt.xlabel('x')
plt.ylabel('log_odds(x)')

plt.yticks([-7, 0, 7])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()
plt.show()

# ### Q1. 위의 그래프를 어떻게 해석하면 좋을까?
# - P값이 1에 가까워 질수록 ...
# - P값이 0에 가까워 질수록 ...

# - log함수는 단조 증가함수입니다. 따라서, log(odds)함수는 odds함수와 증감 경향이 동일하다고 할 수 있습니다.
# - 즉, odds가 증가하면 log(odds)도 증가하고, odds가 감소하면 log(odds)도 감소합니다.
# - 또한 반대로, log(odds)가 증가하면 odds역시 증가하고, log(odds)가 감소하면 odds도 감소합니다.

# - 위의 그래프에서, 
# - p값이 1에 가까워질수록, 즉 p가 증가할 수록 log(odds)의 값이 증가합니다. 따라서 odds 역시 증가합니다.
# - p값이 0에 가까워질수록, 즉 p가 감소할 수록 log(odds)의 값이 감소합니다. 따라서 odds 역시 감소합니다.

x = np.arange(0.005, 1, 0.005)
y = -np.log(1-x)
plt.plot(x,y)
plt.show()

# ### Q2. Logit 값에 역산을 취해주면 어떻게 될까?

y = 1/-np.log(1-x)

plt.plot(x,y)
plt.show()
# - 역산을 취한다는게... 역수를 취하는건가요...?
# - 원래의 x 값의 변화에 따른 변화 양상이 반대가 됩니다.
# - x값이 작아질수록 커지고, x 값이 커질수록 작아집니다.
