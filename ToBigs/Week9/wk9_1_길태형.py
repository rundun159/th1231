import numpy as np
N=int(input())
#가상의 첫번째 방을 만듭니다.
#첫번째 방에는 금괴가 하나도 없습니다.
#단지 결과값을 구할때, getGold(0,0)만 호출해도, 모든 방을 처음 들어간 것처럼
#탐색할 수 있도록 구현합니다.
golds=np.zeros(N+1,int)
for i in range(N):
    golds[i+1]=int(input())
cache=np.full((N+1,3),-1,dtype=int)
def getGold(idx,seq): #idx번째 방을, seq번째 연속으로 들어갈때 얻을 수 있는 최고 양.
    if idx > N:
        return 0
    if seq>=3:
        return 0
    if cache[idx][seq]!=-1:
        return cache[idx][seq]
    #구현을 편하게 하기 위해
    #ret을 golds[idx]가 아닌 0으로 초기화합니다.
    ret = 0
    #다음 방을 연속해서 들어갑니다.
    ret = getGold(idx+1,seq+1)
    for i in range(idx+2,N):
        #연속하지 않고 다른 방들을 들어가는 경우입니다. 그중 최대만 저장합니다.
        ret=max(ret,getGold(i,1))
    #idx번째 방의 금괴를 더합니다.
    ret += golds[idx]
    cache[idx][seq]=ret
    return ret
print(getGold(0,0))

