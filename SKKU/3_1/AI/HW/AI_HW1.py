## This Code implements simple single Perceptron
import numpy as np
import random
OrData_x = np.array([(0,0),(0,1),(1,0),(1,1)])
OrData_y = np.array([-1,1,1,1])
AndData_x = np.array([(0,0),(0,1),(1,0),(1,1)])
AndData_y = np.array([-1,-1,-1,1])
myOwn_Data_x=np.array([(0,0),(0,1),(1,0),(1,1)])
myOwn_Data_y=np.array([1,-1,-1,1])
learing_rate=0.4
#Data와 parameter를 담을 class
class Par:
    def __init__(self,w=None,b=None):
        self.w=w
        self.b=b
class Data:
    def __init__(self,x=None,y=None):
        self.x=x
        self.y=y
#activatin function
def activate_fn(input):
    if(input>=0):
        return 1
    else:
        return -1
def linear_Transformation(par,data,i): #linear transformation on ith data
    return np.matmul(par.w,data.x[i])+par.b
#Error result를 보여주는 Data의 index를 ret에 저장하여 반환.
def retErrors(par,data):
    ret=np.array([],int)
    for i in range(len(data.x)):
        if(activate_fn(linear_Transformation(par,data,i))!=data.y[i]):
            ret=np.append(ret,int(i))
    return ret
#각 에폭마다 나온 결과를 이용하여 각 에폭의 prediction 결과 출력.
def printResult(par,data,Y):
    print("Among "+str(len(data.x))+" data, there are "+str(len(Y))+" errors.")
    print("The error rate is " + str(float(len(Y)/len(data.x))))
    for i in range(len(data.y)):
        linRet=linear_Transformation(par,data,i)
        print('d(x)=' + str(par.w[0]) + '*'+str(data.x[i][0])+'+' + str(par.w[1]) + '*'+str(data.x[i][1])+'+' + str(par.b) +"= "+str(linRet)+", real class : "+str(data.y[i]), end=' ')
        if(activate_fn(linRet)!=data.y[i]):
            print("Error")
        else:
            print('')
    print("The parameters were: " + str(par.w)+" , "+str(par.b))
    return
#Error Data를 바탕으로 Parameter Update
def update_par(par,data,Y,epoch,printProcess=False):
    errors=np.zeros(2)
    if(printProcess):
        print(str(epoch+1)+"th Epoch")
        print("d(x)="+str(par.w[0])+"x1 + "+str(par.w[1])+"x2 + "+str(par.b))
        print("Y={",end='')
        print(Y[0],end='')
        for i in range(1,len(Y)):
            print(', '+str(Y[i]),end='')
        print('}')
    temp=Par(par.w,par.b)
    for i in Y:
        errors+=data.y[i]*data.x[i]
    par.w+=learing_rate*errors
    par.b+=learing_rate*np.sum(data.y[Y])
    if(printProcess):
        print("w("+str(epoch+1)+") = w("+str(epoch)+") + 0.4(",end='')
        print('t'+str(Y[0])+'*x'+str(Y[0]),end='')
        for i in range(1,len(Y)):
            print("+t"+str(Y[i])+"*x"+str(Y[i]),end='')
        print(")="+str(temp.w)+"0.4[",end='')
        print(str(data.y[Y[0]])+"*"+str(data.x[Y[0]]),end='')
        for i in range(1,len(Y)):
            print("+"+str(data.y[Y[i]])+"*"+str(data.x[Y[i]]),end='')
        print("]="+str(par.w))

        print("b(" + str(epoch + 1) + ") = b(" + str(epoch) + ") + 0.4(", end='')
        print('t' + str(Y[0]) , end='')
        for i in range(1, len(Y)):
            print("+t" + str(Y[i]), end='')
        print(")=" + str(temp.b) + "+0.4*", end='')
        retSum=0
        for i in range(len(Y)):
            retSum+=data.y[Y[i]]
        print(str(retSum)+"="+str(par.b))
    return par
#maxEpoch까지 Parameter Batch Update
def training_process(par,data,maxEpoch=10):
    if(np.all(data.y==myOwn_Data_y)):
        print("w(0)="+str(par.w)+", b(0)="+str(par.b))
    for i in range(maxEpoch):
        Y=retErrors(par,data)
        print("===============================")
        print(str(i+1)+"th Epoch")
        printResult(par,data,Y)
        if(len(Y)==0):
            return par
        if len(Y)!=0:
            par = update_par(par,data,Y,i,np.all(data.y==myOwn_Data_y))
    return par
#Parameter 초기화.
def initiate_Par():
    par=Par(np.random.rand(2)*2-1,random.random()*2-1)
    return par
#Input Data Table 형식으로 출력.
def showData(data):
    for i in range(len(data.x)):
        print("Index : "+str(i)+" | X: "+str(data.x[i])+", Y: "+str(data.y[i]))
def main():
    data=np.array([Data(OrData_x,OrData_y),Data(AndData_x,AndData_y),Data(myOwn_Data_x,myOwn_Data_y)],Data)
    functionName=['OR','And','MyOwn']
    for i in range(len(functionName)):
        print(functionName[i]+" function")
        showData(data[i])
        par=initiate_Par()
        par=training_process(par,data[i])
        print("===============================")
        print("final "+functionName[i] +" Parameter")
        print("W: "+ str(par.w)+", b: "+str(par.b))
        print("\n")
    return
if __name__ == "__main__":
    main()
