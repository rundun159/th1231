import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from pyspark import SparkConf, SparkContext
import pickle
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


h = .02  # step size in the mesh

with open('./pickles/test_img', 'rb') as f:
   test_x = pickle.load(f)
with open('./pickles/train_img', 'rb') as f:
   train_x = pickle.load(f)
with open('./pickles/train_csv_y', 'rb') as f:
    train_y = pickle.load(f)
with open('./pickles/test_csv_y', 'rb') as f:
    test_y = pickle.load(f)

# gamma1=0.5
# gamma2=0.5
# coef=0.5
# degree=3
# prac=0.5

def ret_custom_rbf(gamma):
    def custom_rbf(x1, x2):
        ret = np.zeros(shape=(len(x1),len(x2)),dtype=np.float)
        for idx1, _x1 in enumerate(x1):
            for idx2, _x2 in enumerate(x2):
                diff = _x1-_x2
                ret[idx1][idx2]=math.exp(gamma*-1*np.matmul(diff,np.transpose(diff)))
        return ret
    return custom_rbf
def ret_custom_poly(gamma,coef,degree):
    def custom_poly(x1, x2):
        return np.power(gamma*np.matmul(x1,np.transpose(x2)+coef),degree)
    return custom_poly

def ret_TH_custom_kernel(gamma1,gamma2,coef,degree,prac):
    def TH_custom_kernel(x1,x2):
        TH_rbf = ret_custom_rbf(gamma=gamma1)
        TH_poly = ret_custom_poly(gamma=gamma2,coef=coef,degree=degree)
        return TH_rbf(x1,x2)*prac + (1-prac)*TH_poly(x1,x2)
    return TH_custom_kernel

# def final_custom_kernel(x1,x2):
#     ret = np.zeros(shape=(len(x1), len(x2)), dtype=np.float)
#     for idx1, _x1 in enumerate(x1):
#         for idx2, _x2 in enumerate(x2):
#             diff = _x1 - _x2
#             ret[idx1][idx2] = math.exp(gamma1 * -1 * np.matmul(diff, np.transpose(diff)))
#     return ret*prac + np.power(gamma2*np.matmul(x1,np.transpose(x2)+coef),degree)

numPartition = 100

trX, trY = train_x, train_y
tsX, tsY = test_x, test_y

NUM_OF_FEATURES = trX.shape[1]

clf = svm.SVC(kernel=ret_custom_rbf(float(1/NUM_OF_FEATURES)))
# clf = svm.SVC(kernel=ret_custom_rbf(float(1/NUM_OF_FEATURES)))
# clf = svm.SVC(kernel=ret_TH_custom_kernel(gamma1=0.5,gamma2=0.5,coef=0.5,degree=2,prac=0.5))
clf.fit(trX, trY)
pred = clf.predict(tsX)
with open('./pickles/pred_result_pca', 'wb') as f:
    pickle.dump(pred,f)

#
#
# conf = SparkConf()
# sc = SparkContext(conf=conf)
#
# gamma1=0.5
# gamma2=0.5
# coef=0.5
# degree=3
# prac=0.5
# def custom_rbf(x1, x2):
#     ret = np.zeros(shape=(len(x1),len(x2)),dtype=np.float)
#     for idx1, _x1 in enumerate(x1):
#         for idx2, _x2 in enumerate(x2):
#             diff = _x1-_x2
#             ret[idx1][idx2]=math.exp(gamma1*-1*np.matmul(diff,np.transpose(diff)))
#     return ret
#
# trRDDs = sc.parallelize(trX.tolist(), numPartition)
# tsRDDs = sc.parallelize(tsX.tolist(), numPartition)
#
# trRDDs.cache()
# tsRDDs.cache()
# TH_Kernel = SVC(kernel=custom_rbf)
# # TH_Kernel = SVC(kernel=final_custom_kernel)
# TH_Kernel.fit(trRDDs.collect(), trY)
# print('Kernel Done1')
# TH_Kernel = sc.broadcast(TH_Kernel)
# print('Kernel Done2')
# TH_Kernel_result = tsRDDs.map(lambda x:TH_Kernel.value.predict(np.array(x).reshape(1,-1)))
# print('Kernel Done3')
# TH_Kernel_result = TH_Kernel_result.collect()
# print('Kernel Done4')
# TH_Kernel_pred = [int(x[0]) for x in TH_Kernel_result]
# print('Kernel Done5')
# TH_Kernel_acc = accuracy_score(tsY.astype(np.int).tolist(), TH_Kernel_pred)
# print('Kernel Done6')
# TH_Kernel_f1 = f1_score(tsY.astype(np.int).tolist(), TH_Kernel_pred, average='macro')
# print('Kernel Done7')
# TH_Kernel_tn, TH_Kernel_fp, TH_Kernel_fn, TH_Kernel_tp = confusion_matrix(
#     TH_Kernel_pred, tsY.astype(np.int).tolist()).ravel()
# f = open("result_custom_kernel.txt", "w")
# f.write("Custom Kernel ACC: {:.4f}\n".format(TH_Kernel_acc))
# f.write("Custom Kernel F1score: {:.4f}\n".format(TH_Kernel_f1))
# f.write("Custom Kernel Confusion\n")
# f.write("{} {}\n".format(TH_Kernel_tn, TH_Kernel_fp))
# f.write("{} {}\n".format(TH_Kernel_fn, TH_Kernel_tp))
# f.close()
#
# sc.stop()
