import numpy as np
import pickle
from sklearn import svm
from kernel.kernel import *
h = .02  # step size in the mesh

FILE_NAME = "CSV_COSINE"
FILE_NAME_LIST=['CSV_LINEAR','CSV_RBF','CSV_HAM','CSV_COSINE','CSV_INTERPOLATION']
KERNEL_LIST = ['linear','rbf',get_Hamming_Dist,ret_Cosine_Dist,interpolation_HAM_COS]

with open('./pickles/train_csv_x', 'rb') as f:
    train_x = pickle.load(f)
with open('./pickles/test_csv_x', 'rb') as f:
    test_x = pickle.load(f)
with open('./pickles/train_csv_y', 'rb') as f:
    train_y = pickle.load(f)
with open('./pickles/test_csv_y', 'rb') as f:
    test_y = pickle.load(f)

gamma1=0.5
gamma2=0.5
coef=0.5
degree=3
prac=0.5

trX, trY = train_x, train_y
tsX, tsY = test_x, test_y

NUM_OF_FEATURES = trX.shape[1]


def main_exp(i):
    print(FILE_NAME[i])
    clf = svm.SVC(kernel=KERNEL_LIST[i],verbose=True)
    clf.fit(trX, trY)
    pred = clf.predict(tsX)
    with open('./pickles/'+FILE_NAME_LIST[i], 'wb') as f:
        pickle.dump(pred,f)

    real_y_true=test_y==1
    real_y_false=test_y!=1
    pred_y_true=pred==1
    pred_y_false=pred!=1
    tp=np.sum(pred[real_y_true]==1)
    fp=np.sum(test_y[pred_y_true]!=1)
    fn=np.sum(test_y[pred_y_false]==1)
    tn=np.sum(test_y[pred_y_false]!=1)

    DATA_LEN=len(tsX)
    TRUE_LABEL=np.sum(real_y_true)
    TRUE_PRED=np.sum(pred_y_true)

    acc = float((tp+tn)/DATA_LEN)
    prec = float(tp/TRUE_PRED)
    rec = float(tp/TRUE_LABEL)
    try:
        f1 = 2*prec*rec/(prec+rec)
    except:
        f1 =0
    with open('./results/csv/'+FILE_NAME_LIST[i]+'.txt','w') as f:
        f.write("ACC: {:.4f}\n".format(acc))
        f.write("F1score: {:.4f}\n".format(f1))
        f.write("Precision: {:.4f}\n".format(prec))
        f.write("Recall: {:.4f}\n".format(rec))
        f.write("Confusion\n")
        f.write("TP : {}  FP : {}\n".format(tp, fp))
        f.write("FN : {}  TN : {}\n".format(fn, tn))
        f.close()

for i in range(len(FILE_NAME_LIST)):
    main_exp(i)