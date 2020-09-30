import numpy as np
import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg.distributed import RowMatrix
from sklearn.datasets import fetch_openml
import pickle

with open('./pickles/train_img', 'rb') as f:
   train_img = pickle.load(f)
with open('./pickles/test_img', 'rb') as f:
   test_img = pickle.load(f)

conf = SparkConf()
conf.set("spark.master","local")
sc = SparkContext(conf=conf)

rdd = sc.parallelize(train_img.tolist(),10)
rdd.cache()
mat = RowMatrix(rdd)

pc_rdd = mat.computePrincipalComponents(300)
pc = pc_rdd.toArray()
pct = np.transpose(pc)

with open('./pickles/pca_img', 'wb') as f:
   pickle.dump(pct, f)

image_shape = (28,28)
fig,axes = plt.subplots(2, 8, figsize=(15,12),subplot_kw = {'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pct, axes.ravel())):
   ax.imshow(component.reshape(image_shape), cmap='gray_r')

plt.savefig('result.png')
sc.stop()