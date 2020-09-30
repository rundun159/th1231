import random
import numpy as np
from optim.Optimizer import SGD
from model.SoftmaxClassifier import SoftmaxClassifier
from utils import load_data, accuracy, display_image_predictions, load_label_dict

class Model_record:
    def __init__(self,acc,model,lr,epochs):
        self.model_acc = acc
        self.model = model
        self.model_lr = lr
        self.model_epochs = epochs
        self.model_test_acc=None
    def get_train_acc(self, testData):
        test_x, test_y = test_data
        pred, prob = self.model.eval(test_x)
        test_acc = accuracy(pred, test_y)
        self.model_test_acc=test_acc
    def del_model(self):
        del(self.model)


np.random.seed(428)

# ========================= EDIT HERE =========================
# 1. Choose DATA : Fashion_mnist, Iris
# 2. Adjust Hyperparameters

# DATA
DATA_NAME = 'Fashion_mnist'

# HYPERPARAMETERS
num_epochs = [50]
batch_size = 512
learning_rate = [0.00001]

show_plot = True
# =============================================================
assert DATA_NAME in ['Iris', 'Fashion_mnist']
grid_search = [(x, y) for x in num_epochs for y in learning_rate]

# Load dataset, model and evaluation metric
train_data, test_data = load_data(DATA_NAME)
train_x, train_y = train_data

num_train = len(train_x)
perm = np.random.permutation(num_train)
num_valid = int(len(train_x) * 0.1)
valid_idx = perm[:num_valid]
train_idx = perm[num_valid:]

valid_x, valid_y = train_x[valid_idx], train_y[valid_idx]
train_x, train_y = train_x[train_idx], train_y[train_idx]

num_data, num_features = train_x.shape
num_label = int(train_y.max()) + 1
print('# of Training data : %d \n' % num_data)

results = {}
best_acc = -1
best_model = None
best_lr = None
best_epochs = None
model_list=[]

# For each set of parameters in 'grid_search', train and evaluate softmax classifier.
# Save search history in dictionary 'results'.
#   - KEY: tuple of (# of epochs, learning rate)
#   - VALUE: accuracy on validation data
# Save the best validation accuracy and optimized model in 'best_acc' and 'best_model'.
test_x, test_y = test_data

for ep, lr in grid_search:
    # Make model & optimizer
    model = SoftmaxClassifier(num_features, num_label)
    optim = SGD()

    model.train(train_x, train_y, ep, batch_size, lr, optim)

    pred, prob = model.eval(valid_x)

    valid_acc = accuracy(pred, valid_y)
    print('Accuracy on valid data : %f\n' % valid_acc)

    results[ep, lr] = valid_acc
    
    if valid_acc > best_acc:
        best_acc = valid_acc
        best_model = model
        best_lr=lr
        best_epochs=ep
    model_list.append(Model_record(valid_acc,model,lr,ep))
    model_list[-1].get_train_acc(test_data)
    model_list[-1].del_model()
for ep, lr in sorted(results):
    valid_acc = results[(ep, lr)]
    print('# epochs : %d lr : %e valid accuracy : %f' % (ep, lr, valid_acc))
    
print('best validation accuracy achieved: %f' % best_acc)

# Evaluate best model on test data
# test_x, test_y = test_data
# pred, prob = best_model.eval(test_x)
pred, prob = model.eval(test_x)
test_acc = accuracy(pred, test_y)
print('test accuracy of best model : %f' % test_acc)
print("best lr : %f" % lr)
print("best epochs : %f " %ep)

print("TH")
model_list.sort(key=lambda ele:ele.model_test_acc,reverse=True)
print("Why")
# for idx in range(5):
#     print("%d th model. acc: %f lr:  %f epochs: %f test_acc: %f" %(idx+1,model_list[idx].model_acc,model_list[idx].model_lr,model_list[idx].model_epochs,model_list[idx].model_test_acc))
# Plot prediction of the best model
if show_plot and DATA_NAME == 'Fashion_mnist':
    print("Why??")
    num_test = len(test_x)
    test_x = test_x[:, 1:]
    test_x = test_x.reshape(num_test, 28, 28)
    
    random_idx = np.random.choice(num_test, 5)
    sample_data = test_x[random_idx]
    sample_label = test_y[random_idx]
    sample_prob = prob[random_idx]

    label_dict = load_label_dict(DATA_NAME)
    print("Why??????")
    display_image_predictions(sample_data, sample_label, sample_prob, label_dict)