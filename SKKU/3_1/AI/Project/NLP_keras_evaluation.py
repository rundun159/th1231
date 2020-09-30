import numpy as np
import sys
from keras import backend as K
import tensorflow as tf

def class_F1_th(class_idx):
        def F1(y_true, y_pred):
                class_pred = K.cast(y_pred >= 0.5, 'int32')
                # class 0
                idx = tf.constant([class_idx])

                class_pred_idx = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
                class_true_idx = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

                class_recall_idx = K.cast(K.sum(class_pred_idx * class_true_idx), 'float32') / (
                                K.cast(K.sum(class_true_idx), 'float32') + sys.float_info.epsilon)
                class_prec_idx = K.cast(K.sum(class_pred_idx * class_true_idx), 'float32') / (
                        K.cast(K.sum(class_pred_idx), 'float32') + sys.float_info.epsilon)
                class_F1_idx=(2*class_recall_idx*class_prec_idx)/(class_recall_idx+class_prec_idx+sys.float_info.epsilon)
                return class_F1_idx
        return F1

def recall_th(class_idx):
        def recall(y_true, y_pred):
                class_pred = K.cast(y_pred >= 0.5, 'int32')
                # class 0
                idx = tf.constant([class_idx])

                class_pred_idx = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
                class_true_idx = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

                class_recall_idx = K.cast(K.sum(class_pred_idx * class_true_idx), 'float32') / (
                                K.cast(K.sum(class_true_idx), 'float32') + sys.float_info.epsilon)

                return class_recall_idx
        return recall


def precision_th(class_idx):
        def prec(y_true, y_pred):
                class_pred = K.cast(y_pred >= 0.5, 'int32')
                # class 0
                idx = tf.constant([class_idx])

                class_pred_idx = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
                class_true_idx = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

                class_prec_idx = K.cast(K.sum(class_pred_idx * class_true_idx), 'float32') / (
                        K.cast(K.sum(class_pred_idx), 'float32') + sys.float_info.epsilon)

                return class_prec_idx
        return prec


def macro_avg_recall_th(y_true,y_pred):
        class_pred = K.cast(y_pred>=0.5,'int32')
        #class 0
        idx = tf.constant([0])

        class_pred_0=K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)),'int32')
        class_true_0 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_recall_0=K.cast(K.sum(class_pred_0*class_true_0),'float32')/(K.cast(K.sum(class_true_0),'float32')+sys.float_info.epsilon)

        idx = tf.constant([1])

        class_pred_1 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_1 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_recall_1 = K.cast(K.sum(class_pred_1 * class_true_1), 'float32') / (
                        K.cast(K.sum(class_true_1), 'float32') + sys.float_info.epsilon)

        idx = tf.constant([2])

        class_pred_2 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_2 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_recall_2 = K.cast(K.sum(class_pred_2 * class_true_2), 'float32') / (
                        K.cast(K.sum(class_true_2), 'float32') + sys.float_info.epsilon)

        idx = tf.constant([3])

        class_pred_3 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_3 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_recall_3 = K.cast(K.sum(class_pred_3 * class_true_3), 'float32') / (
                        K.cast(K.sum(class_true_3), 'float32') + sys.float_info.epsilon)

        idx = tf.constant([4])

        class_pred_4 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_4 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_recall_4 = K.cast(K.sum(class_pred_4 * class_true_4), 'float32') / (
                        K.cast(K.sum(class_true_4), 'float32') + sys.float_info.epsilon)

        class_recall=class_recall_0+class_recall_1+class_recall_2+class_recall_3+class_recall_4
        print("Macro avg Recall_th function is ready")
        return class_recall/(5+sys.float_info.epsilon)


def macro_avg_prec_th(y_true, y_pred):
        class_pred = K.cast(y_pred >= 0.5, 'int32')
        # class 0
        idx = tf.constant([0])

        class_pred_0 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_0 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_prec_0 = K.cast(K.sum(class_pred_0 * class_true_0), 'float32') / (
                        K.cast(K.sum(class_pred_0), 'float32') + sys.float_info.epsilon)

        idx = tf.constant([1])

        class_pred_1 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_1 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_prec_1 = K.cast(K.sum(class_pred_1 * class_true_1), 'float32') / (
                K.cast(K.sum(class_pred_1), 'float32') + sys.float_info.epsilon)

        idx = tf.constant([2])

        class_pred_2 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_2 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_prec_2 = K.cast(K.sum(class_pred_2 * class_true_2), 'float32') / (
                K.cast(K.sum(class_pred_2), 'float32') + sys.float_info.epsilon)

        idx = tf.constant([3])

        class_pred_3 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_3 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_prec_3 = K.cast(K.sum(class_pred_3 * class_true_3), 'float32') / (
                K.cast(K.sum(class_pred_3), 'float32') + sys.float_info.epsilon)

        idx = tf.constant([4])

        class_pred_4 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_4 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_prec_4 = K.cast(K.sum(class_pred_4 * class_true_4), 'float32') / (
                K.cast(K.sum(class_pred_4), 'float32') + sys.float_info.epsilon)

        class_prec = class_prec_0 + class_prec_1 + class_prec_2 + class_prec_3 + class_prec_4
        print("Macro avg Precision_th function is ready")
        return class_prec / (5+sys.float_info.epsilon)


def macro_avg_F1_th(y_true,y_pred):
        class_pred = K.cast(y_pred >= 0.5, 'int32')
        # class 0
        idx = tf.constant([0])

        class_pred_0 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_0 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_prec_0 = K.cast(K.sum(class_pred_0 * class_true_0), 'float32') / (
                        K.cast(K.sum(class_pred_0), 'float32') + sys.float_info.epsilon)
        class_recall_0 = K.cast(K.sum(class_pred_0 * class_true_0), 'float32') / (
                K.cast(K.sum(class_true_0), 'float32') + sys.float_info.epsilon)
        class_F1_0 = (2 * class_recall_0 * class_prec_0) / (
                        class_recall_0 + class_prec_0 + sys.float_info.epsilon)

        idx = tf.constant([1])

        class_pred_1 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_1 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_prec_1 = K.cast(K.sum(class_pred_1 * class_true_1), 'float32') / (
                        K.cast(K.sum(class_pred_1), 'float32') + sys.float_info.epsilon)
        class_recall_1 = K.cast(K.sum(class_pred_1 * class_true_1), 'float32') / (
                K.cast(K.sum(class_true_1), 'float32') + sys.float_info.epsilon)
        class_F1_1 = (2 * class_recall_1 * class_prec_1) / (
                        class_recall_1 + class_prec_1 + sys.float_info.epsilon)

        idx = tf.constant([2])

        class_pred_2 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_2 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_prec_2 = K.cast(K.sum(class_pred_2 * class_true_2), 'float32') / (
                        K.cast(K.sum(class_pred_2), 'float32') + sys.float_info.epsilon)
        class_recall_2 = K.cast(K.sum(class_pred_2 * class_true_2), 'float32') / (
                K.cast(K.sum(class_true_2), 'float32') + sys.float_info.epsilon)
        class_F1_2 = (2 * class_recall_2 * class_prec_2) / (
                        class_recall_2 + class_prec_2 + sys.float_info.epsilon)

        idx = tf.constant([3])

        class_pred_3 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_3 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_prec_3 = K.cast(K.sum(class_pred_3 * class_true_3), 'float32') / (
                        K.cast(K.sum(class_pred_3), 'float32') + sys.float_info.epsilon)
        class_recall_3 = K.cast(K.sum(class_pred_3 * class_true_3), 'float32') / (
                K.cast(K.sum(class_true_3), 'float32') + sys.float_info.epsilon)
        class_F1_3 = (2 * class_recall_3 * class_prec_3) / (
                        class_recall_3 + class_prec_3 + sys.float_info.epsilon)

        idx = tf.constant([4])

        class_pred_4 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(class_pred), idx)), 'int32')
        class_true_4 = K.cast(tf.transpose(tf.nn.embedding_lookup(tf.transpose(y_true), idx)), 'int32')

        class_prec_4 = K.cast(K.sum(class_pred_4 * class_true_4), 'float32') / (
                        K.cast(K.sum(class_pred_4), 'float32') + sys.float_info.epsilon)
        class_recall_4 = K.cast(K.sum(class_pred_4 * class_true_4), 'float32') / (
                K.cast(K.sum(class_true_4), 'float32') + sys.float_info.epsilon)
        class_F1_4 = (2 * class_recall_4 * class_prec_4) / (
                        class_recall_4 + class_prec_4 + sys.float_info.epsilon)

        class_F1 = class_F1_0 + class_F1_1 + class_F1_2 + class_F1_3 + class_F1_4
        print("Macro avg F1_th function is ready")
        return class_F1 / (5+sys.float_info.epsilon)
