#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:50:27 2019

@author: ma
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score


#def precessing_data(self, continue_data=None, cat_data):
#        '''
#        data应该是一个pandas数据框, 返回离散值编码后的数据框和对应的fild索引
#        '''

class DeepFM():
    '''
    输入是一个 N*F的数据框， 其中N是样本个数，F是特征的fild个数，特征包含连续特征和
    离散特征，但是数据格式要求为连续特征在前，离散特征在后。
    '''
    
    def __init__(self, feature_size, fild_size, embedding_size=8,
                 deep_layer_size=[50,50,50], activation_function=tf.nn.relu,
                 epochs=20, batch_size=50, learning_rate=0.001, l2_reg=0.0,
                 dropout_rate=[0.5,0.5,0.5], is_training=True):
        
        self.feature_size = feature_size  # m
        self.embedding_size = embedding_size # k
        self.fild_size = fild_size  # F
        
        self.deep_layer_size = deep_layer_size
        self.activation_function = activation_function
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.is_training = is_training
    
        self.build_graph()
    
    
    def init_weights(self):
        '''初始化FM和网络的权重'''
        weights = {}
        
#        FM weight
        weights['FM_embedding'] = tf.Variable(tf.random_normal(shape=[self.feature_size, self.embedding_size],
                                               mean=0, stddev=0.01), name='FM_embedding')
        weights['FM_linear'] = tf.Variable(tf.random_normal(shape=[self.feature_size, 1]), name='FM_linear')
        
##        DNN weight
#        deep_len = len(self.deep_layer_size)
#        input_size = self.fild_size * self.embedding_size
#        for i in range( deep_len):
#            xavier_init = np.sqrt(2 / (input_size+self.deep_layer_size[i]))
#            weights['deep_layer{}'.format(i+1)] = tf.Variable(tf.random_normal(shape=[input_size, 
#                                                self.deep_layer_size[i]], mean=0, stddev=xavier_init),
#                                                name='deep_layer{}'.format(i+1))
#            weights['deep_bias{}'.format(i+1)] = tf.Variable(tf.zeros([self.deep_layer_size[i]]),
#                                                           name='deep_bias{}'.format(i+1))
#            input_size = self.deep_layer_size[i]
#        weights['deep_output'] = tf.Variable(tf.random_normal(shape=[self.deep_layer_size[-1], 1]),
#                                           name='deep_output')
#        weights['deep_output_bias'] = tf.Variable(tf.zeros(shape=[1]), name='deep_output_bias')
        
        return weights
        

    def build_graph(self):
        '''初始化网络的图结构'''
        
        self.weights = self.init_weights()
        
        self.feat_index = tf.placeholder(tf.int32, shape=[None, self.fild_size], name='X_index')
        self.feat_value = tf.placeholder(tf.float32, shape=[None, self.fild_size], name='X_value')
        self.y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
        
#        FM compoent
#        linear term
        w = tf.nn.embedding_lookup(self.weights['FM_linear'], self.feat_index) # 应该是None*F*1维的
        feat_value = tf.reshape(self.feat_value, [-1, self.fild_size, 1]) #由None*1*F，变为None*F*1
        FM_linear_term = tf.reduce_sum(tf.multiply(w, feat_value), 1) # 应该是None * 1
        
#        interaction term
        v = tf.nn.embedding_lookup(self.weights['FM_embedding'], self.feat_index) # 应该是None*F*k维
        
#        v的形状
#        [[[.....],[...k维]，共F个
#                ]，
#        [[],...],
#        [[],....],None个
#                ]
        
        
#        和的平方
        FM_sum_square = tf.square(tf.reduce_sum(tf.multiply(v, feat_value), 1)) # None*k
#        平方的和
        FM_square_sum = tf.reduce_sum(tf.square(tf.multiply(v, feat_value)), 1) # None*k
        
        FM_interaction_term = 0.5*tf.reduce_sum(tf.subtract(FM_sum_square, FM_square_sum), 1) # None*1
        FM_interaction_term = tf.reshape(FM_interaction_term, [-1, 1])
        
        y_FM = FM_linear_term + FM_interaction_term #None*1
        
#        DNN compoent
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        bn_params = {
                'is_training': self.is_training,
                'decay': 0.995,
                'updates_collections': None}
        
        v = tf.multiply(v, feat_value)
        input_X = tf.reshape(v, shape=[-1, self.fild_size * self.embedding_size]) # None*(F*k)
        for i in range(len(self.deep_layer_size)):
            hidden_bn = tf.contrib.layers.fully_connected(inputs = input_X,
                                                      num_outputs = self.deep_layer_size[i],
                                                      normalizer_fn = tf.contrib.layers.batch_norm,
                                                      normalizer_params = bn_params,
                                                      weights_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg),
                                                      activation_fn = self.activation_function
                                                      ) 
            input_X = hidden_bn
            
        y_deep = tf.contrib.layers.fully_connected(inputs=input_X, num_outputs=1,
                                                   activation_fn = tf.identity,
                                                   weights_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)
                                                   )    
        
        
#        with tf.variable_scope('deep'):
#            
#            input_X = tf.reshape(v, shape=[-1, self.fild_size * self.embedding_size]) # None*(F*k)
#            for i in range(len(self.deep_layer_size)):
#                next_layer_input = tf.add(tf.matmul(input_X, self.weights['deep_layer{}'.format(i+1)]),
#                                    self.weights['deep_bias{}'.format(i+1)])
#                
#                layer_output = tf.nn.relu(next_layer_input)
#                input_X = layer_output
#        y_deep = tf.add(tf.matmul(input_X, self.weights['deep_output']),
#                                    self.weights['deep_output_bias']) # None*1

#        FM+DNN
        logit = y_FM + y_deep # None*1
#        logit = tf.reshape(logit,[-1,1])
#        输出概率
        self.y_pred = tf.sigmoid(logit)
        
#        loss
        with tf.variable_scope('loss'):
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=self.y)
            self.loss = tf.reduce_mean(entropy)+ \
            self.l2_reg * (tf.nn.l2_loss(self.weights['FM_linear']) + tf.nn.l2_loss(self.weights['FM_embedding']))
            
#        optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.train_op = self.optimizer.minimize(self.loss)
#        打开一个会话
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    
    def train_model(self, X_index, X_value, labels):
        
        for epoch in range(self.epochs):
            for X_index_batch, X_value_batch, y_batch in self.random_shuffle_batch(X_index, X_value, labels):
                
                self.sess.run(self.train_op, feed_dict={self.feat_index:X_index_batch, self.feat_value:X_value_batch, self.y:y_batch})
            loss, auc = self.evaluate(X_index, X_value, labels)
            print(epoch, 'loss:',loss,'auc',auc)
                
    def evaluate(self, X_index, X_value, labels):
#        print(X_index, X_value, labels)
#        print(labels.shape)
        
        y_pred, loss = self.sess.run([self.y_pred, self.loss], feed_dict={self.feat_index:X_index, self.feat_value:X_value, self.y:labels})
        
#        print(self.sess.run([self.y_linear, self.y_interaction, self.y_pred], feed_dict={self.feat_index:X_index, self.feat_value:X_value}))
        auc = roc_auc_score(labels, y_pred)
        return loss, auc
    
    def predict(self, X_index, X_value):
        y_pred =self.sess.run(self.y_pred, feed_dict={self.feat_index:X_index, self.feat_value:X_value})
        return y_pred
        
    def random_shuffle_batch(self, X_index, X_value, y):
        rnd_index = np.random.permutation(len(X_index))
        n_batches = len(X_index) // self.batch_size
        for idx in np.array_split(rnd_index, n_batches):
            #print(idx, rnd_index, n_batches)
            #print(X_index, X_value, y)
            X_index_batch, X_value_batch, y_batch = X_index[idx], X_value[idx], y[idx]  
            yield X_index_batch, X_value_batch, y_batch
        
#deep_fm = DeepFM(feature_size=6, fild_size=2)
#feat_index = np.array([[1,3],[2,4],[1,5]])
#feat_value = np.array([[1,1],[2,2],[1,1]])
#y = np.array([1,1,0]).reshape([-1,1])
#
#test_index = np.array([[0,1],[3,5]])
#test_value = np.array([[1,1],[1,3]])
#deep_fm.train_model(feat_index, feat_value , y)
#print(deep_fm.predict(test_index, test_value))
#print(deep_fm.predict(feat_index, feat_value))
        