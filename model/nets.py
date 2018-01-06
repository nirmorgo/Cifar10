# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, dropout, flatten

def simplenet(self):
    x, regularizer = self.X, self.regularizer
    x = flatten(x,scope='flatten')
    
    h1 = fully_connected(inputs=x,
                        num_outputs = 512,
                        activation_fn=tf.nn.relu,
                        weights_regularizer=regularizer,
                        trainable=True,
                        scope='Layer1')
    h2 = fully_connected(inputs = h1,
                        num_outputs=512,
                        activation_fn=tf.nn.relu,
                        weights_regularizer=regularizer,
                        trainable=True,
                        scope='Layer2')
    h3 = fully_connected(inputs = h2,
                        num_outputs=512,
                        activation_fn=tf.nn.relu,
                        weights_regularizer=regularizer,
                        trainable=True,
                        scope='Layer3')
    h4 = fully_connected(inputs = h3,
                        num_outputs=256,
                        activation_fn=tf.nn.relu,
                        weights_regularizer=regularizer,
                        trainable=True,
                        scope='Layer4')
    y_out = fully_connected(inputs=h4,
                        num_outputs=self.Nlabels,
                        activation_fn=None,
                        weights_regularizer=regularizer,
                        trainable=True,
                        scope='Output_Layer')
    return y_out

def convnet(self):
    x = self.X
    regularizer = self.regularizer          
    h1 = tf.layers.conv2d(inputs=x,
                filters=64,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu,
                kernel_regularizer=regularizer,
                name='conv1')
    h1 = tf.layers.max_pooling2d(inputs=h1,
                      pool_size=3,
                      strides=2,
                      padding='valid',
                      name='pool1')       
    h2 = tf.layers.conv2d(inputs=h1,
                          filters=128,
                          kernel_size=3,
                          padding='same',
                          activation=tf.nn.relu,
                          kernel_regularizer=regularizer,
                          name='conv2')
    h2 = tf.layers.max_pooling2d(inputs=h2,
                                pool_size=2,
                                strides=2,
                                padding='valid',
                                name='pool2')
    
    h3 = tf.layers.conv2d(inputs=h2,
                          filters=128,
                          kernel_size=3,
                          padding='same',
                          activation=tf.nn.relu,
                          kernel_regularizer=regularizer,
                          name='conv3')
    h3 = tf.layers.max_pooling2d(inputs=h3,
                                pool_size=2,
                                strides=2,
                                padding='valid',
                                name='pool3')
    
    h4 = tf.layers.conv2d(inputs=h3,
                          filters=64,
                          kernel_size=3,
                          padding='same',
                          activation=tf.nn.relu,
                          kernel_regularizer=regularizer,
                          name='conv4')
    h4 = tf.layers.max_pooling2d(inputs=h4,
                                pool_size=2,
                                strides=2,
                                padding='valid',
                                name='pool4')

    h5 = tf.contrib.layers.flatten(h4,scope='flatten')
    
    h6 = fully_connected(inputs=h5,
                        num_outputs=512,
                        activation_fn=tf.nn.relu,
                        weights_regularizer=regularizer,
                        scope='fc6')
    h7 = fully_connected(inputs=h6,
                        num_outputs=256,
                        activation_fn=tf.nn.relu,
                        weights_regularizer=regularizer,
                        scope='fc7')
    
    y_out = fully_connected(inputs=h7,
                        num_outputs=self.Nlabels,
                        activation_fn=None,
                        weights_regularizer=regularizer,
                        scope='Output_Layer')
    return y_out

def convnet2(self):
    x, is_train = self.X, self.is_train
    params = self.params
    regularizer = self.regularizer          
    h1 = tf.layers.conv2d(inputs=x,
              filters=64,
              kernel_size=3,
              padding='same',
              activation=tf.nn.relu,
              kernel_regularizer=regularizer,
              name='conv1')
    h1 = tf.layers.max_pooling2d(inputs=h1,
                    pool_size=3,
                    strides=2,
                    padding='valid',
                    name='pool1')   
    h1 = dropout(h1, keep_prob=params['keep_ratio'], is_training=is_train, scope='dropout1')    
    h2 = tf.layers.conv2d(inputs=h1,
              filters=128,
              kernel_size=3,
              padding='same',
              activation=tf.nn.relu,
              kernel_regularizer=regularizer,
              name='conv2')
    h2 = tf.layers.max_pooling2d(inputs=h2,
                    pool_size=2,
                    strides=2,
                    padding='valid',
                    name='pool2') 
    h2 = dropout(h2, keep_prob=params['keep_ratio'], is_training=is_train, scope='dropout2') 
    h3 = tf.layers.conv2d(inputs=h2,
              filters=128,
              kernel_size=3,
              padding='same',
              activation=tf.nn.relu,
              kernel_regularizer=regularizer,
              name='conv3')
    h3 = tf.layers.max_pooling2d(inputs=h3,
                    pool_size=2,
                    strides=2,
                    padding='valid',
                    name='pool3') 
    h3 = dropout(h3, keep_prob=params['keep_ratio'], is_training=is_train, scope='dropout3')
    h4 = tf.layers.conv2d(inputs=h3,
              filters=64,
              kernel_size=3,
              padding='same',
              activation=tf.nn.relu,
              kernel_regularizer=regularizer,
              name='conv4')
    h4 = tf.layers.max_pooling2d(inputs=h4,
                    pool_size=2,
                    strides=2,
                    padding='valid',
                    name='pool4')
    h4 = dropout(h4, keep_prob=params['keep_ratio'], is_training=is_train, scope='dropout4')
    h5 = flatten(h4,scope='flatten')
    
    h6 = fully_connected(inputs=h5,
                        num_outputs=512,
                        activation_fn=tf.nn.relu,
                        weights_regularizer=regularizer,
                        scope='fc6')
    h6 = dropout(h6, keep_prob=params['keep_ratio'], is_training=is_train, scope='dropout6')
    h7 = fully_connected(inputs=h6,
                        num_outputs=256,
                        activation_fn=tf.nn.relu,
                        weights_regularizer=regularizer,
                        scope='fc7')   
    h7 = dropout(h7, keep_prob=params['keep_ratio'], is_training=is_train, scope='dropout7')
    y_out = fully_connected(inputs=h7,
                        num_outputs=self.Nlabels,
                        activation_fn=None,
                        weights_regularizer=regularizer,
                        scope='Output_Layer')
    return y_out

