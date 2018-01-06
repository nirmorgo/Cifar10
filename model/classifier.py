import numpy as np
import tensorflow as tf
from model.nets import *

class Classifier():
    def __init__(self, params, Ndims, net=simplenet):
        tf.reset_default_graph()
        self.params = params
        self.Ndims = list(Ndims)
        self.Nlabels=10
        self.temp_folder = params['temp_folder']
        self.learning_rate = params['learning_rate']
        self.verbose = params['verbose']
        self.net = net
        self.best_acc_v = 0       # Used to monitor best set of parameters
        self.init_graph()
        self.training = None
        self.train_writer = tf.summary.FileWriter(self.temp_folder + 'tensorflow_logs/train', graph=self.sess.graph)
        self.val_writer = tf.summary.FileWriter(self.temp_folder + 'tensorflow_logs/val', graph=self.sess.graph)

    def init_graph(self):
        with tf.variable_scope('Inputs'):
            self.X = tf.placeholder(tf.float32, [None]+self.Ndims)
            self.y = tf.placeholder(tf.float32, None)
            self.is_train = tf.placeholder(tf.bool)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.params['WD'], scope='L2_reg')
        with tf.variable_scope('Net'):  
            self.y_out = self.net(self)
        with tf.variable_scope('Loss'):
            self.loss = self.loss_func()
        with tf.variable_scope('Optimizer'):
            # Create the gradient descent optimizer with the given learning rate.
            self.train_step = self.training_step()      
        with tf.variable_scope('Monitors'):
            with tf.variable_scope('Accuracy'):
                self.accuracy = self.accuracy_monitor()
                tf.summary.scalar('Accuracy', self.accuracy)
            tf.summary.scalar('Loss', self.loss)
        self.merged = tf.summary.merge_all()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
    
    def loss_func(self):
        with tf.variable_scope('SoftMax'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y, logits=self.y_out)
            loss = tf.reduce_mean(cross_entropy)
        with tf.variable_scope('Regularizer'):
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
        loss += reg_term
        return loss
    
    def training_step(self):   
        learning_rate = self.params['learning_rate']
        # Create the gradient descent optimizer with the given learning rate.
        step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)    
        return step
    
    def accuracy_monitor(self):
        correct_predictions = tf.equal(tf.argmax(self.y_out,1), tf.argmax(self.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))        
        return accuracy
    
    def update_parameters(self, params):
        self.params = params
        self.save_model_to_checkpoint()
        self.close()
        self.init_graph()
        self.restore_model_from_last_checkpoint()

    def train_iteration(self, i, data, batch_size, iters_per_epoch=None):
        
        if self.training:
            summary_train, acc_t, _, g_step = self.sess.run([self.merged, self.accuracy, self.train_step, self.global_step],
                                                            feed_dict=data.get_train_feed_dict(X=self.X, y=self.y, is_train=self.is_train, batch_size=batch_size))
            self.train_writer.add_summary(summary_train, g_step)

            if i % 10 == 0:
                summary_val, acc_v = self.sess.run([self.merged, self.accuracy],
                                                       feed_dict=data.get_val_feed_dict(X=self.X, y=self.y, 
                                                                                    is_train=self.is_train, 
                                                                                    batch_size=batch_size*4))
                self.val_writer.add_summary(summary_val, g_step)
                    
                # Save the weights that give the best validation accuracy
                if acc_v > self.best_acc_v:
                    self.best_acc_v = acc_v
                    self.save_weights_to_checkpoint(path=self.temp_folder+'/model_files/best_weights/model')

            if iters_per_epoch != None:
                if i % iters_per_epoch == 0:
                    if self.verbose:
                        acc_v = self.sess.run(self.accuracy,feed_dict=data.get_val_feed_dict(X=self.X, y=self.y,
                                                                                        is_train=self.is_train, 
                                                                                        batch_size=batch_size*4))
                        epoch_num = g_step // iters_per_epoch
                        print('__________________________________________________')
                        print("Epoch %d, Train accuracy is:%.3f Validation accuracy:%.3f"%(epoch_num,acc_t,acc_v))                        
            if i % 50 == 0:
                if self.verbose:
                    acc_v = self.sess.run(self.accuracy, feed_dict=data.get_val_feed_dict(X=self.X, y=self.y,
                                                                                        is_train=self.is_train,
                                                                                        batch_size=batch_size*4))
                    print("Step %d, Train accuracy is:%.3f Validation accuracy:%.3f"%(g_step,acc_t,acc_v))

    def train(self, data, epochs=20, batch_size=32):
        self.train_writer = tf.summary.FileWriter(self.temp_folder + 'tensorflow_logs/train', graph=self.sess.graph)
        self.val_writer = tf.summary.FileWriter(self.temp_folder + 'tensorflow_logs/val', graph=self.sess.graph)
        self.training = True
        Ntrain = data.train_idxs.shape[0]
        iters_per_epoch = Ntrain // batch_size
        iterations = iters_per_epoch * epochs
        for i in range(iterations):
            self.train_iteration(i, data, batch_size, iters_per_epoch)
        self.training = False
        print("Train Finished. Best validation accuracy:",self.best_acc_v)
        self.save_model_to_checkpoint() 
    
    def stop_train(self):
        if self.training:
            self.training = False    

    def predict(self, input, max_batch_size=256):
        n_samples = input.shape[0]
        if n_samples > max_batch_size:
            y_out = np.zeros([input.shape[0], self.Nlabels])
            n_iters = n_samples // max_batch_size
            for i in range(n_iters):
                idx = i * max_batch_size
                sample = input[idx:idx+max_batch_size]
                y_out[idx:idx+max_batch_size] = self.sess.run(self.y_out, feed_dict={self.X:sample, self.is_train:False})
            idx += max_batch_size
            y_out[idx:] = self.sess.run(self.y_out, feed_dict={self.X:input[idx:], self.is_train:False})
        else:
            y_out = self.sess.run(self.y_out, feed_dict={self.X:input, self.is_train:False})
        y_out[y_out < -40] = -40    # a cutoof to avoid overflow
        return 1 / (1 + np.exp(-y_out))
    
    def save_model_to_checkpoint(self, path=None):
        saver = tf.train.Saver()
        if path is None:
            path = self.temp_folder+'/model_files/model'
        saver.save(self.sess, path)

    def restore_model_from_last_checkpoint(self, path=None):
        if path is None:
            path = self.temp_folder+'/model_files/model'
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
    
    def save_weights_to_checkpoint(self, path=None):
        if path is None:
            path = self.temp_folder+'/model_files/model'
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.save(self.sess, path)

    def load_weights_from_checkpoint(self, path=None):
        if path is None:
            path = self.temp_folder+'/model_files/model'
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.restore(self.sess, path)    
    
    def reset(self):
        self.sess.close()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def close(self):        
        self.sess.close()
        tf.reset_default_graph()
        self.train_writer.close()
        self.val_writer.close()