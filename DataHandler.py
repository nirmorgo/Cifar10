import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import misc

#%%
class Cifar10_Data():
    def __init__(self):
        self.train_idxs = None
        self.val_idxs = None
        self.rand_seed = None
        self.current_train_idx = 0 # Will be used to select random samples from entire epoch
        self.current_val_idx = 0
        self.Ntrain = 0
        self.Nval = 0
        self.Ndims = (32,32,3)
        self.encoding = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,
                         'dog':5,'frog':6, 'horse':7, 'ship': 8, 'truck': 9}
        self.decoding = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
                         5:'dog',6:'frog', 7:'horse',  8:'ship', 9:'truck'}

    def restart_epoch(self):
        self.rand_train_idxs = np.random.permutation(self.Ntrain)
        self.current_train_idx = 0
    
    def restart_val_set(self):
        self.rand_val_idxs = np.random.permutation(self.Nval)
        self.current_val_idx = 0
    
    def get_train_data(self):
        return self.train_idxs
    
    def get_val_data(self):
        return self.val_idxs
    
    def load_images(self, N=50000):
        self.Xdata = []
        for idx in range(N):
            img_path = self.data_path + str(idx+1) + '.png'
            self.Xdata.append(misc.imread(img_path))
        self.Xdata = np.array(self.Xdata)
        self.mean = np.mean(self.Xdata,axis=(0,1,2,3))
        self.std = np.std(self.Xdata,axis=(0,1,2,3))
        self.Xdata = (self.Xdata-self.mean)/(self.std+1e-7)
        

    def load_batch(self, idxs):
        '''
        input: a list of index numbers
        outputs:
            X = tensor of loaded images
            y = a matrix with encoded labels
        '''
        X = self.Xdata[idxs] 
        batch_size = len(idxs)
        y = np.zeros((batch_size,10))
        for i, idx in enumerate(idxs):
            label=self.labels[idx]['label']
            y[i,self.encoding[label]] = 1            
        return X, y
    
    def get_train_feed_dict(self, X, y, is_train, batch_size=None):
        '''
        gets a feed dictionary out of the data set.
        inputs: 
                X, y: tensorflow place holders with same dimensions as X and y in the dataset
                is_train: tensorflow boolean, will return validation data if false
                batch_size: relevant for train only, return a randomly selected batch from dataset
        output: a feed dictionary for tensorflow session.
        '''
        np.random.seed(None)
        if self.current_train_idx + batch_size > self.Ntrain:
            self.restart_epoch()
        idxs = self.rand_train_idxs[self.current_train_idx:self.current_train_idx + batch_size]
        self.current_train_idx += batch_size
        X_batch, y_batch = self.load_batch(idxs)
        return {X: X_batch, y: y_batch, is_train: True}
     
    def get_val_feed_dict(self, X, y, is_train, batch_size=None):
        np.random.seed(None)
        if self.current_val_idx + batch_size > self.Nval:
            self.restart_val_set()
        idxs = self.rand_val_idxs[self.current_val_idx:self.current_val_idx + batch_size]
        self.current_val_idx += batch_size
        X_batch, y_batch = self.load_batch(idxs)
        return {X : X_batch, y: y_batch, is_train: False}


    def load_and_split(self, data_path, labels_path, val_size=0.2, random_seed=None):
        from sklearn.model_selection import train_test_split
        self.data_path = data_path
        labels = pd.read_csv(labels_path)
        N = 50000
        self.load_images(N=N)
        self.labels = labels.T.to_dict()
        imagesNum = N
        idxs = np.arange(imagesNum) + 1
        self.train_idxs, self.val_idxs= train_test_split(idxs, test_size=val_size, random_state=random_seed) 
        self.train_idxs = np.sort(self.train_idxs)
        self.val_idxs = np.sort(self.val_idxs)
        self.Ntrain = self.train_idxs.shape[0]
        self.Nval = self.val_idxs.shape[0]
        self.restart_epoch()
        self.restart_val_set()
        return 
    
    
#%%   
def preprocess_X(X, smoothing=True, edge_detection=True, std=None, mean=None):    
    pass
#%%

def plot_a_sample(x, y=None):
    pass
#%%
def read_params(path):
    import ast
    params = {}
    with open(path) as f_in:
        lines = (line.rstrip() for line in f_in) 
        lines = list(line for line in lines if line) # Non-blank lines in a list
    for line in lines:
        if line[0] == '#':
            continue
        words = line.split()
        params[words[0]] = ast.literal_eval(words[2])
    return params
#%%

