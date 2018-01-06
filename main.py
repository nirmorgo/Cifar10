from DataHandler import *
from model.nets import *
from model.train_functions import *
from model.classifier import Classifier
from scipy import misc
import numpy as np
import pandas as pd

params = read_params('C:/Users/Nir/Documents/Python/kaggle/cifar10/params.txt')
data = Cifar10_Data()
data.load_and_split(params['input_data_path'],params['labels_path'])
#xtrain=data.train_idxs
#xval=data.val_idxs
#batch = data.get_train_feed_dict('X','y','train',128)
#%%
cls = Classifier(params, data.Ndims, net=convnet2)
cls.train(data, epochs=10, batch_size=128)
#cls.load_weights_from_checkpoint(params['pre-traind_model_path'])


#tensorboard --logdir=run:\Users\Nir\Documents\Python\kaggle\cifar10\tmp\tensorflow_logs
#%% Get The Test Data And Classify It
test_path = params['test_data_path']
labels=[]
for batch_num in range(600):
    X = []
    for idx in range(500):
        img_path = test_path + str(batch_num*500+idx+1) + '.png'
        X.append(misc.imread(img_path))
    X = np.array(X)
    X = (X-data.mean)/(data.std+1e-7)
    preds = cls.predict(X)
    preds = np.argmax(preds, axis=1)
    for i in np.arange(preds.shape[0]):
        labels.append(data.decoding[preds[i]])
    print('Classified: ',batch_num*500,'out of 300000')
output = pd.DataFrame()
output['id']=range(300000)
output['id'] = output['id'] + 1
output['label']=labels
write_path = params['project_path'] + 'submissions/save1.csv'
output.to_csv(write_path, index=False)
