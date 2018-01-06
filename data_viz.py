from DataHandler import *
params = read_params('C:/Users/Nir/Documents/Python/kaggle/icebergs/params.txt')
data = Iceberg_Data()
data.load_json_and_split(data_path=params['input_data_path'], val_size=0.5, preprocess=False)
X,Y = data.X_train, data.y_train
#%%
i = 1
x, y = X[i], Y[i]
band_1 = x[:,:,0]
band_2 = x[:,:,1]
mean = x[:,:,2]
plt.figure(figsize=(16,8))
plt.subplot(1,3,1)
plt.imshow(band_1)
plt.title('band 1')
plt.subplot(1,3,2)
plt.title('band 2')
plt.imshow(band_2)
plt.subplot(1,3,3)
plt.title('mean')
plt.tight_layout()
plt.imshow(mean)
if y[0] == 1:
    title = 'A Ship!'
elif y[1] == 1:
    title = 'Icberg!'
else:
    title = 'A sample'
plt.suptitle(title, fontsize=16)
#%%
from scipy import signal
# filters
xder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
yder = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
smooth = np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9
#%%
# Plot band_1
x_modified = np.zeros_like(X)
fig = plt.figure(1,figsize=(15,15))
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    arr = signal.convolve2d(X[i,:,:,1],smooth,mode='same', boundary='symm')
    x_modified[i,:,:,1] = arr
    ax.imshow(arr,cmap='inferno')
    ax.set_title('Smoothed')
plt.tight_layout()    
plt.show()
#%%
# total derivative
fig = plt.figure(1,figsize=(15,15))
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    arr_y = signal.convolve2d(x_modified[i,:,:,1],yder,mode='same', boundary='symm')
    arr_x = signal.convolve2d(x_modified[i,:,:,1],xder,mode='same', boundary='symm')
    der_magnitude = np.hypot(arr_x,arr_y)
    ax.imshow(der_magnitude,cmap='inferno')
    ax.set_title('Smoothed')
plt.tight_layout()    
plt.show()