import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import Model,load_model
from utils import beamformer_LIS_multiuser,big_scale,pinv,power_model
from sklearn import preprocessing
from scipy import io

#%% system parameters
# antennas
M = 8
# reflecting elements
N = 64
# user number
K = 6
# noise sigma
sigma = 1
# Pmax
Pmax = 100
#calculate the SNR in dB
SNR = Pmax / sigma
#calculate the Rmin
Rmin = math.log2(1 + SNR / (2*K)) - math.log2(1 + SNR / (2*K))
#Rmin = 0

#%% training
# training parameters
total_num = 100000
lr = 0.01      ##0.001
epochs = 1000 #1000
batch_size = 5000 #5000

# load from dataset
dataset = io.loadmat('./bl_data_8_64_6.mat')
G_0 = dataset['G_list']
G_0 = np.reshape(G_0,(N,total_num,M))
G_0 = np.transpose(G_0,(1,0,2))
# flatten
G = np.reshape(G_0,(total_num,-1))#T*MN

# load from dataset
hr_0 = dataset['Hr_list']
hr_0 = np.reshape(hr_0,(K,total_num,N))
hr_0 = hr_0[:K,:,:]
hr_0 = np.transpose(hr_0,(1,0,2))
# flatten
hr = np.reshape(hr_0,(total_num,-1))#T*KN

# compute std with random theta
theta = np.zeros([total_num,N,N],dtype=complex)
for i in range(total_num):
    theta[i,:,:] = np.diag(np.exp(1j*2*math.pi*np.random.rand(N)) )
H_0 = np.matmul( np.matmul(hr_0,theta),G_0)
print(H_0.shape)
H = np.reshape(H_0,(total_num,-1))
train_dataset = np.concatenate((np.real(H),np.imag(H)),axis=-1)
std = np.std(train_dataset)

G_0 = np.expand_dims(G_0,axis=1) #original 2 dim
hr_0 = np.expand_dims(hr_0,axis= -1)
G_0 = np.tile(G_0,(1,K,1,1))
hr_0 = np.tile(hr_0,(1,1,1,M))
Ghr_0 = G_0*hr_0
Ghr = np.reshape(Ghr_0,(total_num,-1))#T*MNK
#Ghr_0 = np.matmul(G_0,hr_0)
#Ghr = np.reshape(Ghr_0,(total_num,-1))

# feed the original channels to the model as "label", for loss computation only
train_labelset = np.concatenate((np.real(G),np.imag(G),np.real(hr),np.imag(hr)),axis=-1)
#print(train_labelset.shape)
# feed features to the model for training 
train_dataset = np.concatenate((np.real(Ghr),np.imag(Ghr)),axis=-1)
#print(train_dataset.shape)
# standarization preprocessing
scaler = preprocessing.StandardScaler().fit(train_dataset)
train_dataset = scaler.transform(train_dataset)

# TODO 1: Above is single user data generation
# how to generate multiuser channels in a graceful way

best_model_path = './models/best_%d_and_%d_and_%d.h5'%(M,N,K)
# always keep the best model so far, save_weights_only requires less memory
checkpointer = ModelCheckpoint(best_model_path,verbose=1,save_best_only=True,save_weights_only=True)
# reduce learning rate to 1/3 when validation loss doesn't decrease in 10 epochs
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.333, patience=10, verbose=1, mode='auto',min_delta=0.0001,min_lr=0.00001)
# stop training when validation loss doesn't decrease in 20 epochs
early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=20)

# initial model
mean = 0
# true联合训练，false只训练前半个网络
trainable = True
model = beamformer_LIS_multiuser(M,N,K,lr,total_num,sigma,Pmax,Rmin,mean,std,trainable)


#model_p = power_model(M,N,K,lr,total_num,sigma,Pmax,Rmin)
#model_p.load_weights("./power_%d_and_%d_and_%d.h5"%(M,N,K))
#weights = []
#for layer in model_p.layers:
#    weights.append(layer.get_weights())
#
#for i in range(48,54):
#    print(model.layers[i])
#    model.layers[i].set_weights(weights[i-47])


# 其实train_labelset没有用到
train_dataset = np.concatenate([train_dataset,train_labelset],axis=-1)
#model.load_weights(best_model_path)

model.fit(train_dataset,train_labelset,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,validation_split=0.2,callbacks=[checkpointer,early_stopping,reduce_lr])



#%% fetch theta
phase_layer_model = Model(inputs=model.input,outputs=model.layers[13].output)
predict_phase = phase_layer_model.predict(train_dataset)
dl_theta = np.cos(predict_phase)+1j*np.sin(predict_phase)
io.savemat('dl_theta.mat',{'dl_theta':dl_theta})


#%% load bl theta
#bl_theta = io.loadmat('bl_theta.mat')['BL_theta_list']

#bl_theta = np.exp(1j*2*np.pi*np.random.rand(32,103))
#
#effective_channel_list = np.zeros((103,2*N))
#for i in range(len(bl_theta)):
#    effective_channel = (hr_0[i].dot(np.diag(bl_theta[:,i]))).dot(G_0[i])
#    effective_channel = np.reshape(effective_channel,-1)
#    effective_channel_list[i,:N] = np.real(effective_channel)
#    effective_channel_list[i,N:] = np.imag(effective_channel)
#predict = effective_channel_list

"""
#%% testing
predict = model.predict(train_dataset)
predict = tf.convert_to_tensor(predict, dtype=tf.complex128)
H =predict[:,:2*K*M]
W =predict[:,2*K*M:]


W = tf.cast(W, tf.complex128)
W = W[:,:K*M]+1j*W[:,K*M:]
W = tf.reshape(W,[-1,M,K])
WW = tf.matmul(W,tf.transpose(W, perm=[0,2,1], conjugate = True))
b = tf.sqrt(Pmax/tf.linalg.trace(WW))
b = tf.tile(tf.reshape(b,(-1,1,1)),(1,M,K))
W = b*W

H = tf.cast(H, tf.complex128)
H = H[:,:K*M]+1j*H[:,K*M:]
H = tf.reshape(H,[-1,K,M])
#print(W.shape, H.shape)
#print(W, H)
HW = tf.matmul(H, W)
WH = tf.transpose(HW, perm=[0,2,1], conjugate = True)
#print(HW.shape, WH.shape)
HWWH = tf.square(tf.abs(HW))
HWWH = tf.cast(HWWH,dtype=tf.float32)
temp0 = tf.reshape(HWWH[:,0,0], (-1,1))
temp1 = tf.reshape(HWWH[:,1,1], (-1,1))
temp2 = tf.reshape(HWWH[:,2,2], (-1,1))
temp3 = tf.reshape(HWWH[:,3,3], (-1,1))

HWWH_KK = tf.concat([temp0,temp1,temp2,temp3], axis= -1)
J = 1 + tf.reduce_sum(HWWH,-1) - HWWH_KK
Rk = tf.reduce_sum(tf.log(1 + (HWWH_KK)/(J) ) / tf.cast(tf.log(2.0),dtype=tf.float32) ,-1) 

with tf.Session():
    print("tf.rate   ", tf.reduce_sum(Rk).eval()/total_num)
    #print("trace(WW)    ", tf.trace(tf.matmul(W,tf.transpose(W, perm=[0,2,1], conjugate = True))).eval())


# zero forcing precoding
layer_model = Model(inputs=model.input, outputs=model.layers[46].output)
predict = layer_model.predict(train_dataset[:]) # 103
predict = tf.convert_to_tensor(predict, dtype=tf.complex128)
H =predict
H = H[:,:K*M]+1j*H[:,K*M:]
X = tf.reshape(H,[-1,K,M])

Y = tf.transpose(X, perm=[0,2,1], conjugate = True)#矩阵共轭
X_0 = tf.matmul(X,Y)#乘法
X_0_inv = pinv(X_0)
X_0_diag = tf.linalg.diag_part(X_0_inv)
lumbda = tf.cast(tf.divide(1,X_0_diag),tf.float32)
# no sort, wrong dimension
lumbda = tf.sort(lumbda, direction='ASCENDING')
q = tf.reduce_sum(tf.sign(lumbda),-1,keepdims=True)
ii = tf.sign(lumbda)-1
temp4 = tf.sign(lumbda)
V = tf.sign(lumbda)-1
Trans = []
for i in range(1,K+1):
    Trans.append(float(i))
#Trans = [1.,2.,3.,4.]
Trans = tf.expand_dims(Trans,0)
O = tf.sign(lumbda)
Z = tf.sign(lumbda) - 1
for i in range(K):
    ii = ii + 1
    V_lum = tf.abs(V - 1)
    lumbda_1 = tf.divide(1,lumbda) * V_lum
    temp1 = 0
    temp2 = tf.reduce_sum(sigma*lumbda_1, -1,keepdims=True)
    alpha = (Pmax - temp1 + temp2) / tf.cast(q,tf.float32)
    temp4 = alpha*lumbda - sigma
    temp4 = temp4/2 * (tf.sign(temp4)+1.0)
    judge_V = tf.equal( tf.sign(temp4) , V) #temp4 0 -> judge true
    V = tf.where(judge_V,O,V)  #judge flase V judge true 1
    V_temp = V * Trans
    V_temp_judge = tf.greater(V_temp,ii)
    V = tf.where(V_temp_judge,Z,V)
    judge_q = tf.reduce_any(judge_V,-1,keepdims=True)
    q = tf.where(judge_q,q-1,q)
pk = temp4
reward = tf.reduce_sum(tf.math.log(1 + (pk)/(sigma) ) / math.log(2), -1)
with tf.Session():
    print("zf rate   ", tf.reduce_sum(reward).eval()/total_num)


#%% np version of ZF
#layer_model = Model(inputs=model.input, outputs=model.layers[46].output)
#predict = layer_model.predict(train_dataset[:]) # 103
##predict = tf.convert_to_tensor(predict, dtype=tf.complex128)
#H = predict
#H = H[:,:K*M]+1j*H[:,K*M:]
#X = np.reshape(H,[-1,K,M])
#
#Y =np.conjugate(np.transpose(X, (0,2,1)))#矩阵共轭
#for j in range(len(X)):
#    X_0 = X[j].dot(Y[j])#乘法
#    X_0_inv = np.linalg.inv(X_0)
#    X_0_diag = np.diag(X_0_inv)
#    lumbda = np.real(1/X_0_diag)
#    lumbda = np.sort(lumbda)
#    q = np.sum(np.sign(lumbda))
#    ii = np.sign(lumbda)-1
#    temp4 = np.sign(lumbda)
#    V = np.sign(lumbda)-1
#    Trans = []
#    for i in range(1,K+1):
#        Trans.append(float(i))
#
#    Trans = np.expand_dims(Trans,0)
#    O = np.sign(lumbda)
#    Z = np.sign(lumbda) - 1
#    for i in range(K):
#        ii = ii + 1
#        V_lum = np.abs(V - 1)
#        lumbda_1 = 1/lumbda * V_lum
#        temp1 = 0
#        temp2 = np.sum(sigma*lumbda_1)
#        alpha = (Pmax - temp1 + temp2) / q
#        temp4 = alpha*lumbda - sigma
#        temp4 = temp4/2 * (np.sign(temp4)+1.0)
#        judge_V = np.equal( np.sign(temp4) , V) #temp4 0 -> judge true
#        V = np.where(judge_V,O,V)  #judge flase V judge true 1
#        V_temp = V * Trans
#        V_temp_judge = np.greater(V_temp,ii)
#        V = np.where(V_temp_judge,Z,V)
#        #??
#        judge_q = tf.reduce_any(judge_V,-1,keepdims=True)
#        q = np.where(judge_q,q-1,q)
#    pk = temp4
#    reward = tf.reduce_sum(tf.log(1 + (pk)/(sigma) ) / math.log(2), -1)
"""
