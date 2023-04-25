import numpy as np
import math
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,BatchNormalization, Conv2D, Flatten,Reshape,Lambda,Dot,Add,Concatenate,Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend

def big_scale(distances):
    path_loss = -20.4*np.log10(distances)
    # convert dB to linear
    big_scale_fading_linear = np.sqrt(np.power(10,path_loss/10))
    big_scale_fading_linear = np.expand_dims(big_scale_fading_linear,axis=-1)
    return big_scale_fading_linear    
    

def beamformer_LIS_multiuser(M,N,K,lr,total_num,sigma,Pmax,Rmin,mean,std,trainable):
    def minus_sum_rate(y_true,y_pred):
        '''
        y_true只是自定义损失函数的格式需要，实际上并没有用到
        '''
        H = y_pred[:,:2*K*M]
        W = y_pred[:,2*K*M:]
        
        
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


        HWWH_KK = tf.reshape(HWWH[:,0,0],(-1,1))

        for i in range(K-1):
            temp = tf.reshape(HWWH[:,i+1,i+1],(-1,1))
            HWWH_KK = tf.concat([HWWH_KK,temp],axis= -1)

        J = 1 + tf.reduce_sum(HWWH,-1) - HWWH_KK
        Rk = tf.reduce_sum(tf.math.log(1 + (HWWH_KK)/(J) ) / tf.cast(tf.math.log(2.0),dtype=tf.float32) ,-1)   
        loss = -Rk

        return loss

    def matrix_dot(IP):
        return backend.batch_dot(IP[0],IP[1])
    
    
    merged_inputs = Input(shape=(M*N*K*2+2*M*N+2*K*N,))
    data = Lambda(lambda x:x[:,:M*N*K*2])(merged_inputs)
    label = Lambda(lambda x:x[:,M*N*K*2:])(merged_inputs) 
    
    # extract G and hr from "label"
    G_real = Lambda(lambda x:x[:,:N*M])(label)
    G_real = Reshape((N,M))(G_real)
    G_imag = Lambda(lambda x:x[:,N*M:2*N*M])(label)
    G_imag = Reshape((N,M))(G_imag)
    hr_real = Lambda(lambda x:x[:,2*N*M:2*N*M+K*N])(label)
    hr_real = Reshape((K,N))(hr_real)
    hr_imag = Lambda(lambda x:x[:,2*N*M+K*N:])(label)
    hr_imag = Reshape((K,N))(hr_imag)
    
    # predict theta based on "data"
    #    merged_input = Reshape((M,N,K*2))(data)
    merged_input = Reshape((2,K,N,M))(data)
    merged_input = Lambda(lambda x:tf.transpose(x,(0,4,3,2,1)))(merged_input)
    merged_input = Reshape((M,N,K*2))(merged_input)
    for i in range(2): # 4层卷积，64个卷积核，kernel 4x4
	    if i==0:
		    temp = Conv2D(32,(4,4),padding='same',data_format='channels_last',activation='relu')(merged_input)
	    else:
		    temp = Conv2D(32,(4,4),padding='same',data_format='channels_last',activation='relu')(temp)
	    temp = BatchNormalization()(temp)
    temp = Flatten()(temp)
    temp = Dense(4*(N),activation='relu')(temp)
    temp = BatchNormalization()(temp)
    out_phase = Dense(N,activation='linear')(temp)
    

    theta_real = Lambda(lambda x:tf.cos(x))(out_phase)
    theta_imag = Lambda(lambda x:tf.sin(x))(out_phase)
        
    theta_real = Lambda(lambda x:tf.linalg.diag(x))(theta_real)
    theta_imag = Lambda(lambda x:tf.linalg.diag(x))(theta_imag)
    
    # compute effective channel with theta, G and hr    
    tmp1 = Lambda(matrix_dot)([hr_real,theta_real])
    tmp2 = Lambda(matrix_dot)([hr_imag,theta_imag])
    tmp2 = Lambda(lambda x:-x)(tmp2)
    tmp3 = Lambda(matrix_dot)([hr_real,theta_imag])
    tmp4 = Lambda(matrix_dot)([hr_imag,theta_real])
    
    middle_real = Add()([tmp1,tmp2])
    middle_imag = Add()([tmp3,tmp4])

    tmp5 = Lambda(matrix_dot)([middle_real,G_real])
    tmp6 = Lambda(matrix_dot)([middle_imag,G_imag])
    tmp6 = Lambda(lambda x:-x)(tmp6)
    tmp7 = Lambda(matrix_dot)([middle_real,G_imag])
    tmp8 = Lambda(matrix_dot)([middle_imag,G_real]) 

    # shape (?,K,M)
    effective_channel_real = Add()([tmp5,tmp6])
    effective_channel_imag = Add()([tmp7,tmp8])
    # shape (?,M,K)
    #effective_channel_real = Lambda(lambda x:tf.transpose(x,(0,2,1)))(effective_channel_real)
    #effective_channel_imag = Lambda(lambda x:tf.transpose(x,(0,2,1)))(effective_channel_imag)
    
    effective_channel_real = Flatten()(effective_channel_real)
    effective_channel_imag = Flatten()(effective_channel_imag)
    
    ########################################################################
    # form input "effective channel" to the second neural network     
    effective_channel = Concatenate()([effective_channel_real,effective_channel_imag])    
    
    # normalization with recorded mean and std
    effective_channel_normalized = Lambda(lambda x:(x-mean)/std)(effective_channel)

    # 注意区分effective_channel和effective_channel_normalized，一个用于计算loss，一个作为网络输入
    #    temp = Dense(32*N,activation='relu',trainable=trainable)(effective_channel_normalized)
    #    temp = BatchNormalization(trainable=trainable)(temp)
    temp = Dense(200,activation='relu',trainable=trainable)(effective_channel_normalized)
    temp = BatchNormalization(trainable=trainable)(temp)
    temp = Dense(200,activation='relu',trainable=trainable)(temp) 
    temp = BatchNormalization(trainable=trainable)(temp)
    temp = Dense(200,activation='relu',trainable=trainable)(temp) 
    temp = BatchNormalization(trainable=trainable)(temp)
    predictions = Dense(2*(M*K), activation='linear',trainable=trainable)(temp)
    
    predictions = Concatenate()([effective_channel,predictions])
    
    model = Model(inputs=merged_inputs, outputs=predictions)
    model.compile(loss=minus_sum_rate, optimizer=Adam(lr=lr))
    model.summary()
    return model


def pinv(a, rcond=1e-15):
    s, u, v = tf.linalg.svd(a)
    Zero = tf.cast(tf.sign(tf.abs(s)) - 1 , tf.complex128)
    # Ignore singular values close to zero to prevent numerical overflow
    limit = rcond * tf.reduce_max(s)
    non_zero = tf.greater(s, limit)
    s = tf.cast(s,tf.complex128)
    reciprocal = tf.where(non_zero, tf.divide(1,s), Zero)
    lhs = tf.matmul(v, tf.linalg.diag(reciprocal))
    return tf.matmul(lhs, tf.conj(u), transpose_b=True)



def power_model(M,N,K,lr,total_num,sigma,Pmax,Rmin):
    # TODO 3: how to compute loss with y_true and predicted θ in multiuser case?
    # Hint1: combine with transmit beamforming like zero-forcing 
    # Hint2: transmit power allocation for users with total power constraint
    def minus_sum_rate(y_true,y_pred):
        """ phase = y_pred  
        #print(y_pred)
        # convert real angle to complex number, e^jθ = cos(θ)+j*sin(θ)
        phase = tf.cast(phase,tf.complex128)
        theta_real = tf.cos(phase)
        #print(theta_real.shape)
        theta_imag = tf.sin(phase)
        theta = theta_real+1j*theta_imag
        theta = tf.linalg.diag(theta) """
        #print(theta)
        #theta = tf.expand_dims(theta,0)
        #print(phase)
        W = y_pred
        W = tf.cast(W, tf.complex128)
        W = W[:,:K*M]+1j*W[:,K*M:]
        W = tf.reshape(W,[-1,M,K])

        WW = tf.matmul(W,tf.transpose(W, perm=[0,2,1], conjugate = True))
        b = tf.sqrt(Pmax/tf.trace(WW))
        b = tf.tile(tf.reshape(b,(-1,1,1)),(1,M,K))
        W = b*W

        H = y_true
        H = tf.cast(H, tf.complex128)
        H = H[:,:K*M]+1j*H[:,K*M:]
        H = tf.reshape(H,[-1,K,M])
        print(W.shape, H.shape)
        print(W, H)
        HW = tf.matmul(H, W)
        WH = tf.transpose(HW, perm=[0,2,1], conjugate = True)
        print(HW.shape, WH.shape)

        HWWH = tf.square(tf.abs(HW))
        HWWH = tf.cast(HWWH,tf.float32)

        temp0 = tf.reshape(HWWH[:,0,0], (-1,1))
        temp1 = tf.reshape(HWWH[:,1,1], (-1,1))
        temp2 = tf.reshape(HWWH[:,2,2], (-1,1))
        temp3 = tf.reshape(HWWH[:,3,3], (-1,1))

        HWWH_KK = tf.concat([temp0,temp1,temp2,temp3], axis= -1)

        J = 1 + tf.reduce_sum(HWWH,-1) - HWWH_KK
        print(J.shape)
        print(HWWH_KK.shape)


        Rk = tf.reduce_sum(tf.log(1 + (HWWH_KK)/(J) ) / math.log(2.0) ,-1) 
        print(Rk.shape)
        a = tf.cast(tf.linalg.trace(WW),tf.float32)
        loss = -Rk+0.2*a
        return loss        
    
    # input layer, notice the input shape
    #merged_inputs = Input(shape=(2*(M*N*K),))
    merged_inputs = Input(shape=(2*(M*K),))
    # FC2
    temp = Dense(32*N,activation='relu')(merged_inputs)
    temp = BatchNormalization()(temp)    
    
    temp = Dense(200,activation='relu')(temp)
    # BN
    temp = BatchNormalization()(temp)
    # FC3
    temp = Dense(300,activation='relu')(temp) 
    # BN
    temp = BatchNormalization()(temp)
    # FC4
    temp = Dense(200,activation='relu')(temp) 
    # BN
    temp = BatchNormalization()(temp)
    # output layer, with dimension N, i.e., the number of reflecting elements
    out_phase = Dense(2*(M*K), activation='linear')(temp)
    model = Model(inputs=merged_inputs, outputs=out_phase)
    # compile the model with self defined loss function
    # and initial learning rate lr
    model.compile(loss=minus_sum_rate, optimizer=Adam(lr=lr))
    # print the summary of the model
    model.summary()
    return model
