import os
os.environ['THEANO_FLAGS'] = "device=gpu"
os.environ['CUDA_VISIBLE_DEVICES']='1'

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from sklearn import preprocessing
from keras.layers import Input, Dense,Flatten, Reshape,Lambda,MaxPooling2D,Dropout,UpSampling2D
# Reshape 函数可以用来改变张量
from keras.layers import Conv2D, Conv2DTranspose,Conv3D
from keras.models import Model
from keras import backend
from keras.models import Sequential
from numpy import random
from keras import optimizers

from mkdir_script import grid_search_script

import matlab.engine
eng=matlab.engine.start_matlab()
from keras import metrics
#修改：
from keras.callbacks import TensorBoard

#plot toolboxs
import pandas as pd
from pandas import Series,DataFrame
# import plotly.plt as py
import cufflinks as cf
# load data
eeg=loadmat('./data/eeg_all_same_scale2.mat')
# fmri=loadmat('./data/fmri_averige_non.mat')#噪音一致
fmri=loadmat('./data/fmri_svd_long.mat')
# fmri=loadmat('./data/fmri_averige.mat')
eeg_ori=loadmat('./original/X_all_same_scale2.mat')


x_ori=eeg_ori['X_ori']
X_test=eeg['eeg_test']
X_train=eeg['eeg_train']
Y_test=fmri['fmri_test']
Y_train=fmri['fmri_train']
# scale
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
Y_train = min_max_scaler.fit_transform(Y_train)
Y_test = min_max_scaler.transform(Y_test)
# This estimator scales and translates each feature individually
# such that it is in the given range on the training set

import sys
script_name=os.path.basename(sys.argv[0]).split(".")[0]

# hyper-parameters for neural network
fine_tune_mode=(0,1,1) #0:relu 1:tanh 2:sigmoid
                       #0:MSE 1:Gussian
                       #0:10^2 1:10^3 2:10^4 3:Nan
switch_item=0  #0:activation 1:objectivation 2:loss_weight 3:neural_structure
training=1
maxiter =70
# maxiter =150
nb_epoch = 1
batch_size = 10
time_filter=3
spat_filter=63
filt=48

#Z-dimention
K = 6
C = 5

#information about signals
[numTrn,chan,tmc,]=X_train.shape
X_train = X_train.reshape([X_train.shape[0],chan, tmc,1])
X_test = X_test.reshape([X_test.shape[0],chan, tmc,1])

print (X_train.shape)
print (Y_train.shape)
print (X_test.shape)
print (Y_test.shape)

# eeg的数据维度
D1 = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
# fmri 的数据维度
D2 = Y_train.shape[1]

#np.random.seed(1000)
numTrn=X_train.shape[0]
numTest=X_test.shape[0]

#hyper-parameters for bayesian model
tau_alpha = 1
tau_beta = 1
eta_alpha = 1
eta_beta = 1
gamma_alpha = 1
gamma_beta = 1

Beta = 1 # Beta-VAE for Learning Disentangled Representations
rho=0.1  # posterior regularization parameter
k=10     # k-nearest neighbors
t = 10.0 # kernel parameter in similarity measure
samplingnum = 100   # Monte-Carlo sampling

# X to X` neural network
if backend.image_data_format() == 'channels_first':
    original_eeg_size = (1, chan, tmc)
elif backend.image_data_format()=='channels_last':
    original_eeg_size = (chan, tmc, 1)

#input
X=Input(shape=original_eeg_size,name='X')
Y = Input(shape=(D2,),name='Y')
Y_mu = Input(shape=(D2,),name='Y_mu')
Y_lsgms = Input(shape=(D2,),name='Y_lsgms') #每个维度相互独立下，从而可以使得只需要求每个特征的方差。不用协方差矩阵
#encoder
cnnencoder=Sequential((
        Conv2D(64,kernel_size=(1,time_filter),padding='same',
                 # activation='relu',
                 strides=1,
                 name='conv1',
                 input_shape=original_eeg_size
               ),
        Dropout(0.6),
        Conv2D(filters=128,kernel_size=(chan,1),
                 strides=1,padding='valid',
                 activation='relu',
                 name='conv4'
               ),
        MaxPooling2D((1,2)),

        Conv2D(filters=128,kernel_size=(1,time_filter),
                 strides=(1,2),padding='same',
                 activation='relu',
                 name='conv3'
               ),
        Dropout(0.6),

        Flatten(),
        Dense(K,activation='relu'),
))
cnnencoder.summary()
#decoder
cnndecoder=Sequential((
             Dense(1280,input_shape=(K,)),
             Reshape([1,10,128]),

             Conv2DTranspose(128,kernel_size=(1,time_filter),strides=(1,2),padding='same',
                         activation='relu',bias_initializer='zeros',
                          name='deconv2'),
             UpSampling2D((1,2)),
             Conv2DTranspose(filters=128, kernel_size=(chan, 1), strides=1, padding='valid',
                             activation='relu', bias_initializer='zeros',
                             name='deconv1',
                             ),

             Conv2DTranspose(64,kernel_size=(1,time_filter),
                        activation='relu',bias_initializer='zeros',
                        strides=1,padding='same',name='deconv4'),

))
cnndecoder.summary()

#latent variable Z
Z_mu=Dense(K,name='z_u')(cnnencoder(X))
Z_lsgms=Dense(K,name='z_lgvr')(cnnencoder(X))
def reparameter(args):
    z_u,z_logvar=args
    #generate a N*K random array
    epsilon=backend.random_normal(shape=(backend.shape(z_u)[0],K),mean=0.,stddev=1.0)
                       #这里标准差和噪声应该是做了一个点乘
    return z_u+backend.exp(z_logvar/2)*epsilon
Z=Lambda(reparameter,output_shape=(K,))([Z_mu,Z_lsgms])
X_u_output=Conv2DTranspose(1, kernel_size=(1, 2),
                    activation='tanh',
                    strides=1, padding='valid', name='x_u')
X_v_output=Conv2DTranspose(1,kernel_size=(1,2),
                            activation='relu',
                            strides=1,padding='valid',name='x_var')
X_u=X_u_output(cnndecoder(Z))

X_lsgm=X_v_output(cnndecoder(Z))

# loss function
logc=np.log(2*np.pi)
def log_normalX(x,xu,lsgm):
    xu = backend.flatten(xu)
    lsgm=backend.flatten(lsgm)
    x_r=backend.flatten(x)
    return backend.mean(-0.5*logc-0.5*lsgm-0.5*((x_r-xu)**2/backend.exp(lsgm)),axis=-1)
def Y_normal_logpdf(y, mu, lsgms):
    return backend.mean(-(0.5 * logc + 0.5 * lsgms) - 0.5 * ((y - mu)**2 / backend.exp(lsgms)), axis=-1)
# loss function
Lp=0.5 * backend.mean(1 + Z_lsgms - Z_mu ** 2 - backend.exp(Z_lsgms), axis=-1)
Lx=log_normalX(X,X_u,X_lsgm)
Ly = Y_normal_logpdf(Y, Y_mu, Y_lsgms)
# L=backend.mean(Lx*10000+Lp+Ly)
L=backend.mean(Lx*1000+Lp+500*Ly)
vae_loss=-L

# compile DGMM
DGMM=Model(inputs=[X, Y, Y_mu, Y_lsgms],outputs=[X_u,X_lsgm])
opt_method = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
DGMM.add_loss(vae_loss)
DGMM.compile(optimizer=opt_method)
DGMM.summary()

# compile encoder
encoder = Model(inputs=X, outputs=[Z_mu,Z_lsgms])
# compile decoder
# decoder=Model(input=Z,output=)
Z_predict = Input(shape=(K,),name='z_predict')
X_mu_predict=X_u_output(cnndecoder(Z_predict))
eegreconstruct = Model(inputs=Z_predict, outputs=X_mu_predict)

for run in range(training):
    # Initialization
    Z_mu = np.mat(random.random(size=(numTrn,K)))
    B_mu = np.mat(random.random(size=(K,D2)))
    R_mu = np.mat(random.random(size=(numTrn,C)))
    sigma_r = np.mat(np.eye((C)))
    H_mu = np.mat(random.random(size=(C,D2)))
    sigma_h = np.mat(np.eye((C)))

    tau_mu = tau_alpha / tau_beta
    eta_mu = eta_alpha / eta_beta
    gamma_mu = gamma_alpha / gamma_beta

    Y_mu = np.array(Z_mu * B_mu + R_mu * H_mu)   #理论上是期望fMRI与EEG样本数是一致的
    Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2)))

    savemat('data.mat', {'Y_train':Y_train,'Y_test':Y_test})
    S=np.mat(eng.calculateS(k, t))

    # Loop training
    L_record=[]
    for l in range(maxiter):
        print ('**************************************************iter=', l)
        # update Z
        hist=DGMM.fit([X_train, Y_train, Y_mu, Y_lsgms],
                shuffle=True,
                verbose=2,
                epochs=nb_epoch,
                batch_size=batch_size,
                # validation_data=(X_test, None),
                callbacks=[TensorBoard(log_dir='./logs')])
        #等于说每次循环只做一个epoch的拟合优化
       # print(hist.history['loss'])
        L_record = np.append(L_record,hist.history['loss'])
        [Z_mu,Z_lsgms] = encoder.predict(X_train)
        print('Z_mu=',np.mean(Z_mu,0))
        print('Z_var=',np.mean(Z_lsgms,0))
        Z_mu = np.mat(Z_mu)
        # update B
        temp1 = np.exp(Z_lsgms)
        temp2 = Z_mu.T * Z_mu + np.mat(np.diag(temp1.sum(axis=0)))
        temp3 = tau_mu * np.mat(np.eye(K))
        sigma_b = (gamma_mu * temp2 + temp3).I
        B_mu = sigma_b * gamma_mu * Z_mu.T * (np.mat(Y_train) - R_mu * H_mu)
        # update H
        RTR_mu = R_mu.T * R_mu + numTrn * sigma_r
        sigma_h = (eta_mu * np.mat(np.eye(C)) + gamma_mu * RTR_mu).I
        H_mu = sigma_h * gamma_mu * R_mu.T * (np.mat(Y_train) - Z_mu * B_mu)
        # update R
        HHT_mu = H_mu * H_mu.T + D2 * sigma_h
        sigma_r = (np.mat(np.eye(C)) + gamma_mu * HHT_mu).I
        R_mu = (sigma_r * gamma_mu * H_mu * (np.mat(Y_train) - Z_mu * B_mu).T).T
        # update tau
        tau_alpha_new = tau_alpha + 0.5 * K * D2
        tau_beta_new = tau_beta + 0.5 * ((np.diag(B_mu.T * B_mu)).sum() + D2 * sigma_b.trace())
        tau_mu = tau_alpha_new / tau_beta_new
        tau_mu = tau_mu[0,0]
        # update eta
        eta_alpha_new = eta_alpha + 0.5 * C * D2
        eta_beta_new = eta_beta + 0.5 * ((np.diag(H_mu.T * H_mu)).sum() + D2 * sigma_h.trace())
        eta_mu = eta_alpha_new / eta_beta_new
        eta_mu = eta_mu[0,0]
        # update gamma
        gamma_alpha_new = gamma_alpha + 0.5 * numTrn * D2
        gamma_temp = np.mat(Y_train) - Z_mu * B_mu - R_mu * H_mu
        gamma_temp = np.multiply(gamma_temp, gamma_temp)
        gamma_temp = gamma_temp.sum(axis=0)
        gamma_temp = gamma_temp.sum(axis=1)
        gamma_beta_new = gamma_beta + 0.5 * gamma_temp
        gamma_mu = gamma_alpha_new / gamma_beta_new
        gamma_mu = gamma_mu[0,0]
        # calculate Y_mu
        Y_mu = np.array(Z_mu * B_mu + R_mu * H_mu)                        #可以理解成Y是由q(z)采样而来，对其的期望采用蒙特卡尔采样的方式，并且采样点为1
        Y_lsgms = np.log(1 / gamma_mu * np.ones((numTrn, D2)))

    # evaluate the reconstruction signals from X to X`
    X_scalp_test=DGMM.predict([X_test, Y_test, Y_mu, Y_lsgms])  #用的是训练集生成的Y_u、Y_var
    X_scalp_train=DGMM.predict([X_train[0:60], Y_train[0:60], Y_mu, Y_lsgms])

    savemat("X_train_dgmm.mat",{'X_re':X_scalp_test})
    savemat("X_test_dgmm.mat",{'X_re':X_scalp_train})
    #检验重构的MSE
    def mse_evaluate(x,x_re):
        mse=np.mean((x-x_re)**2)
        return mse

    # mse_test=mse_evaluate(X_test,X_vae_out)
    print('test MSE of model from X to X = ',mse_evaluate(X_test,X_scalp_test))
    print('train MSE of model from X to X = ',mse_evaluate(X_train[0:60],X_scalp_test))
    #重构的效果不错，现在检查一下从Y预测X的效果


    # reconstruct X (eeg) from Y (fmri)
    if backend.image_data_format() == 'channels_first':
        X_reconstructed_mu = np.zeros((numTest, 1, chan, tmc))
    else:
        X_reconstructed_mu = np.zeros((numTest, chan, tmc,1))

    HHT = H_mu * H_mu.T + D2 * sigma_h
    Temp = gamma_mu * np.mat(np.eye(D2)) - (gamma_mu**2) * (H_mu.T * (np.mat(np.eye(C)) + gamma_mu * HHT).I * H_mu)
    for i in range(numTest):
         s=S[:,i]

         z_sigma_test = (B_mu * Temp * B_mu.T + (1 + rho * s.sum(axis=0)[0,0]) * np.mat(np.eye(K)) ).I
         vv=(np.mat(Y_test)[i, :]).T
         z_mu_test = (z_sigma_test * (B_mu * Temp * (np.mat(Y_test)[i,:]).T + rho * np.mat(Z_mu).T * s )).T
         if backend.image_data_format() == 'channels_first':
             temp_mu = np.zeros((1, 1, chan, tmc))
         else:
             temp_mu = np.zeros((1,chan, tmc,1))

         epsilon_std = 1
         for l in range(samplingnum):
            epsilon=np.random.normal(0,epsilon_std,1)  # 最后一维表示维度
            z_test = z_mu_test + np.sqrt(np.diag(z_sigma_test))*epsilon
            x_reconstructed_mu = eegreconstruct.predict(z_test, batch_size=1)#这里指定一次输出1个预测值
            temp_mu = temp_mu + x_reconstructed_mu

         temp_mu= temp_mu / samplingnum
         # x_reconstructed_mu = temp_mu / samplingnum
         X_reconstructed_mu[i,:,:,:] = temp_mu
    mse_test=mse_evaluate(X_test,X_reconstructed_mu)
    print('test MSE of model from Y to X = ',mse_evaluate(X_test,X_reconstructed_mu))#区分大小写
    savemat('X_re_from_y.mat',{'X_re':X_reconstructed_mu})

# 调整形状
    X_reconstructed_mu = X_reconstructed_mu.reshape([numTest, chan, tmc])
    x=np.transpose(X_reconstructed_mu,(1,0,2))   #注意matlab和numpy的permute不太一样。
    x=np.reshape(x,[chan,numTest*tmc])

    grid_search_script(inputs=[x,x_ori],mse_test=mse_test,
                        output_matrix=None,script_name=script_name,
                       model=DGMM,
                        epoch=maxiter, Mode=fine_tune_mode, modify_term=switch_item,
                        validate_run=run+1
                       )



