import os
# import datetime
import time
#import tool
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from keras import utils

def grid_search_script(inputs,output_matrix,
                       mse_test,script_name,
                       model,
                       epoch=80,
                       Mode=(0,0,1),
                       modify_term=2,
                       validate_run=1): #位置参数放在最前，也可以将该参数直接用关键字实参传递，具体的参数顺序：位置参数、默认参数、变长参数、关键字参数、命名关键字参数。
                                                                                #可变参数：  变长参数   *args — 存放 元组 参数
                                                                                #           关键字参数  **kwargs — 存放 字典 参数，
    activation=('relu',
                'tanh',
                'sigmoid',
                'linear'
                )
    objectivation=('M',#MSE
                   'G', #Gussian
                   'R' #sparse_regular
                   )
    loss_weight=('102',
                 '103', #1000:1
                 '104',
                 'Nan'  #N:0
                 )
    img_name=activation[Mode[0]]+'_'+objectivation[Mode[1]]+'_'+loss_weight[Mode[2]]
    file_mode=('activation',
               'objective',
               'loss_weight',
               'neural_structure'
               )
    file_name=file_mode[modify_term]


    data=time.localtime()
    day=data[2]
    mon=data[1]
    def mkdir(path):
        folder = os.path.exists(path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
            print("---  new folder...  ---")
            # print("---  OK  ---")
            return path + '/'
        else:
            print("---  exist...  ---")
            # print("---  OK  ---")
            return path+'/'
    path=('./picture/file_data:%s_%s'%(mon,day))
    file_path=mkdir(path)
    #创建第二层
    script_path=mkdir(file_path+script_name)

    eph_path=mkdir(script_path+file_name)   #day/script/modify/
    #创建文件
    img_path=mkdir(eph_path+str(epoch)+'epoch') #day/script/modify/epoch

    #载入数据：
    if isinstance(inputs, list):
        x,x_ori=inputs
    else:
        raise ValueError('Need 2 inputs, please check the input')

    #需要把输出调整排序，并重新调整形状
    #画模型的结构图
    utils.plot_model(model, to_file=(eph_path+script_name+'.png'))

    #画图
    plt.rcParams['figure.figsize']=(25,8)
    plt.rcParams['savefig.dpi']=512
    plt.rcParams['figure.dpi']=512
    plt.rcParams.update({'font.size': 13})   #字体大小
    plt.figure()
    # plt.subplot(211)
    # plt.plot(x_ori[1])
    # plt.legend('Original')
    # plt.subplot(212)
    plt.title('MSE value is: '+str(mse_test))
    p1,=plt.plot(x_ori[0])
    p2,=plt.plot(x[0])
    plt.legend([p1,p2],["Original",'Reconstruction'],loc='best',fontsize='large')
    plt.savefig(img_path+img_name+'_with_'+str(epoch)+'_epoch_in_'+str(validate_run)+'_run'+'.png')
    plt.close('all')

    #save matrix
    savemat(img_path+img_name+'_with_'+str(epoch)+'_epoch_in_'+str(validate_run)+'_run_test'+'.mat',{'X_re':x})
    #save model
    model.save(img_path+img_name+'_model_weight.h5')

    if isinstance(output_matrix, list):
        source_weight, source_bias = output_matrix
        savemat(img_path+'source_activation'+'_in_'+str(validate_run)+'_run.mat', {'source_w': np.transpose(source_weight)})
        savemat(img_path + 'source_bias' + '_in_' + str(validate_run) + '_run.mat', {'source_w': np.transpose( source_bias)})
    elif isinstance(output_matrix, np.ndarray):
        source_path=img_path + img_name + '_with_' + str(epoch) + '_epoch_in_' + str(validate_run) + '_run_source' + '.mat'
        savemat(img_path + img_name + '_with_' + str(epoch) + '_epoch_in_' + str(validate_run) + '_run_source' + '.mat',
                {'X_source': output_matrix})
        with open(file_path+'path.txt',"w",encoding='utf-8') as f:
            f.write(source_path)
    elif output_matrix==None:
        pass
    else:
         raise ValueError('Need 2 inputs, please check the input')


    # savemat(img_path+"source_bias.mat",{'source_b': source_bias})

    #aim: M_relu_L103_with_80_epoch.png
