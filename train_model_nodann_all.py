# -*- coding: utf-8 -*-
# @Time    : 2021/7/25 上午1:43
# @Author  : Dai Pu wei
# @Email   : 771830171@qq.com
# @File    : train_MNIST2MNIST_M.py
# @Software: PyCharm


import os
import pickle

import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from config.config import config
from utils.dataset_utils import batch_generator,split_dataset
from model.MNIST2MNIST_M_train import MNIST2MNIST_M_DANN

'''
gpus = [tf.config.experimental.list_physical_devices(device_type='GPU')[1]]
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
'''

def run_main():
    """
       这是主函数
    """
    # 初始化参数配置类
    print("Hello world")
    seq_len = 1664
    portion = 1
    seqencetonum = True
    Dann = True
    #input_shape = (seq_len,)
    #image_size = 28
    init_learning_rate = 3e-2
    momentum_rate = 0.9
    batch_size = 64
    #epoch = 200
    
    finetune_model_path = os.path.join(os.path.abspath(os.getcwd()),"model_add0.3direct_newati_nogcc","0EEEpoch006-train_loss-0.233-val_loss-0.895-domain_loss-0.000-train_cls_acc-0.906-val_cls_acc-0.658-domain_cls_acc-0.000.h5")
    finetune = True

    embed_model_path = os.path.abspath(os.getcwd())+"/embedding/CNN_w2v_C.h5"
    checkpoints_dir = os.path.abspath(os.getcwd())+"/checkpoints/models/SGD-lr={0}-momentum={1}/batch_size={2}".format(init_learning_rate,momentum_rate,batch_size)
    if(os.path.exists(checkpoints_dir)):
        print('yes')
    logs_dir = os.path.abspath(os.getcwd())+"/logs/new/SGD-lr={0}-momentum={1}/batch_size={2}".format(init_learning_rate,momentum_rate,batch_size)
    config_dir = os.path.abspath(os.getcwd())+"/config/models/SGD-lr={0}-momentum={1}/batch_size={2}".format(init_learning_rate,momentum_rate,batch_size)
    
    source_dataset_path = "./data_addnew0.5_again/C_test_positive_addnew.pkl"#./indirect+0.3direct_segbybinary/C_test_positive_adddirect.pkl"
    target_dataset_path = "./data_segbybinary_test/C_test_positive.pkl"
    
    cfg = config(embed_model_path = embed_model_path,
                 finetune_model_path = finetune_model_path,
                 finetune=False,
                 dir_name =os.path.abspath(os.path.abspath(os.getcwd())+"/model_addnew0.5_again_tranformer"),
                 checkpoints_dir = checkpoints_dir,
                 logs_dir = logs_dir,
                 config_dir = config_dir,
                 input_shape = (int(seq_len)+1,),
                 seq_len = seq_len,
                 init_learning_rate = init_learning_rate,
                 momentum_rate= momentum_rate,
                 batch_size=batch_size,
                 epoch = 10,
                 portion = portion,
                 Dann = False
                 )

    # Load MNIST
    # data_dir,seq_len,p,l
    if(seqencetonum):
        x_train,y_train,x_val,y_val = split_dataset(source_dataset_path,cfg.seq_len,cfg.portion,0)
        target_x_train, target_y_train, target_x_val, target_y_val = split_dataset(target_dataset_path,cfg.seq_len,cfg.portion,1)
    else:
        source = pkl.load(open(source_dataset_path, 'rb'))
        x_train=np.array(source['0'][:int(0.9*len(source['0']))])
        x_val=np.array(source['0'][int(0.9*len(source['0'])):])
        y_train = np.array(source['1'][:int(0.9*len(source['1']))])
        y_val = np.array(source['1'][int(0.9*len(source['1'])):])
       

        # Load target
        target = pkl.load(open(target_dataset_path, 'rb'))
        target_x_train = target['0'][:int(0.9*len(target['0']))]
        target_x_val = target['0'][int(0.9*len(target['0'])):]
	
    # 构造数据生成器
    #print(np.array([x_train, to_categorical(y_train)]).shape)
    print('.....',len(x_train),'.....')
    train_source_datagen = batch_generator([x_train, to_categorical(y_train)],cfg.batch_size // 2)
    train_target_datagen = batch_generator([target_x_train, to_categorical(target_y_train)],cfg.batch_size // 2)
    val_target_datagen = batch_generator([target_x_val, to_categorical(target_y_val)],cfg.batch_size)

    # 初始化每个epoch的训练次数和每次验证过程的验证次数
    train_source_batch_num = int(len(x_train)/(cfg.batch_size//2))
    train_target_batch_num = int(len(target_x_train)/(cfg.batch_size//2))
    train_iter_num = int(np.max([train_source_batch_num,train_target_batch_num]))
    val_iter_num = int(len(x_val)/cfg.batch_size)

    # 初始化DANN，并进行训练
    model = MNIST2MNIST_M_DANN(cfg)
    model.train(train_source_datagen,train_target_datagen,val_target_datagen,train_iter_num,val_iter_num)
    #model.save_weights('dann_ICR.h5')
    #model.evaluate(target_x_train,target_y_train)


def gen_pkl(file_name,st):
    b = pkl.load(open(file_name, 'rb'))
    length = len(b[0])

    row_size = 256
    row = length // row_size

    c = {}
    bytes = []
    labels = []
    for i in range(row):
        s0 = b[0][i * row_size:(i * row_size + row_size)]
        s1 = b[1][i * row_size:(i * row_size + row_size)]
        bytes.append(s0)
        labels.append(s1)
    c['0'] = bytes
    c['1'] = labels
    if st == 0:
        pkl.dump(c, open('source.pkl', 'wb'))
    else:
        pkl.dump(c, open('target.pkl', 'wb'))
'''
Dann = '1'

#open('lighttpd-O1.pkl', 'wb')
ss_32=['ar-clangO2-m32', 'diff3-clangO2-m32', 'readelf-clangO2-m32']
tt_64=['fzsftp-clangO2-64', 'cmp-clangO2-64', 'hmmer-clangO2-64']
s_name = 'ar-clangO2-m32'
t_name = 'fzsftp-clangO2-64'
'''
if __name__ == '__main__':
    # file=open('train.pkl','rb')
    # a=pkl.load(file)
    # #print(a)
    # a1000=a[0][len(a[0])-1]
    # b1000=a[1][len(a[1])-1]
    #
    '''
    s_name = 'cmp-clangO2-64'
    t_name = 'diff3-clangO2-m32'
    gen_pkl(s_name + '.pkl', 0)
    gen_pkl(t_name + '.pkl', 1)
    s = pkl.load(open('source.pkl', 'rb'))
    t = pkl.load(open('target.pkl', 'rb'))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    '''
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_main()
    # for i in ss_32:
    #     s_name = i
    #     for j in tt_64:
    #         t_name=j
    #         gen_pkl(s_name + '.pkl', 0)
    #         gen_pkl(t_name + '.pkl', 1)
    #         s = pkl.load(open('source.pkl', 'rb'))
    #         t = pkl.load(open('target.pkl', 'rb'))
    #         run_main()
    #         Dann = '1'
    #         run_main()

