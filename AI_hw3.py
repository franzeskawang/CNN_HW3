import os
import glob
import time
import numpy as np
import tensorflow as tf
from skimage import io, transform

#讀取圖片
def read_img(root='/training/', weight=128,height=128):
    imgs = []
    labels = []
    
    
    print('Start read the image ...')#
    
    for i in range(10):#
        class_path = root + "Sample{:0>3d}/".format(i+1)
        print(class_path)#
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = io.imread(img_path)
            img = transform.resize(img, (weight, height,1))
            label=[0]*10
            label[i]=1
            imgs.append(img)
            labels.append(label)
            
    print('Finished ...')#
    imgss=np.asarray(imgs, np.float32)
    labelss=np.asarray(labels, np.float32)

    return imgss, labelss

# 打亂順序
def messUpOrder(data, label):
    num_example = data.shape[0]#總共幾筆資料
    print("data.shape:[0]",data.shape[0])#
    new_arange = np.arange(num_example)
    np.random.shuffle(new_arange)
    data = data[new_arange]
    label = label[new_arange]

    return data, label


def parametric_relu(_x):#拿來取代activation=tf.nn.relu
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg
# 構建網絡
def buildCNN(weight=128, height=128, color=1, mode=False):
    print("CNN Structure")
    # 佔位符
    x = tf.placeholder(tf.float32, shape=[None, weight, height, color], name='x')#訓練資料
    y_ = tf.placeholder(tf.int32, shape=[None, 10], name='y_')#10:label#對的label

    # 第一個卷積層 + 池化層
    # Input Tensor Shape: [batch_size, 128, 128, 1]
    # Output Tensor Shape: [batch_size, 128, 128, 32]
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=parametric_relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    print("conv1.shape",conv1.shape)
    # Input Tensor Shape: [batch_size, 128, 128, 32]
    # Output Tensor Shape: [batch_size, 64, 64, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    print("pool1.shape",pool1.shape)
    
    # 第二個卷積層 + 池化層
    # Input Tensor Shape: [batch_size, 64, 64, 32]
    # Output Tensor Shape: [batch_size, 64, 64, 32]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    print("conv2.shape",conv2.shape)
    # Input Tensor Shape: [batch_size, 64, 64, 32]
    # Output Tensor Shape: [batch_size, 32, 32, 32]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print("pool2.shape",pool2.shape)


    
    # 全連接層
    flat = tf.reshape(pool2, [-1, 32*32*32])
    dense1 = tf.layers.dense(inputs=flat,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    print("dense1.shape",dense1.shape)

    
    # dropout
    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode)
    print("dropout.shape",dropout.shape)
    # logits訓練出來的各種機率
    logits = tf.layers.dense(inputs=dropout,
                             units=10,  
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))


    #loss
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))   
    
    #OP
    train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
    
    #loss2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
    #train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    print("logits.shape",logits.shape)
    print("y_.shape",y_.shape)
    
    return logits, x, y_, loss, train_op

# 定義一個函數，按批次取數據
def generate_batch(inputs, targets, batch_size):
    
    sup=len(inputs)
    assert batch_size < sup, 'batch_size太大了'
    
    inputs_data=[]
    targets_data = []
    r = np.random.choice(range(sup), batch_size, replace=False)
    for i in r:
        inputs_data.append(inputs[i])  
        targets_data.append(targets[i])  
    return inputs_data, targets_data

# generate_batch test




def train(astring,root='./training/'):
    data, label = read_img(root=root, weight=128,height=128)
    x_train, y_train = messUpOrder(data=data, label=label)
    test_data, test_label = read_img(root='./validation/', weight=128,height=128)
    x_test, y_test = messUpOrder(data=data, label=label)
 
    logits, x, y_, loss, train_op = buildCNN(mode=True)
    pred=tf.argmax(logits,axis=1)
    gt=tf.argmax(y_,axis=1)
    #correct_prediction = tf.equal(pred, gt)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    n_epoch = 81
    batch_size = 50
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # Initializing all variables
        tf.initialize_all_variables().run()

        for epoch in range(n_epoch):
            batch_inputs, batch_labels = generate_batch(inputs=x_train, targets=y_train, batch_size=batch_size)
            #training
            a,b,c,d,e,f,g = sess.run([logits,x, y_,loss,pred,gt,train_op], feed_dict={x: batch_inputs, y_: batch_labels})
            #test
            a_,b_,c_,d_,e_,f_ = sess.run([logits,x, y_,loss,pred,gt], feed_dict={x: test_data, y_: test_label})
            if epoch%10==0:
                                
                print("epoch",epoch)
                print("train_loss",d)
                print("test_loss",d_)
                correct = np.count_nonzero(np.array(e) == np.array(f))
                accuracy = float(correct) / len(f) * 100  
                correct_ = np.count_nonzero(np.array(e_) == np.array(f_))
                accuracy_ = float(correct_) / len(f_) * 100
                print("train_acc",accuracy,"%")
                print("test_acc",accuracy_,"%")
                
                                        
        save_path = saver.save(sess, "./model/model.ckpt")
        print("save_path",save_path)
    sess.close()
        # Final embeddings are ready for you to use. 
        # Need to normalize for practical use



def test(astring,root='./validation/'):    
    tf.reset_default_graph()
    
    data, label = read_img(root=root, weight=128,height=128)
    x_test, y_test = messUpOrder(data=data, label=label)
   
    logits, x, y_, loss, train_op = buildCNN(mode=False)
    pred=tf.argmax(logits,axis=1)
    gt=tf.argmax(y_,axis=1)
   
    

    n_epoch = 1
    batch_size = 200
    
    with tf.Session() as sess:

        saver = tf.train.Saver()
        # Initializing all variables
        tf.initialize_all_variables().run()
        saver.restore(sess, "./model/model.ckpt")
        for epoch in range(n_epoch):
            print("epoch",epoch)
            batch_inputs, batch_labels = generate_batch(inputs=x_test, targets=y_test, batch_size=batch_size)
            p,g = sess.run([pred,gt], feed_dict={x: batch_inputs, y_: batch_labels})
            correct_ = np.count_nonzero(np.array(p) == np.array(g))
            accuracy_ = float(correct_) / len(p) * 100
            print("test_acc",accuracy_,"%")

    sess.close()
    # make your model give prediction for images from data_dir
    # the following code is just a placeholder
    return p, g
