#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:47:24 2020

@author: dev
"""

#%%
import numpy as np
from cnn.utils import *
from cnn.layers import categorical_crossentropy,softmax,convolution_layer,convolution_backward,maxpool_layer,maxpool_backward
import pickle
import os
from tqdm import tqdm
import numpy.random as random
import cv2
import glob
from tensorflow.keras.preprocessing.image import img_to_array

def model(image,labels,params,conv_stride,pool_filtr,pool_stride):
    [filtr1,filtr2,weight1,weight2,bias1,bias2,bias3,bias4]=params
    
    
    conv1=convolution_layer(image,filtr1,bias1,conv_stride)
    #print(conv1)
    conv1[conv1<=0]=0
    conv2=convolution_layer(conv1,filtr2,bias2,conv_stride)
    conv2[conv2<=0]=0
    
    pooled=maxpool_layer(conv2,pool_filtr,pool_stride)
    (no_filtr,dim,_)=pooled.shape
    finl=pooled.reshape((no_filtr*dim*dim,1))
    z=weight1.dot(finl)+bias3
    z[z<=0]=0
    output_layer=weight2.dot(z)+bias4
    prob=softmax(output_layer)
    
    loss=categorical_crossentropy(prob,labels)
    
    der_out=prob-labels
    dw4=weight2.dot(z.T)
    db4 = np.sum(der_out, axis = 1).reshape(bias4.shape)
    dz = weight2.T.dot(der_out)
    dz[z<=0] = 0 
    dw3 = dz.dot(finl.T)
    db3 = np.sum(dz, axis = 1).reshape(bias3.shape)
    dfc = weight1.T.dot(dz) 
    dpool = dfc.reshape(pooled.shape)
    
    dconv2 = maxpool_backward(dpool, conv2, pool_filtr, pool_stride) 
    dconv2[conv2<=0] = 0
    
    dconv1,df2,db2=convolution_backward(dconv2,conv1,filtr2,conv_stride)
    
    dconv1[conv1<=0] = 0 
    dimage, df1, db1 = convolution_backward(dconv1, image, filtr1, conv_stride)
    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4] 
    
    return grads, loss
    


def AdamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
    
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
      #  print("batch============",batch)  
    X = batch[:,0:-1] 
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:,-1] 
    #print("y in batch  =  ",Y)
    cost_ = 0
    batch_size = len(batch)
    
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    dw3 = np.zeros(w3.shape)
    dw4 = np.zeros(w4.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)
    
    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)
    
    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)
 
    for i in range(batch_size):
    # =============================================================================
    #         print("*Y shape =  ",Y[i])
    #         print("i == ",i)
    # =============================================================================
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)
        
        grads, loss = model(x, y, params, 1, 2, 2)
           # print("loss = ",loss)
        [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads
        
        df1+=df1_
        db1+=db1_
        df2+=df2_
        db2+=db2_
        dw3+=dw3_
        db3+=db3_
        dw4+=dw4_
        db4+=db4_
        
        cost_+= loss
    
    # Parameter Update  
    
    v1 = beta1*v1 + (1-beta1)*df1/batch_size # momentum update
    s1 = beta2*s1 + (1-beta2)*(df1/batch_size)**2 # RMSProp update
    f1 -= lr * v1/np.sqrt(s1+1e-7) # combine momentum and RMSProp to perform update with Adam
    
    bv1 = beta1*bv1 + (1-beta1)*db1/batch_size
    bs1 = beta2*bs1 + (1-beta2)*(db1/batch_size)**2
    b1 -= lr * bv1/np.sqrt(bs1+1e-7)
       
    v2 = beta1*v2 + (1-beta1)*df2/batch_size
    s2 = beta2*s2 + (1-beta2)*(df2/batch_size)**2
    f2 -= lr * v2/np.sqrt(s2+1e-7)
                   
    bv2 = beta1*bv2 + (1-beta1) * db2/batch_size
    bs2 = beta2*bs2 + (1-beta2)*(db2/batch_size)**2
    b2 -= lr * bv2/np.sqrt(bs2+1e-7)
    
    v3 = beta1*v3 + (1-beta1) * dw3/batch_size
    s3 = beta2*s3 + (1-beta2)*(dw3/batch_size)**2
    w3 -= lr * v3/np.sqrt(s3+1e-7)
    
    bv3 = beta1*bv3 + (1-beta1) * db3/batch_size
    bs3 = beta2*bs3 + (1-beta2)*(db3/batch_size)**2
    b3 -= lr * bv3/np.sqrt(bs3+1e-7)
    
    v4 = beta1*v4 + (1-beta1) * dw4/batch_size
    s4 = beta2*s4 + (1-beta2)*(dw4/batch_size)**2
    w4 -= lr * v4 / np.sqrt(s4+1e-7)
    
    bv4 = beta1*bv4 + (1-beta1)*db4/batch_size
    bs4 = beta2*bs4 + (1-beta2)*(db4/batch_size)**2
    b4 -= lr * bv4 / np.sqrt(bs4+1e-7)
    
    cost_ = cost_/batch_size
    cost.append(cost_)
    
    params = [f1, f2, w3, w4, b1, b2, b3, b4]
    
    return params, cost


def train(num_classes = 5, lr = 0.01, beta1 = 0.95, beta2 = 0.99, img_dim = 28, img_depth = 1, f = 5, num_filt1 = 8, num_filt2 = 8, batch_size = 16, num_epochs = 10, save_path = '/home/dev/Test/Face4.pkl'):

    # training data
    #m =50000
    #X = extract_data('train-images-idx3-ubyte.gz', m, img_dim)
    #y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
        
    data = []
    labels = []
    Imgdata="/home/dev/Pictures/agimg"
    IMAGE_DIMS = (28, 28, 3)
    imagePaths = glob.glob(Imgdata + '/*jpeg')
    random.seed(42)
    random.shuffle(imagePaths)
    #f= open("/home/ajit/Documents/ImageProcessing/MultiClass/labels.txt","w+")
    for imagePath in imagePaths:
        image = cv2.imread (imagePath)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        image = img_to_array(image)
        image=image.reshape(IMAGE_DIMS[1]*IMAGE_DIMS[0])
        data.append(image)
        imgpath= imagePath.split(os.path.sep)[-1]
        label = imgpath.split('_')
        label=float(label[0])
        labels.append(label)
    data = np.array(data, dtype='float64') / 255.0    
 
    X=data
  
    labels=np.array(labels)
    labels=labels.reshape(labels.shape[0],1)
    y_dash=labels
    print("ydash ",y_dash.shape)
    print("X  ",X.shape)
    
    #X-= int(np.mean(X))
    #X/= int(np.std(X))
    
    train_data = np.hstack((X,y_dash))
    np.random.shuffle(train_data)

    f1, f2, w3, w4 = (num_filt1 ,img_depth,f,f), (num_filt2 ,num_filt1,f,f), (128,800), (num_classes, 128)
    f1 = create_filters(f1)
    f2 = create_filters(f2)
    w3 = create_weight(w3)
    w4 = create_weight(w4)

    b1 = np.zeros((f1.shape[0],1))
    b2 = np.zeros((f2.shape[0],1))
    b3 = np.zeros((w3.shape[0],1))
    b4 = np.zeros((w4.shape[0],1))

    params = [f1, f2, w3, w4, b1, b2, b3, b4]
    
    cost = []
   
    print("LR:"+str(lr)+", Batch Size:"+str(batch_size))

    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x,batch in enumerate(t):
            params, cost = AdamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
            t.set_description("Cost: %.2f" % (cost[-1]))
            
    to_save = [params, cost]
    
    with open(save_path, 'wb') as file:
        pickle.dump(to_save, file)
        
    return cost
        
#%%
# =============================================================================
# import wikipedia
# from gtts import gTTS
# =============================================================================
import pickle
import cv2
from cnn.main import *
from cnn.utils import *
import matplotlib.pyplot as plt
if __name__ == '__main__':
    save_path = '/home/dev/Test/Face4.pkl'
    cost = train(save_path = save_path)
    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    # print("f1 shape==  ",f1)
    img=cv2.imread("/home/dev/Pictures/index.jpeg")

    #plt.imshow(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   # ret,img=cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)
    #plt.imshow(img)
    img=cv2.resize(img,(28,28))
    img=img.reshape(1,28,28)  
    pred, prob = predict(img, f1, f2, w3, w4, b1, b2, b3, b4)
    print(pred)
    print(prob)
   
    
    
# =============================================================================
# 
#     data=wikipedia.summary(pred,sentences=5)
#     
#     myobj = gTTS(text=data, lang='en-in', slow=False) 
#     
#     myobj.save("/home/dev/Pictures/tom11.mp3") 
# =============================================================================
#%%
    
if __name__ == '__main__':
    
    save_path = "/home/dev/Test/Face4.pkl";
    
    cost = train(save_path = save_path)

    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params    
    
    print("***********module saved successfully************")
        