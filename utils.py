#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:51:20 2020
@author: dev
"""

import numpy as np
from cnn.layers import convolution_layer,maxpool_layer,softmax,categorical_crossentropy

def create_filters(size,scale=1.0):
    stndev=scale/np.sqrt(np.prod(size))
    return np.random.normal(loc=0,scale=stndev,size=size)

def create_weight(size):
    return np.random.standard_normal(size=size)*0.01

def nan_arguments(array):
    arg=np.nanargmax(array)
    index=np.unravel_index(arg,array.shape)
    return index

def predict(filter1,filter2,weight1,weight2,bias1,bias2,bias3,bias4,conv_stride=1,pool_filt=2,pool_stride=2):
    conv1=convolution_layer(image,filter1,bias1,conv_stride)
    conv1[conv1<=0]=0
    conv2=convolution_layer(conv1,filter2,bias2,conv_stride)
    conv2[conv2<=0] = 0 
    pooled=maxpool_layer(conv2,pool_filt,pool_stride)
    
    (no_filtr,diamention,_)=pooled.shape
    finl=pooled.reshape(no_filtr*diamention*diamention,1)
    
    z=weight1.dot(finl)+bias3
    
    z[z<=0]=0
    
    output_layer=weight2.dot(z)+bias4
    
    probability=softmax(output_layer)
    
    return np.argmax(probability),np.max(probability)


    
    
    
    