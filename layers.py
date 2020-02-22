#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:14:43 2020

@author: dev
"""

#%%
import numpy as np
#from cnn.utils import nan_arguments

#%%
def nan_arguments(array):
    arg=np.nanargmax(array)
    index=np.unravel_index(arg,array.shape)
    return index
def convolution_layer(image,filtr,bias,stride):
    
    no_filt,no_chnnl_filt,filtr,_= filtr.shape
    no_chnnl,input_dim,_=image.shape
    
    out_dim=int((input_dim-filtr)/stride)+1
    
    assert no_chnnl_filt==no_chnnl,"diamention of image must match diamention of filter"
    
    out=np.zeros((no_filt,out_dim,out_dim))
    
    for curr_f in range(no_filt):
        curr_y = out_y = 0
        while curr_y + filtr <= input_dim:
            curr_x = out_x = 0
            while curr_x + filtr <= input_dim:
                
                out[curr_f,out_y,out_x]=np.sum(filt[curr_f] * image[:,curr_y:curr_y+filtr, curr_x:curr_x+filtr]) + bias[curr_f]
                curr_x +=stride
                out_x+=1
            curr_y +=stride
            out_y +=1
    
    return out

def maxpool_layer(image,filtr=2,stride=2):
    no_channel,height,width=image.shape
    h=int((height-filtr)/stride)+1
    w=int((width-filtr)/stride)+1
    
    downsampled=np.zeros((no_channel,h,w))
    for i  in range(no_channel):
        curr_y=out_y=0
        while curr_y+filtr <=height:
            curr_x=out_x=0
            while curr_x+filtr<=width:
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+filtr, curr_x:curr_x+filtr])
                curr_x+=stride
                out_x+=1
            curr_y+=stride
            curr_y+=1
    return downsampled


def softmax(x):    
    out=np.exp(x)
    return out/np.sum(out)

def categorical_crossentropy(probability,label):    
    return -np.sum(label*np.log(probability))


def convolution_backward(dconv_prev,conv_in,filtr,stride):
    no_channel,no_filtr,filtr,_=filtr.shape
    (_,oris_dim,_)=conv_in.shape
    dout=np.zeros(conv_in.shape)
    dfilt=np.zeros(filtr.shape)
    dbias=np.zeros(no_filtr,1)
    
    for curr_filt in range (no_filtr):
        curr_y=out_y=0
        while curr_y+filtr<=oris_dim:
            curr_x=out_x=0
            while curr_x+filtr<=oris_dim:
                dfilt[curr_filt] += dconv_prev[curr_filt, out_y, out_x] * conv_in[:, curr_y:curr_y+filtr, curr_x:curr_x+filtr]
                dout[:, curr_y:curr_y+filtr, curr_x:curr_x+filtr] += dconv_prev[curr_filt, out_y, out_x] * filtr[curr_filt] 
                curr_x+=stride
                out_x+=1
            curr_y+=stride
            out_y+=1
        dbias[curr_filt]=np.sum(dconv_prev[curr_filt])
        
    return dout,dfilt,dbias

def maxpool_backward(dpool,oris,filtr,stride):
    no_chnnl,oris_dim,_=oris.shape
    dout=np.zeros(oris.shape)
    
    for curr_chnnl in no_chnnl:
        curr_y=out_y=0
        
        while curr_y+filtr<=oris_dim:
            curr_x=out_x=0
            while curr_x+filtr<=oris_dim:
                (a, b) = nan_arguments(oris[curr_chnnl, curr_y:curr_y+filtr, curr_x:curr_x+filtr])
                dout[curr_chnnl, curr_y+a, curr_x+b] = dpool[curr_chnnl, out_y, out_x]
                curr_x+=stride
                out_x+=1
            curr_y+=stride
            curr_y+=1
    return dout


        
        
        
    
            
            
            
            
            

    