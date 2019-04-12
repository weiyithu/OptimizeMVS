''' Single-image encoder model. 
'''
import numpy as np
import tensorflow as tf
import tflearn

def encoder(opt, image, r_):
        r = tf.reshape(r_, shape=[-1, opt.rand_dim])
        r = tflearn.layers.core.fully_connected(r, 256, activation='relu', weight_decay=1e-3, regularizer='L2')
        r = tflearn.layers.core.fully_connected(r, 768*3, activation='relu', weight_decay=1e-3, regularizer='L2') # [B*5, 768*3]
        r = tf.reshape(r, shape=[-1, 24, 32, 3]) # [B*5, 24, 32, 3]
        r = tflearn.layers.conv.conv_2d(r, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')  # [B*5, 24, 32, 32]
        r = tflearn.layers.conv.conv_2d(r, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2') # [B*5, 24, 32, 128]
 
        x=image
        x = tf.image.resize_images(image, [192, 256])
	#192 256
	x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x0=x
	x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	#96 128
	x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x1=x
	x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	#48 64
	x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x2=x
	x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	#24 32
	x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x = tf.concat([x, r], axis=3)

	x3=x
	x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	#12 16
	x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x4=x
	x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	#6 8
	x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x5=x
	x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x_additional=tflearn.layers.core.fully_connected(x,2048,activation='relu',weight_decay=1e-3,regularizer='L2')
	x_additional=tflearn.layers.core.fully_connected(x_additional,1024,activation='relu',weight_decay=1e-3,regularizer='L2')
	x_additional=tflearn.layers.core.fully_connected(x_additional,512*3,activation='linear',weight_decay=1e-3,regularizer='L2')
	x_additional=tf.reshape(x_additional,(-1,512,3))
	x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
	x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
	x=tf.nn.relu(tf.add(x,x5))
	x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
	x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
	x=tf.nn.relu(tf.add(x,x4))
	x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
	x=tf.nn.relu(tf.add(x,x3))
	x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.conv.conv_2d(x,3*2,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
	x=tf.reshape(x,(-1,32*24*2,3))
	x=tf.concat([x_additional,x],axis=1)
	x=tf.reshape(x,(-1,2048,3))
        return x

