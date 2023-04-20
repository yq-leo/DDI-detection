# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 20:53:45 2022
@author: QI YU
@email: yq123456leo@outlook.com
"""

import keras
from keras import layers, models, callbacks
import time

def ConvLSTM(input_shape, num_classes):
    input_layer = layers.Input(shape = input_shape, name = 'input')
    
    conv1 = layers.Conv1D(name = 'conv1', filters = 32, kernel_size = 8, 
                          strides = 2, activation = 'relu', padding = 'same')(input_layer)
    bn1 = layers.BatchNormalization(name = 'bn1')(conv1) 
    pool1 = layers.MaxPooling1D(name = 'pool1', pool_size = 2, strides = 2, padding = 'same')(bn1)
    
    conv2 = layers.Conv1D(name = 'conv2', filters = 32, kernel_size = 4, 
                          strides = 2, activation = 'relu', padding = 'same')(pool1) 
    bn2 = layers.BatchNormalization(name = 'bn2')(conv2) 
    pool2 = layers.MaxPooling1D(name = 'pool2', pool_size = 2, strides = 2, padding = 'same')(bn2) 
    
    conv3 = layers.Conv1D(name = 'conv3', filters = 32, kernel_size = 4, 
                          strides = 1, activation = 'relu', padding = 'same')(pool2) 
    bn3 = layers.BatchNormalization(name = 'bn3')(conv3)
    
    # Global Layers
    gmaxpl = layers.GlobalMaxPooling1D(name = 'gmaxpl')(bn3) 
    gmeanpl = layers.GlobalAveragePooling1D(name = 'gmeanpl')(bn3) 
    mergedlayer = layers.concatenate([gmaxpl, gmeanpl], axis=1)
    
    fl = layers.Flatten()(mergedlayer)
    rv = layers.RepeatVector(300)(mergedlayer) 
    lstm1 = layers.LSTM(128, return_sequences = True, name = 'lstm1')(bn3)
    do3 = layers.Dropout(0.5, name = 'do3')(lstm1)
    
    lstm2 = layers.LSTM(64, name = 'lstm2')(do3) 
    do4 = layers.Dropout(0.2, name = 'do4')(lstm2) 
    
    flat = layers.Flatten(name = 'flat')(do4) 
    output_layer = layers.Dense(num_classes, activation = 'softmax', name = 'output')(flat) 
    
    model = models.Model(inputs = input_layer, outputs = output_layer)  
    return model

def model_train(model, number_epoch, train_x, train_y):   
    optimizer = keras.optimizers.RMSprop(lr = 0.01, rho = 0.9, epsilon = None, decay = 0.0)

    # a stopping function should the validation loss stop improving
    #earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

    model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = optimizer)
    #plot_model(model, show_shapes=True, to_file='ConvLSTM.png')   
    tensorboardRNN = callbacks.TensorBoard(log_dir = "RNN_logs/{}".format(time()))
    
    #for i in range(number_epoch):
    history1 = model.fit(train_x, train_y, validation_split = 0.1, callbacks = [tensorboardRNN], batch_size = 32, epochs = int(number_epoch), shuffle = False)
    #model.reset_states()        
    
    print(model.summary())

    return model, history1

if __name__ == "__main__":
    input_shape = (400, 1)
    num_classes = 2
    
    model = ConvLSTM(input_shape, num_classes)
    #model, _ = model_train(model, 20)