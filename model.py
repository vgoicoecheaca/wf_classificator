from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, 
                                    LSTM,
                                    TimeDistributed,
                                    Input,
                                    MaxPool1D,
                                    RepeatVector,
                                    Flatten,
                                    UpSampling1D,
                                    Concatenate,
                                    Dropout,
                                    Reshape,
                                    Dense)

import tensorflow as tf

class ModelWF(Model):
    def __init__(self,config):
        super(ModelWF,self).__init__()
        self.sequence_size          = config("data","sequence_size","int")
        self.truncated_size         = config("data","truncated_size","int")
        self.l2                     = config("training","l2","float")
        self.activation             = config("training","activation",'str')
        self.act_class              = config("training","act_class",'str')
        self.act_reg                = config("training","act_reg",'str')

    def call(self,train=True):
        input                 = Input(shape=(self.sequence_size,1),name="input")
        base                  = self.base(input)                                                                   
        flatten               = Flatten()(base)
        dense                 = Dense(32,activation='relu')(flatten)
        out                   = Dense(1,activation='sigmoid',name='class')(dense)
                                                                                                            
        return Model(input,outputs=out)
    
    def base(self,x):
        conv                 = Conv1D(filters=2,kernel_size=1,name="conv0",activation=self.activation)(x)
        pool                 = MaxPool1D(2)(conv)     
        conv                 = Conv1D(filters=4,kernel_size=2,name="conv2",kernel_regularizer=l2(self.l2),activation=self.activation)(pool)
        pool                 = MaxPool1D(2)(conv)     
        conv                 = Conv1D(filters=8,kernel_size=2,name="conv3",kernel_regularizer=l2(self.l2),activation=self.activation)(pool)
        pool                 = MaxPool1D(2)(conv)     
        conv                 = Conv1D(filters=16,kernel_size=2,name="conv4",kernel_regularizer=l2(self.l2),activation=self.activation)(pool)
        pool                 = MaxPool1D(4)(conv)     
        conv                 = Conv1D(filters=32,kernel_size=2,name="conv5",kernel_regularizer=l2(self.l2),activation=self.activation)(pool)
        pool                 = MaxPool1D(4)(conv)
        conv                 = Conv1D(filters=64,kernel_size=2,name="conv6",kernel_regularizer=l2(self.l2),activation=self.activation)(pool)
        pool                 = MaxPool1D(4)(conv)

        return pool

