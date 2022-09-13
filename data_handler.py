'''Data Handler structure credits to https://github.com/Socret360/object-detection-in-keras'''


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import tensorflow as tf
tf.autograph.set_verbosity(8)
from tensorflow.data import Dataset as ds
import csv 

from utils.augmentation_utils import flip_sequence,window_slicing
import matplotlib.pyplot as plt
plt.style.use('mystyle.mlstyle')

class DataHandler(tf.keras.utils.Sequence):
    def __init__(self,samples,config,augment=True):
        self.sequence_size           = config("data","sequence_size","int")
        self.normalization           = config("data","normalization","int")
        self.n_classes               = config("data","n_classes","int") + 1
        self.batch_size              = config("data","batch_size","int")
        self.truncated_size          = config("data","truncated_size","int")
        self.steps                   = np.arange(0,self.sequence_size,self.truncated_size)

        self.test                    = config('training','test','bool')

        self.flipping                = config('augmentation','flipping','float')
        self.sliding                 = config('augmentation','sliding','float')

        self.sequences  = self.load_sequences(samples[0])
        self.locations  = self.load_locations(samples[1])
        self.parameters = self.load_parameters(samples[2])

        self.indices   = range(0,self.n_entries)

        self.augment       = augment
        self.augmentations = [flip_sequence.flip_sequence(p=self.flipping),
                              window_slicing.window_slicing(p=self.sliding)]              # window sliding, should change the name... 

        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        np.random.shuffle(self.index)

    def __augment(self, sequence, locations):
        augmented_sequence,augmented_locations = sequence,locations
        for aug in self.augmentations:
            augmented_sequence, augmented_locations = aug(augmented_sequence,augmented_locations)

        if self.augment:
            return augmented_sequence, augmented_locations
        else:
            return sequence,locations

    def __getitem__(self,index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X, y_class, y_reg  = self.__get_data(batch)   
        X, y_class, y_reg  = tf.convert_to_tensor(X), tf.convert_to_tensor(y_class), tf.convert_to_tensor(y_reg)

        ##for testing only
        if self.test:
            for i in np.random.randint(low=0,high=self.batch_size,size=10):
                plt.plot(X[i])
                plt.vlines(x=y_reg[i]*self.sequence_size,ymin=0,ymax=1,color='red')
                print("WF sart at ", y_reg[i]*self.sequence_size)
                print(y_reg[i])
                print("WF with ",np.argwhere(y_class[i]==1),"hits")
                print(y_class[i]) 
                plt.show()
            exit()

        return X, [y_class,y_reg]

    def __get_data(self,batch):
        X                   = []
        y_reg,y_class       = [],[]
        self.augmented_locs = []
        self.pars           = []

        for batch_idx in batch:
            sequence , locations = self.sequences[batch_idx], self.locations[batch_idx] 
            if 0 in locations: locations = []                                                   # make sure 0 hits are empty so code below recognizes them accordingly (len)
            sequence, locations = self.__augment(sequence,locations)

            #define labels as the start of the hit (time, regression) and the ammount of hits (int, classification) 
            y_true_class = np.zeros(self.n_classes,dtype='int32')
            #assert len(locations) <= self.n_classes, "Found WF with n_hits > n_classes"

            y_true_class[len(locations) if len(locations) <self.n_classes else -1] = 1                                            # one-hot encoded
            y_true_reg = locations[0] / self.sequence_size if np.any(locations) else 0

            sequence = np.negative(sequence)
            if self.normalization == 1: 
                sequence = (sequence - np.average(sequence)) / np.std(sequence)
            else:
                sequence = (sequence - np.min(sequence)) / (np.max(sequence)-np.min(sequence))
            
            X.append(np.expand_dims(np.array(sequence,dtype=float),axis=-1))                  
            y_reg.append(y_true_reg)
            y_class.append(y_true_class)
            self.augmented_locs.append(locations)
            self.pars.append(self.parameters[batch_idx]) 

        return X, np.asarray(y_class), np.asarray(y_reg) 

    def load_sequences(self,sample): 
        df            = pd.read_csv(sample,header=None,delimiter=" ")        
        df            = df.to_numpy().reshape(df.shape[0]*int(self.sequence_size/self.truncated_size),self.truncated_size)              # truncate waveforms into chunks
        self.n_entries  = df.shape[0]
        self.input_size = df.shape

        return df 
    
    def load_locations(self,sample):
        hits_array = np.array([],dtype=object)
        with open(sample, newline='',) as f:
            hits_array = np.append(hits_array,list(csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)))

        #split the locations into their corresponding truncated wavforms
        hits_reshaped = np.zeros(self.n_entries,dtype=object)
        n = 0 
        for hits in hits_array:
            for i,step in enumerate(self.steps):
                if step != self.steps[-1]:
                    h = np.array(hits)[(hits>step) & (hits<=self.steps[i+1])]
                else:
                    h = np.array(hits)[(hits>=step)]
                hits_reshaped[n] = h.tolist() - step if len(h) != 0  else []
                n += 1 

        return tf.ragged.constant(hits_reshaped)

    def load_parameters(self,sample):  return np.loadtxt(sample) 
        
    def get_locs(self): return np.asarray(self.augmented_locs)
    def get_pars(self): return self.pars
