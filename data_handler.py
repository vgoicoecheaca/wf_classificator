'''Data Handler structure credits to https://github.com/Socret360/object-detection-in-keras'''

#import warnings
#warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
tf.autograph.set_verbosity(8)
from tensorflow.data import Dataset as ds
import csv 

from utils.augmentation_utils import flip_sequence,window_slicing
import matplotlib.pyplot as plt
plt.style.use('mystyle.mlstyle')

class DataHandler():
    def __init__(self,samples,config,augment=True):
        self.normalization           = config("data","normalization","str")
        self.batch_size              = config("data","batch_size","int")
        self.sequence_size           = config("data","sequence_size","int")
        self.truncated_size          = config("data","truncated_size","int")

        self.test                    = config('training','test','bool')

        self.flipping                = config('augmentation','flipping','float')
        self.sliding                 = config('augmentation','sliding','float')

        self.sequences  = self.load_sequences(samples[0])
        self.steps                   = np.arange(0,self.sequence_size,self.truncated_size)
        self.locations  = self.load_locations(samples[1])
        self.parameters = self.load_parameters(samples[2])

        self.augment       = augment
        self.augmentations = [flip_sequence.flip_sequence(p=self.flipping),
                              window_slicing.window_slicing(p=self.sliding)]              # window sliding, should change the name... 
    def __call__(self,lim):
        # process each sequence, create labels
        self.lim          = lim
        self.x            = np.zeros((len(self.sequences),self.sequence_size),dtype=float)
        #self.y_true_class = np.zeros((len(self.sequences),4 if self.lim==None else 1),dtype=float)
        self.y_true_class = np.zeros((len(self.sequences),2),dtype=float)
        self.y_true_reg   = np.zeros((len(self.sequences),1),dtype=float)
        for i,(seq,locs) in enumerate(zip(self.sequences,self.locations)):
            if 0 in locs: locs = []                                                   # make sure 0 hits are empty so code below recognizes them accordingly (len)
            self.x[i], locs = self.__augment(np.negative(self.sequences[i]),locs)
            if self.lim != None:
                if self.lim==0:
                    self.y_true_class[i] = 0 if len(locs)==0 else 1               
                else:
                    self.y_true_class[i] = 1 if len(locs)<=2 else 0
            else:
                self.y_true_class[i,len(locs) if len(locs) <=2 else -1] = 1 
            self.y_true_reg[i] = locs[0] / self.sequence_size if np.any(locs) else 0
            if self.normalization == "av": self.x[i] = (self.x[i] - np.average(self.x[i])) / np.std(self.x[i])
            if self.normalization == 'minmax': self.x[i] = (self.x[i] - np.min(self.x[i])) / (np.max(self.x[i]) - np.min(self.x[i]))
                                                                                                                                                                   
        if self.test:
            self.show_wf_and_label(5)

        return self.x, [self.y_true_class,self.y_true_reg]

    def __augment(self, sequence, locations):
        augmented_sequence,augmented_locations = sequence,locations
        for aug in self.augmentations:
            augmented_sequence, augmented_locations = aug(augmented_sequence,augmented_locations)

        if self.augment:
            return augmented_sequence, augmented_locations
        else:
            return sequence,locations

    def show_wf_and_label(self,n):
            print(self.y_true_reg.shape)
            for i in np.random.randint(low=0,high=self.batch_size,size=n):
                plt.plot(self.x[i])
                print(self.y_true_reg[i])
                plt.vlines(x=self.y_true_reg[i]*self.sequence_size,ymin=0,ymax=np.max(self.x[i]),color='red')
                print("WF sart at ",self.y_true_reg[i]*self.sequence_size)
                print(self.y_true_reg[i])
                print("WF with label",self.y_true_class[i])
                print(self.y_true_reg[i]) 
                plt.savefig("plots/fig"+str(i)+".png")
                plt.clf()
            plt.plot(self.x[10])
            plt.show()
            exit()

    def load_sequences(self,sample): 
        df            = pd.read_csv(sample,header=None,delimiter=" ")        
        df            = df.to_numpy().reshape(df.shape[0]*int(self.sequence_size/self.truncated_size),self.truncated_size)              # truncate waveforms into chunks
        self.n_entries      = df.shape[0]
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
        
    def get_locs(self): return self.locations.numpy()
    def get_pars(self): return self.parameters
