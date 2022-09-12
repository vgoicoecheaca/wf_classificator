import warnings
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC

from config import Config
from data_handler import DataHandler
from model import ModelWF

config = Config()

DISABLE_GPU = config("training","disable_gpu","bool")

if DISABLE_GPU:
    try:
        # Disable all GPUS
        visible_devices = tf.config.get_visible_devices()
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

# path to data files
#train_sample              = ["wfs/7to8vov_60to200ma_wfs.txt","wfs/7to8vov_60to200ma_hits.txt","wfs/7to8vov_60to200ma_pars.txt"]
#val_sample                = ["wfs/7to8vov_60to200ma_wfs_val.txt","wfs/7to8vov_60to200ma_hits_val.txt","wfs/7to8vov_60to200ma_pars_val.txt"]

train_sample              = ["wfs/7to8vov_60to200ma_wfs.txt","wfs/7to8vov_60to200ma_hits.txt","wfs/7to8vov_60to200ma_pars.txt"]
training_data_generator   = DataHandler(train_sample,config)
#validaiton_data_generator = DataHandler(val_sample,config)

model = ModelWF(config) 
model = model(True)

print(model.summary())

optimizer = Adam(learning_rate=config("training","lr","float"))

model.compile(optimizer=optimizer,loss=["categorical_crossentropy","mae"])

#history = model.fit(x=training_data_generator,validation_data=validaiton_data_generator,
         #epochs=config("training","epochs","int"),initial_epoch=0)
history = model.fit(x=training_data_generator,
         epochs=config("training","epochs","int"),initial_epoch=0)

#model.save_weights("models/model_various.h5")
#plt.style.use('mystyle.mlstyle')
#styles = ['b-','r-','g-','b--','r--','g--']
#s      = 0
#for loss in history.history.keys():
#   if "loss" in loss:
#        plt.plot([i for i in range(config("training","epochs","int"))], history.history[loss],styles[s],label=loss)
#        s += 1
#plt.legend(loc='best')
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.grid(True)
#plt.savefig("loss.pdf")
#plt.show()
#plt.clf()

seq_len = 4000
model.load_weights("models/model_various.h5")
x, [y_class_true,y_reg_true] = validaiton_data_generator[0]
pred_class,pred_reg = model.predict(x)
idx = [i for i in range(5)]
for i in idx:
    plt.plot(x[i],color='blue')
    print("Pred Hits",np.argmax(pred_class[i]))
    print("True Hits",np.argmax(y_class_true[i]))
    print("Pred pos",pred_reg[i])
    print("True pos",y_reg_true[i])
    plt.vlines(x=pred_reg[i]*seq_len,ymin=0,ymax=1,color='orange',alpha=0.8,label="start reco")
    plt.vlines(x=y_reg_true[i]*seq_len,ymin=0,ymax=1,color='green',alpha=0.8,label="start truth")
    plt.text(x=seq_len*0.8,y=0.85,s="True hits = "+str(np.argmax(y_class_true[i])),color='green')
    plt.text(x=seq_len*0.8,y=0.80,s="Reco hits = "+str(np.argmax(pred_class[i])),color='orange')
    plt.legend(loc='best')
    plt.show()
exit()

