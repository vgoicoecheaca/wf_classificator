import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import AUC
from keras.callbacks import EarlyStopping

from config import Config
from data_handler import DataHandler
from model import ModelWF

config = Config()
plt.style.use('mystyle.mlstyle')
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
data_path       = "wfs/"
train_sample    = [data_path+i for i in config("training","data","str").split(",")]
val_sample      = [data_path+i for i in config("training","val_data","str").split(",")]

# call the data generators
training_data_generator   = DataHandler(train_sample,config)
validation_data_generator = DataHandler(val_sample,config)

#importing datasets
x_train, [y_train_class,y_train_reg]             = training_data_generator()
x_val,   [y_val_class,y_val_reg]                 = validation_data_generator()

# call the model
model = ModelWF(config) 
model = model(True)
print(model.summary())

lr = config("training","lr","float")
loss = CategoricalCrossentropy(from_logits=True)
optimizer = Adam(learning_rate=lr)

model.compile(optimizer=optimizer,loss=["categorical_crossentropy","mae"])

#train
history = model.fit(x=x_train,y=[y_train_class,y_train_reg],validation_data=(x_val,[y_val_class,y_val_reg]) ,batch_size=config("data","batch_size","int"),
         epochs=config("training","epochs","int"),initial_epoch=0,
         callbacks=[EarlyStopping(monitor=config("training","monitor","str"),patience=config("training","patience","int"),mode="min")])

model.save_weights("models/"+config("training","model","str"))
styles = ['b-','r-','g-','b--','r--','g--']
s      = 0
for loss in history.history.keys():
   if "loss" in loss:
        plt.plot(history.history[loss],styles[s],label=loss)
        s += 1
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("plots/loss.pdf")
plt.clf()

