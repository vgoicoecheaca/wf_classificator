import warnings
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
import keras_tuner
from keras.callbacks import EarlyStopping

from config import Config
from data_handler import DataHandler
from model import ModelWF

config      = Config()
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

training_data_generator   = DataHandler(train_sample,config)
validation_data_generator = DataHandler(val_sample,config)

#importing datasets
x_train, [y_train_class,y_train_reg]             = training_data_generator()
x_val,   [y_val_class,y_val_reg]                 = validation_data_generator()

model = ModelWF(config) 

#HyperParameters
hp_dense     = [int(i) for i in config("hyperparameters","units","str").split(",")]
hp_dense     = [hp_dense[i:i+3] for i in range(0,len(hp_dense),3)]
hp_act_reg   = config("hyperparameters","act_reg","str").split(",")
hp_act_hid   = config("hyperparameters","act","str").split(",")
hp_lr        = [float(i) for i in config("hyperparameters","lr","str").split(",")]
hp_l2        = [float(i) for i in config("hyperparameters","l2","str").split(",")]

def hp_build_model(hp):
    units             = [hp.Int("units",min_value=hp_dense[i][0],max_value=hp_dense[i][1],step=hp_dense[i][-1]) for i in range(len(hp_dense)) ]
    acts              = [hp.Choice("activation_reg_out",hp_act_reg),hp.Choice("activation_hidden",hp_act_hid)]
    lr                = hp.Float("lr",min_value=hp_lr[0],max_value=hp_lr[1],sampling='log')
    l2                = hp.Float("l2",min_value=hp_l2[0],max_value=hp_l2[1],sampling='log') 

    hp_model   = model.HyperModel(*units,*acts,l2)
    optimizer = Adam(learning_rate=lr) 
    hp_model.compile(optimizer=optimizer,loss=["categorical_crossentropy","mae"])

    return hp_model

hp_build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.RandomSearch(
    hypermodel=hp_build_model,
    objective=config("hyperparameters","objective","str"),
    max_trials=config("hyperparameters","max_trials","int"),
    executions_per_trial=config("hyperparameters","exec_per_trial","int"),
    overwrite=True,
    directory="hp",
    project_name="wf_classificator",
)

tuner.search(x=x_train,y=[y_train_class,y_train_reg],validation_data=(x_val,[y_val_class,y_val_reg]),
        epochs=config("hyperparameters","epochs","int"),
        callbacks = [EarlyStopping(monitor=config("hyperparameters","objective","str"),patience=config("hyperparameters","patience","int"),mode="min")])
)

models     = tuner.get_best_models(num_models=1)
best_model = models[0]

print(tuner.results_summary())
print(best_model.summary())

history = best_model.fit(x=training_data_generator,validation_data=validation_data_generator,epochs=config("training","epochs","int"),
            callbacks=[EarlyStopping(monitor=config("training","monitor","str"),patience=config("training","patience","int"),mode="min")])

best_model.save_weights("models/"+config("training","model","str"))

styles = ['b-','r-','g-','b--','r--','g--']
s      = 0
for loss in history.history.keys():
   if "loss" in loss:
        plt.plot([i for i in range(config("training","epochs","int"))], history.history[loss],styles[s],label=loss)
        s += 1
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("plots/loss.pdf")
plt.clf()
