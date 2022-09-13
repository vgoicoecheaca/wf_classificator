import warnings
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
import keras_tuner

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
train_sample              = ["wfs/20to200ma_7to8vov_5p5snr_wfs.txt","wfs/20to200ma_7to8vov_5p5snr_hits.txt","wfs/20to200ma_7to8vov_5p5snr_pars.txt"]
val_sample                = ["wfs/20to200ma_7to8vov_5p5snr_val_wfs.txt","wfs/20to200ma_7to8vov_5p5snr_val_hits.txt","wfs/20to200ma_7to8vov_5p5snr_val_pars.txt"]

training_data_generator   = DataHandler(train_sample,config)
validation_data_generator = DataHandler(val_sample,config)

model = ModelWF(config) 

#HyperParameters
hp_dense = [int(i) for i in config("hyperparameters","units","str").split(",")]
hp_dense = [hp_dense[i:i+3] for i in range(0,len(hp_dense),3)]
hp_act   = config("hyperparameters","act_reg","str").split(",")
hp_lr    = [float(i) for i in config("hyperparameters","lr","str").split(",")]


def hp_build_model(hp):
    units      = [hp.Int("units",min_value=hp_dense[i][0],max_value=hp_dense[i][1],step=hp_dense[i][-1]) for i in range(len(hp_dense)) ]
    activation = hp.Choice("activation",hp_act)
    lr         = hp.Float("lr",min_value=hp_lr[0],max_value=hp_lr[1],sampling='log')

    hp_model   = model.HyperModel(*units,activation)

    optimizer = Adam(learning_rate=lr) 
    hp_model.compile(optimizer=optimizer,loss=["categorical_crossentropy","mae"])

    return hp_model

hp_build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.RandomSearch(
    hypermodel=hp_build_model,
    objective="val_loss",
    max_trials=3,
    executions_per_trial=3,
    overwrite=True,
    directory="hp",
    project_name="wf_classificator",
)

tuner.search(x=training_data_generator,validation_data=validation_data_generator,epochs=1)
models  = tuner.get_best_models(num_models=1)
best_model = models[0]
print(tuner.results_summary())
print(best_model.summary())

history = best_model.fit(x=training_data_generator,validation_data=validation_data_generator,epochs=2)

model.save_weights("models/model_5p5SNR_hp.h5")
plt.style.use('mystyle.mlstyle')
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
plt.savefig("loss.pdf")
plt.show()
plt.clf()

seq_len = 4000
#model.load_weights("models/model_various.h5")
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

