import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
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
x_train_bg, [y_train_class_bg,y_train_reg_bg]     = training_data_generator(0)
x_val_bg,   [y_val_class_bg,y_val_reg_bg]         = validation_data_generator(0)

x_train_max, [y_train_class_max,y_train_reg_max]  = training_data_generator(0)
x_val_max,   [y_val_class_max,y_val_reg_max]      = validation_data_generator(0)

# call the model, fit 
model = ModelWF(config) 
model_bg = model(True)
model_max = model(True)

lr = config("training","lr","float")
optimizer = Adam(learning_rate=lr)
model_bg.compile(optimizer=optimizer,loss=['binary_crossentropy'],metrics=['binary_accuracy'])
model_max.compile(optimizer=optimizer,loss=['binary_crossentropy'],metrics=['binary_accuracy'])

def fit_model(x_train,y_train,x_val,y_val,model,bg=False):
    print(y_train.shape,y_val.shape)
    model.fit(x=x_train,y=y_train,validation_data=(x_val,y_val) ,batch_size=config("data","batch_size","int"),
            epochs=config("training","epochs","int"),initial_epoch=0,
            callbacks=[EarlyStopping(monitor=config("training","monitor","str"),patience=config("training","patience","int"),mode=config("training","mode","str"))])
    model.save_weights("models/model_bg.h5" if bg else "models/model_max.h5")

models_wfs = [model_bg,model_max] 
for i,(x,y,xval,yval) in enumerate(zip([x_train_bg,x_train_max],[y_train_class_bg,y_train_class_max],[x_val_bg,x_val_max],[y_val_class_bg,y_val_class_max])):
    fit_model(x,y,xval,yval,models_wfs[i],bg=False if i==0 else True)

def load_all_models(n_models,models):
    all_models = list()
    for i in range(n_models):
        suf = "bg" if i==0 else "max"
        filename = 'models/model_' + suf + '.h5'
        models[i].load_weights(filename)
        all_models.append(models[i])
    return all_models

models = load_all_models(2,models_wfs)

def stacked_model(models):
    inputs, outputs = [],[]
    for i in range(len(models)):
        model = models[i]
        suf = "bg" if i==0 else "max"
        for layer in model.layers:
            layer.trainable = False
            layer._name = 'stacked_'+suf+"_"+layer.name 
        inputs.append(model.input)
        #inputs.append(model.get_layer("stacked_bg_dense"if i==0 else "stacked_max_dense_1").output)
        outputs.append(model.output)
    merge  = tf.keras.layers.Concatenate()(inputs)
    hidden = tf.keras.layers.Dense(32, activation='relu')(merge)
    output = tf.keras.layers.Dense(1, activation='softmax')(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

stck_model = stacked_model(models)

# fit the stacked model
x_train, [y_train_class,y_train_reg]     = training_data_generator(1)
x_val,   [y_val_class,y_val_reg]         = validation_data_generator(1)

X = [x_train for _ in range(len(stck_model.input))]
X_val = [x_val for _ in range(len(stck_model.input))]

from tensorflow.keras.utils import plot_model
plot_model(stck_model,to_file="stack.png",show_shapes=True)
history = stck_model.fit(x=X,y=y_train_class,validation_data=(X_val,y_val_class),epochs=5,verbose=1,
        callbacks=[EarlyStopping(monitor="val_categorical_accuracy",patience=config("training","patience","int"),mode=config("training","mode","str"))])

stck_model.save_weights("models/"+config("training","model","str"))
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

